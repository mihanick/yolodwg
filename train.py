# rewriting of 
# https://github.com/WongKinYiu/PyTorch_YOLOv4
import argparse
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import test #calculate mAP

from models import Darknet
from utils.autoanchor import check_anchors
from datasets import create_dataloader

from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, fitness_p, fitness_r, fitness_ap50, fitness_ap, fitness_f, strip_optimizer, get_latest_run,\
    check_dataset, check_file, check_git_status, check_img_size, set_logging

from utils.google_utils import attempt_download
from utils.loss import compute_loss
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import ModelEMA, select_device

logger = logging.getLogger(__name__)

def train(hyp, opt, device, tb_writer=None):
    logger.info(f'Hyperparameters {hyp}')
    save_dir = Path(opt.save_dir)
    epochs = opt.epochs
    image_size = opt.img_size
    batch_size = opt.batch_size
    total_batch_size = opt.batch_size
    weights = opt.weights

    # Directories
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    last = weights_dir / 'last.pt'
    best = weights_dir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'

    init_seeds(42)

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

    train_path = data_dict['train']    
    val_path = data_dict['val']

    # number of classes and class names
    nc = int(data_dict['nc'])
    names = data_dict['names']

    if opt.single_cls:
        nc = 1
        names = ['item']
    assert len(names) == nc, "Number of names found in dataset should be equal to number of classes"

    #Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        attempt_download(weights) # download weights if not found locally
        chkpt = torch.load(weights, map_location=device) # load checkpoint
        model = Darknet(opt.cfg).to(device)
        state_dict = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()} #magic!
        model.load_state_dict(state_dict, strict=False)
    else:
        model = Darknet(opt.cfg).to(device)
    
    # Optimizer
    nbs = 64 # nominal batch size
    accumulate = max(round(nbs/ total_batch_size), 1) # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs # scale weight decay

    pg0, pg1, pg2 = [], [], [] #optimizer parameter groups

    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)
        elif 'Conv2d.weight' in k or 'm.weight' in k or 'w.weight' in k:
            pg1.append(v)
        else:
            pg0.append(v)
    
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999)) #adjust beta1
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    del(pg0, pg1, pg2)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    best_fitness_p = 0.0
    best_fitness_r = 0.0
    best_fitness_ap50 = 0.0
    best_fitness_ap = 0.0
    best_fitness_f = 0.0

    if pretrained:
        # Optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness_p = chkpt['best_fitness_p']
            best_fitness_r = chkpt['best_fitness_r']
            best_fitness_ap50 = chkpt['best_fitness_ap50']
            best_fitness_ap = chkpt['best_fitness_ap']
            best_fitness_f = chkpt['best_fitness_f']

        # Results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results']) # write results.txt
        
        # Epochs
        start_epoch = chkpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, 'training has finished nothing to resume'
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, chkpt['epoch'], epochs))
            epochs += chkpt['epoch']  # finetune additional epochs
        
        del chkpt, state_dict

    # Image sizes
    gs = 64  #int(max(mode.stride)) #grid size max stride
    img_size = opt.img_size

    # EMA
    ema = ModelEMA(model)

    # DataLoaders
    dataloader, dataset = create_dataloader(
        path=train_path,
        image_size=image_size,
        batch_size=batch_size,
        stride=gs,
        opt=opt,
        hyp=hyp, 
        augment=True,
        cache_images=opt.cache_images)
    
    max_label_class = np.concatenate(dataset.labels, 0)[:, 0].max()
    n_batches = len(dataloader)
    assert max_label_class < nc, 'Label class should not exceed number of classes'


    ema.updates = start_epoch * n_batches // accumulate # set EMA updates
    val_loader, _ = create_dataloader(
        path=val_path,
        image_size=image_size,
        batch_size=batch_size,
        stride=gs,
        opt=opt,
        hyp=hyp, 
        cache_images=opt.cache_images and not opt.notest,
        workers=opt.workers
    )

    if not opt.resume:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0]) #classes
        
        # TODO: Was commented out 
        plot_labels(labels, save_dir=save_dir)
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)
    
    # Model parameters
    hyp['cls'] *= nc / 80. #scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0 # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * n_batches), 1000) #number of warmup iterations
    maps = np.zeros(nc) #mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)

    logger.info('Image size %g\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (image_size, dataloader.num_workers, save_dir, epochs))

    torch.save(model, weights_dir / 'init.pt')

    # enumerate epochs
    for epoch in range(start_epoch, epochs):
        model.train()

        # Update image weights ()
        # magic?

        mloss = torch.zeros(4, device=device) # mean_losses

        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total = n_batches) # progressbar
        
        optimizer.zero_grad()

        # enumerate batches
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + n_batches * epoch #number of integrated batches since train start

            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            # TODO: Magic?????

            # TODO: Multi-scale

            # Forward
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(device), model)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer) #optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # Plot
            if ni < 3:
                f = save_dir / f'tarin_batch{ni}.jpg'
                plot_images(images=imgs, targets=targets, paths=paths, fname=f)

                # TODO: Show image in tensorboard
                #if tb_writer:
                #    tb_writer.add_image(f, result, dataformats='HWC', global_step = epoch)
                #    tb_writer.add_graph(model, imgs) # add model to tensorboard

            # end batch

        # Scheduler   
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()
    
        final_epoch = epoch + 1 == epochs

        if not opt.notest or final_epoch:
            if epoch >=3:
                results, maps, times = test.test(
                    opt.data,
                    batch_size=batch_size,
                    imgsz=image_size,
                    model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                    single_cls=opt.single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=final_epoch,
                    log_imgs=0
                )

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

        # Log
        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
        for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        fi_p = fitness_p(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        fi_r = fitness_r(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        fi_ap50 = fitness_ap50(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        fi_ap = fitness_ap(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if (fi_p > 0.0) or (fi_r > 0.0):
            fi_f = fitness_f(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        else:
            fi_f = 0.0
        if fi > best_fitness:
            best_fitness = fi
        if fi_p > best_fitness_p:
            best_fitness_p = fi_p
        if fi_r > best_fitness_r:
            best_fitness_r = fi_r
        if fi_ap50 > best_fitness_ap50:
            best_fitness_ap50 = fi_ap50
        if fi_ap > best_fitness_ap:
            best_fitness_ap = fi_ap
        if fi_f > best_fitness_f:
            best_fitness_f = fi_f

        # Save model
        if (not opt.nosave) or (final_epoch and not opt.evolve): # whether to save
            with open(results_file, 'r') as f:
                chkpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'best_fitness_p': best_fitness_p,
                            'best_fitness_r': best_fitness_r,
                            'best_fitness_ap50': best_fitness_ap50,
                            'best_fitness_ap': best_fitness_ap,
                            'best_fitness_f': best_fitness_f,
                            'training_results': f.read(),
                            'model': ema.ema.module.state_dict() if hasattr(ema, 'module') else ema.ema.state_dict(),
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            'wandb_id': None}
            torch.save(chkpt, last)
            if best_fitness == fi:
                torch.save(chkpt, best)
        
    # Strip optimizers
    n = opt.name if opt.name.isnumeric() else ''
    fresults, flast, fbest = save_dir / f'results{n}.txt', weights_dir / f'last{n}.pt', weights_dir / f'best{n}.pt'
    for f1, f2 in zip([weights_dir / 'last.pt', weights_dir / 'best.pt', results_file], [flast, fbest, fresults]):
        if f1.exists():
            os.rename(f1, f2)  # rename
            if str(f2).endswith('.pt'):  # is *.pt
                strip_optimizer(f2)  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None  # upload

    # Finish
    plot_results(save_dir=save_dir)  # save as results.png
    logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/dwg.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', '--img', '--img-size', type=int, default=512, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')

    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    if opt.resume:
        chkpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.paths.isfile(chkpt), 'Error: --resume checkpoint does not exist'
        with open(Path(chkpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        opt.cfg, opt.weights, opt.resume = '', chkpt, True
        logger.info('Resuming training from %s' % chkpt)
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # TODO: DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    
    # Train
    logger.info(opt)
    logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
    tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard

    train(hyp, opt, device, tb_writer)
