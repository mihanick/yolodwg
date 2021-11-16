import os
from numpy.lib.arraysetops import isin
from tqdm import tqdm
import time
import json
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from dataset import DwgDataset, EntityDataset

from models.DwgKeyPointsModel import DwgKeyPointsModel
from models.DwgKeyPointsResNet50 import DwgKeyPointsResNet50
from models.DwgKeyPointsYolov4 import DwgKeyPointsYolov4
from models.DwgKeyPointRcnn import DwgKeyPointsRcnn
from models.torch_utils import bbox_ious
from models.utils import bbox_iou, nms_conf_suppression

from plot import plot_batch_grid, plot_loader_predictions
from loss import Yolo_loss, bboxes_iou, non_zero_loss
from config import get_ram_mem_usage, get_gpu_mem_usage
import config

#------------------------------

def save_checkpoint(model, optimizer, checkpoint_path, iou=0):
    folder = Path(checkpoint_path)
    folder.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'num_boxes' : model.max_boxes,
        'n_box_classes' : model.n_box_classes,

        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),        

        'iou': iou,
    }, checkpoint_path)

def train_epoch(model, loader, device, criterion, optimizer, scheduler=None, epoch=0, epochs=0, plot_prediction=False, plot_folder='runs'):
    '''
    runs entire loader via model.train()
    calculates loss and precision metrics
    plots predictions and truth (time consuming)
    '''
    model.train()

    running_loss = 0.0
    counter = 0
    ch_l = 0

    progress_bar = tqdm(enumerate(loader), total=len(loader))
    for batch_i, (imgs, boxes, keypoints) in progress_bar:
        counter += 1

        coord_l = 0
        cls_l = 0

        optimizer.zero_grad()

        imgs = imgs.to(device)
        if isinstance(model, DwgKeyPointsRcnn):
            keypoints = keypoints.to(device)
            out = model(imgs, keypoints)
            loss = out['loss_keypoint']
            counter = 1
        if isinstance(model, DwgKeyPointsYolov4) and isinstance(criterion, Yolo_loss):
            boxes = boxes.to(device)
            out = model(imgs)
            loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(out, boxes)
        else:
            keypoints = keypoints.to(device)
            out = model(imgs)
            loss, coord_l, cls_l = criterion(out, keypoints)
        #ch_l = my_chamfer_distance(out[:, :, :2],targets[:, :, 2:4])

        running_loss += loss.item()
        loss.backward()
            
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        progress_bar.set_description(f'[{epoch} / {epochs}]Train. GPU:{get_gpu_mem_usage():.1f}G RAM:{get_ram_mem_usage():.0f}% Running loss: {running_loss / counter:.4f}')

    return running_loss / counter

def val_epoch(model, loader, device, criterion=None, epoch=0, epochs=0, plot_prediction=False, plot_folder=None):
    '''
    runs entire loader via model.eval()
    calculates loss and precision metrics
    plots predictions and truth (time consuming)
    '''
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(enumerate(loader), total=len(loader))
        mean_ious = []
        for batch_i, (imgs, true_boxes, keypoints) in progress_bar:
            mean_iou = 0
            # keypoints = keypoints.to(config.device)
            imgs = imgs.to(device)
            out = model(imgs)
            predicted_boxes = out[0] # batch * 1008 * max_boxes * 4:[x1 y1 x2 y2]
            confidences = out[1] # batch * 1008 * n_classes

            predictions = nms_conf_suppression(box_array=predicted_boxes, confs=confidences, conf_thresh=0.25, nms_thresh=0.1)

            # TODO: mean iou(boxes, true_boxes) mention device
            pred_counter = 0
            for img_i, (pred, trueb) in enumerate(zip(predictions, true_boxes)):
                for single_true_box in trueb:
                    pred_counter += 1
                    max_iou = 0
                    for single_predicted_box in pred:
                        iou_ = bbox_iou(single_predicted_box, single_true_box)
                        max_iou = max(iou_, max_iou)
                    mean_iou += max_iou
            mean_iou /= pred_counter
            mean_ious.append(mean_iou)
            progress_bar.set_description(f"[{epoch} / {epochs}]Val . Mean IOU:{mean_iou:.4f}.")

            #debug plot
            if plot_prediction and plot_folder is not None:
                plot_batch_grid(
                            input_images=imgs,
                            true_boxes=true_boxes,
                            true_keypoints=keypoints,
                            predictions=predictions,
                            plot_save_file=f'{plot_folder}/val_{epoch}_{batch_i}.png')
                plt.close()
            
    return np.mean(mean_ious)

def run(
        image_folder='data/images',
        data_file_path='data/ids128.json',
        
        batch_size=4,
        epochs=20,
        checkpoint_interval=10,
        lr=0.001,

        validate=True,
        limit_number_of_records=None,

        checkpoint_path=None
    ):

    # create dataset from images or from cache
    # for debug: take only small number of records from dataset
    entities = EntityDataset(limit_number_of_records=limit_number_of_records)
    if data_file_path.endswith('.json'):
        entities.from_json_ids_pickle_labels_img_folder(ids_file=data_file_path, image_folder=image_folder)
    elif data_file_path.endswith('.cache'):
        entities.from_cache(cache_file=data_file_path)

    dwg_dataset = DwgDataset(entities=entities, batch_size=batch_size)
    dwg_dataset.entities.save_cache('data/ids128.cache')

    train_loader = dwg_dataset.train_loader
    val_loader   = dwg_dataset.val_loader

    assert len(train_loader) > 0 and len(val_loader) > 0, "No data"

    num_classes = dwg_dataset.entities.num_classes

    # create model
    #model = DwgKeyPointsModel(max_points=dwg_dataset.entities.max_labels, num_pnt_classes=dwg_dataset.entities.num_pnt_classes, num_coordinates=dwg_dataset.entities.num_coordinates, num_img_channels=dwg_dataset.entities.num_image_channels)
    # model = DwgKeyPointsResNet50(pretrained=True, requires_grad=False, max_points=dwg_dataset.entities.max_labels, num_pnt_classes=dwg_dataset.entities.num_pnt_classes, num_coordinates=dwg_dataset.entities.num_coordinates, num_img_channels=dwg_dataset.entities.num_image_channels)
    model = DwgKeyPointsYolov4(
                                pretrained=False,
                                requires_grad=True,
                                max_boxes=dwg_dataset.entities.max_boxes,
                                num_pnt_classes=dwg_dataset.entities.max_keypoints_per_box,
                                n_box_classes=num_classes + 1,
                                num_coordinates=dwg_dataset.entities.num_coordinates,
                                num_img_channels=dwg_dataset.entities.num_image_channels)

    # n_labels = entities.max_labels // entities.num_pnt_classes
    #model = DwgKeyPointsRcnn(
    #                        requires_grad=False,
    #                        pretrained=True,
    #                        max_labels=n_labels,
    #                        num_pnt_classes=dwg_dataset.entities.num_pnt_classes,
    #                        num_coordinates=dwg_dataset.entities.num_coordinates,
    #                        num_img_channels=dwg_dataset.entities.num_image_channels)
    model.to(config.device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    scheduler = None
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.95)

    #criterion = non_zero_loss(coordinate_loss_name="MSELoss", coordinate_loss_multiplier=1, class_loss_multiplier=1)
    criterion = Yolo_loss(device=config.device, batch=batch_size, n_classes=num_classes + 1, image_size=dwg_dataset.entities.img_size)

    if checkpoint_path:
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path)
            model.max_boxes = checkpoint['max_boxes']

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # logging to TB
    runs_dir = 'runs'

    # autoincrement run (better to do it after loading model and dataset - less folders in debug)
    run_number = 1
    for log_dir in os.walk(runs_dir):
        for dirno in log_dir[1]:
            if dirno.isdigit():
                no = int(dirno)
                if no >= run_number:
                    run_number = no + 1
    tb_log_path = f'{runs_dir}/{run_number}'

    #https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tb = SummaryWriter(tb_log_path)

    best_iou = 0.0

    start = time.time()
    for epoch in range(epochs):

        train_loss = train_epoch(
                                model=model,
                                loader=train_loader,
                                device=config.device,
                                criterion=criterion,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                epochs=epochs,
                                plot_prediction=False,
                                plot_folder=tb_log_path)

        checkpoint_is_here = checkpoint_interval is not None and epoch % checkpoint_interval == 0
        if checkpoint_is_here:
            if validate:
                val_iou = val_epoch(
                                    model=model,
                                    loader=val_loader,
                                    device=config.device,
                                    criterion=criterion,
                                    epoch=epoch,
                                    epochs=epochs,
                                    plot_folder=tb_log_path,
                                    plot_prediction=True)

            last_epoch = (epoch == epochs - 1)
            checkpoint_is_better = val_iou != 0 and (val_iou >= best_iou)
            should_save_checkpoint = checkpoint_is_here or last_epoch or checkpoint_is_better
            if should_save_checkpoint:
                save_checkpoint(model, optimizer, checkpoint_path=f'{tb_log_path}/checkpoint{epoch}.weights', iou=val_iou)
                # Display generated figure in tensorboard
                figs = plot_loader_predictions(loader=val_loader, model=model, epoch=epoch, plot_folder=tb_log_path)
                for i, fig in enumerate(figs):
                    tb.add_figure(tag=f'run_{run_number}', figure=fig, global_step=epoch, close=True)

                plt.close()

            # save checkpoint for best results
            if checkpoint_is_better:
                best_iou = val_iou
                save_checkpoint(model, optimizer, checkpoint_path=f'{tb_log_path}/best.weights', iou=val_iou)
                # print(f"Best recall: {best_recall:.4f} Best precision: {best_precision:.4f}")

            print(f'[{epoch} / {epochs}]@{(time.time() - start):.0f} sec. train loss: {train_loss:.4f} val_iou:{val_iou:.4f} \n')

            tb.add_scalar("accuracy/train_loss", train_loss, epoch)
            tb.add_scalar("accuracy/iou", val_iou, epoch)

    print(f'[DONE] @{time.time() - start:.0f} sec. Training achieved best iou: {best_iou:.4f}. This run data is at "{tb_log_path}"')
    tb.close()

# TODO: augment: rotate, flip, crop

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/ids128.cache', help='Path to ids.json or dataset.cache of dataset')
    parser.add_argument('--image-folder', type=str, default='data/images', help='Path to source images')
    parser.add_argument('--limit-number-of-records', type=int, default=128, help='Take only this maximum records from dataset')

    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=32, help='Size of batch')
    parser.add_argument('--lr', type=float, default=0.008, help='Starting learning rate')

    parser.add_argument('--checkpoint-interval', type=int, default=50, help='Save checkpoint every n epoch')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to starting checkpoint weights')
    opt = parser.parse_args()
    return vars(opt) # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary

if __name__ == "__main__":
    opt = parse_opt()
    run(
        image_folder=opt['image_folder'],
        data_file_path=opt['data'],

        batch_size=opt['batch_size'],

        epochs=opt['epochs'],
        checkpoint_interval=opt['checkpoint_interval'],
        lr=opt['lr'],
        validate=True,
        limit_number_of_records=opt['limit_number_of_records'],
        
        checkpoint_path=opt['checkpoint'])
