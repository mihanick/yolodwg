import os
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

from plot import plot_batch_grid, plot_loader_predictions

from config import get_ram_mem_usage, get_gpu_mem_usage
import config

#------------------------------

def save_checkpoint(model, optimizer, checkpoint_path, precision=0, recall=0, f1=0):
    
    folder = Path(checkpoint_path)
    folder.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'max_points':model.max_points,
        'num_coordinates':model.num_coordinates,
        'num_pnt_classes':model.num_pnt_classes,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # TODO: Resume training?
        'precision': precision,
        'recall': recall,
        'f1':f1
    }, checkpoint_path)

from chamfer_loss import my_chamfer_distance

def predicted_classes_from_outputs(outs):
    '''
    returns predicted torch.long(batch_size, max_points) classes from nn output
    outs - nn output torch.tensor(batch_size, max_points, features)
        features[:2] - x,y coordinates
        features[2:] - nn outputs for each pnt class
    '''
    # We softmax features[2:] to summ up to 1, than this will be our predictions of classes with 
    # probabilities

    # than we take max probability and count it as a predicted class
    # .long to use it as index
    return torch.argmax(F.softmax(outs[:, :, 2:], dim=2), dim=2).long()

class non_zero_loss(nn.Module):
    def __init__(self, coordinate_loss_name='chamfer_distance', coordinate_loss_multiplier=1, class_loss_multiplier=1):
        '''
        Batch loss calculation only for non-zero predictions.
        coordinate_loss_name could be ordinary pytorch "MSELoss" "L1Loss", "SmoothL1Loss"
        or "chamfer_distance" from pytorch3d.

        MSE and others compares tensors with fixed size of max_points
        chamfer_distance compares variable sets of 2d points
        (only non-zero-class predicted and true points are taken)

        class loss is CrossEntropyLoss
        multipliers are used to equalize values of coordinate and class losses, as
        their value might be too different for optimizer to take into account
        one or another.
        '''
        super().__init__()

        self.coordinate_loss_name = coordinate_loss_name

        if coordinate_loss_name == "chamfer_distance":
            try:
                # This won't work on cpu
                from pytorch3d.loss import chamfer_distance 
                self.coordinate_loss_name = "chamfer_distance"
                self.coordinate_loss_f = chamfer_distance
            except:
                # Fallback chamfer distance to MSE on cpu
                print('[WARNING] Pytorch3d is for chamfer_distance is not available. Fallback ot MSELoss')
                self.coordinate_loss_name = "MSELoss"
                self.coordinate_loss_f = nn.MSELoss()
        elif coordinate_loss_name == "MSELoss":
            self.coordinate_loss_f = nn.MSELoss()
        elif coordinate_loss_name == "L1Loss":
            self.coordinate_loss_f = nn.L1Loss()
        elif coordinate_loss_name == "SmoothL1Loss":
            self.coordinate_loss_f = nn.SmoothL1Loss()
            

        self.coordinate_loss_multiplier = coordinate_loss_multiplier
        self.class_loss_multiplier = class_loss_multiplier
        self.cls_loss_f = nn.CrossEntropyLoss()

    def forward(self, outs, ground_truth):

        # Coordinate loss calculation:
        if self.coordinate_loss_name == "chamfer_distance":
            predicted_classes = predicted_classes_from_outputs(outs)

            # This magic here basically says: get only prediction with non-zero class.
            # than we take only coordinates of it
            # strangely outs[predicted_classes] returns squished tensor
            # so we'll just unsqueeze it here, as for this purposes
            # batch dimension doesn't matter
            predicted_non_zeros = outs[predicted_classes[:, :] > 0,:][:, :2]

            # This magic says only take non-zero points [5th coord is label_class]
            # and only take their coordinates
            true_non_zeros = ground_truth[ground_truth[:, :, 5] > 0][:, 2:4]

            coordinate_loss, _ = self.coordinate_loss_f(predicted_non_zeros.unsqueeze(0), true_non_zeros.unsqueeze(0))
        else:
            coordinate_loss = self.coordinate_loss_f(outs[:, :, :2], ground_truth[:, :, 2:4])

        # Classification loss calculation:
        classification_loss = 0
        for b, gtb in enumerate(ground_truth):
            
            # Take only outs[:, :, 2:] - these are per-class nn outputs
            predicted_classes_map = outs[b, :, 2:]

            # https://stackoverflow.com/questions/57065070/pytorch-dimension-out-of-range-expected-to-be-in-range-of-1-0-but-got-1
            # Importante here that we take pnt class number as input for CrossEntropyLoss
            gt_pnt_class = gtb[:, 1].long()

            classification_loss += self.cls_loss_f(predicted_classes_map, gt_pnt_class)
        
        # Total loss value:
        total_loss = self.coordinate_loss_multiplier * coordinate_loss + self.class_loss_multiplier * classification_loss

        return total_loss, coordinate_loss, classification_loss

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
    for batch_i, (imgs, targets) in progress_bar:
        counter += 1

        imgs = imgs.to(device)
        targets = targets.to(device)

        batch_size = targets.shape[0]

        optimizer.zero_grad()

        out = model(imgs)

        loss, coord_l, cls_l = criterion(out, targets)
        ch_l = my_chamfer_distance(out[:, :, :2],targets[:, :, 2:4])

        running_loss += loss.item()
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        progress_bar.set_description(f'[{epoch} / {epochs}]Train. GPU:{get_gpu_mem_usage():.1f}G RAM:{get_ram_mem_usage():.0f}% Running loss: {running_loss / counter:.4f} coord:{coord_l:.4f} cls:{cls_l:.4f} chd:{ch_l:.4f}')

        #debug plot
        if plot_prediction and plot_folder is not None:
            plot_batch_grid(
                        input_images=imgs,
                        true_keypoints=targets,
                        predictions=out,
                        plot_save_file=f'{plot_folder}/train_{epoch}_{batch_i}.png')
            plt.close()

    return running_loss / counter

def calculate_metrics_per_batch(out, ground_truth, criterion=None):

    if criterion is not None:
        loss, _, _ = criterion(out, ground_truth)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for img_no, points in enumerate(ground_truth):
        tp_this_image = 0
        fp_this_image = 0
        tn_this_image = 0
        fn_this_image = 0

        # https://stasiuk.medium.com/pose-estimation-metrics-844c07ba0a78
        # point is predicted correctly if it is within 0.05 of bound
        prediction = out[img_no]
        predicted_classes = torch.argmax(F.softmax(prediction[:, 2:], dim=1), dim=1).long()

        # https://stackoverflow.com/questions/60922782/how-can-i-count-the-number-of-1s-and-0s-in-the-second-dimension-of-a-pytorch-t
        # https://stackoverflow.com/questions/62150659/how-to-convert-a-tensor-of-booleans-to-ints-in-pytorch
        predicted_empty_points_count = (predicted_classes == 0).long().sum().item()

        for j, point in enumerate(points):
            target_cls = point[5]
            if target_cls != 0: # non-empty point in ground_truth
                #n_dim = point[0] # dimension identifier for this point
                pnt_x, pnt_y = point[2], point[3]
                #points_of_same_dim = points[points[:, 0] == n_dim]
                #points_of_same_dim = points_of_same_dim[points_of_same_dim[:, 2] != 0]

                ## bound box (min and max coordinates of all points of this dimension):
                #min_x, min_y, max_x, max_y = torch.min(points_of_same_dim[:, 2]), torch.min(points_of_same_dim[:, 3]), torch.max(points_of_same_dim[:, 2]), torch.max(points_of_same_dim[:, 3])
                #boundbox_diagonal = torch.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
                #tolerance = 0.1 * boundbox_diagonal
                tolerance = 0.05 #in debug we can ease tolerance

                # find all distances from current point to predictions
                distances_from_current_point = torch.sqrt((prediction[:, 0] - pnt_x) ** 2 + (prediction[:, 1] - pnt_y) **2)
                closest_prediction_idx = torch.argmin(distances_from_current_point)
                closest_pnt = prediction[closest_prediction_idx]
                closest_pnt_cls = predicted_classes[closest_prediction_idx]
                # calculate min distance to compare with tolerance
                if closest_pnt_cls == target_cls and distances_from_current_point[closest_prediction_idx] <= tolerance:
                    tp_this_image += 1
                else:
                    fp_this_image += 1
            else:
                if predicted_empty_points_count > 0:
                    # subtract one zero from predictions
                    predicted_empty_points_count -= 1
                    # and add itt to true negative prediction
                    tn_this_image += 1
                else:
                    # otherwise count prediction as false negative
                    fn_this_image += 1

        tp += tp_this_image
        fp += fp_this_image
        fn += fn_this_image
        tn += tn_this_image
    accuracy, precision, recall, f1 = 0, 0, 0, 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if tp + fp != 0:
        precision = tp / (tp + fp)
    if tp + fn != 0:
        recall = tp / (tp + fn)
    if precision + recall != 0.0:
        f1 = 2 * precision * recall / (precision + recall)

    return loss.item(), accuracy, precision, recall, f1

def val_epoch(model, loader, criterion, device, epoch=0, epochs=0, plot_prediction=False, plot_folder=None):
    '''
    runs entire loader via model.eval()
    calculates loss and precision metrics
    plots predictions and truth (time consuming)
    '''
    model.eval()
    losses = []
    precisions = []
    recalls = []
    f1s = []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(loader), total=len(loader))
        for batch_i, (imgs, ground_truth) in progress_bar:
            ground_truth = ground_truth.to(config.device)
            imgs = imgs.to(device)
            out = model(imgs)

            loss, accuracy, precision, recall, f1 = calculate_metrics_per_batch(out, ground_truth, criterion)

            if plot_prediction and plot_folder is not None:
                plot_batch_grid(
                            input_images=imgs,
                            true_keypoints=ground_truth,
                            predictions=out,
                            plot_save_file=f'{plot_folder}/validation_{epoch}_{batch_i}.png')
                plt.close() # TODO: reduce memory for plotting val!

            progress_bar.set_description(f"[{epoch} / {epochs}]Val  . Precision:{precision:.4f}. Recall:{recall:.4f}. F1:{f1:4f} Runnning loss: {loss:.4f}")

            losses.append(loss)
            precisions.append((precision))
            recalls.append(recall)
            f1s.append(f1)
            
    return np.mean(losses), np.mean(precisions), np.mean(recalls), np.mean(f1s)

def run(
        image_folder='data/images',
        data_file_path='data/ids128.json',
        
        batch_size=4,
        epochs=20,
        checkpoint_interval=10,
        lr=0.001,

        validate=True,
        limit_number_of_records=None
    ):

    runs_dir = 'runs'

    # autoincrement run
    run_number = 1
    for dir in os.walk(runs_dir):
        for dirno in dir[1]:
            if dirno.isdigit():
                no = int(dirno)
                if no >= run_number:
                    run_number = no + 1
    tb_log_path = f'{runs_dir}/{run_number}'

    #https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tb = SummaryWriter(tb_log_path)

    # create dataset from images or from cache
    # debug: take only small number of records from dataset
    entities = EntityDataset(limit_number_of_records=limit_number_of_records)
    if data_file_path.endswith('.json'):
        entities.from_json_ids_pickle_labels_img_folder(ids_file=data_file_path, image_folder=image_folder)
    elif data_file_path.endswith('.cache'):
        entities.from_cache(cache_file=data_file_path)

    dwg_dataset = DwgDataset(entities=entities, batch_size=batch_size)
    #dwg_dataset.entities.save_cache('data/ids128.cache')
    train_loader = dwg_dataset.train_loader
    val_loader   = dwg_dataset.val_loader

    # create model
    model = DwgKeyPointsModel(max_points=dwg_dataset.entities.max_labels, num_pnt_classes=dwg_dataset.entities.num_pnt_classes, num_coordinates=dwg_dataset.entities.num_coordinates, num_img_channels=dwg_dataset.entities.num_image_channels)
    #model = DwgKeyPointsResNet50(pretrained=True, requires_grad=True, max_points=dwg_dataset.entities.max_labels, num_pnt_classes=dwg_dataset.entities.num_pnt_classes, num_coordinates=dwg_dataset.entities.num_coordinates, num_img_channels=dwg_dataset.entities.num_image_channels)
    #model = DwgKeyPointsYolov4(requires_grad=True, pretrained=True, max_points=dwg_dataset.entities.max_labels, num_coordinates=dwg_dataset.entities.num_coordinates, num_img_channels=dwg_dataset.entities.num_image_channels)
    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    scheduler = None 
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    #criterion = non_zero_loss(coordinate_loss_name="MSELoss", coordinate_loss_multiplier=100, class_loss_multiplier=0.001)
    criterion = non_zero_loss(coordinate_loss_name="chamfer_distance", coordinate_loss_multiplier=1, class_loss_multiplier=0.001)

    best_recall = 0.0
    best_precision = 0.0

    for epoch in range(epochs):
        start = time.time()

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

        val_loss, precision, recall, f1 = 0, 0, 0, 0
        if validate:
            val_loss, precision, recall, f1 = val_epoch(
                                model=model,
                                loader=val_loader,
                                device=config.device,
                                criterion=criterion,
                                epoch=epoch,
                                epochs=epochs,
                                plot_folder=tb_log_path,
                                plot_prediction=False)

        last_epoch = (epoch == epochs - 1)
        checkpoint_is_better = recall != 0 and precision != 0 and (recall >= best_recall and precision >= best_precision)
        should_save_checkpoint = (checkpoint_interval is not None and epoch % checkpoint_interval == 0) or last_epoch or checkpoint_is_better
        if should_save_checkpoint:
            save_checkpoint(model, optimizer, checkpoint_path=f'{tb_log_path}/checkpoint{epoch}.weights', precision=precision, recall=recall, f1=f1)
            # Display generated figure in tensorboard
            figs = plot_loader_predictions(loader=val_loader, model=model, epoch=epoch, plot_folder=tb_log_path)
            for i, fig in enumerate(figs):
                tb.add_figure(tag=f'run_{run_number}', figure=fig, global_step=epoch, close=True)

            plt.close()

        # save checkpoint for best results
        if checkpoint_is_better:
            best_precision = precision
            best_recall = recall
            save_checkpoint(model, optimizer, checkpoint_path=f'{tb_log_path}/best.weights', precision=precision, recall=recall, f1=f1)
            # print(f"Best recall: {best_recall:.4f} Best precision: {best_precision:.4f}")

        print(f'[{epoch} / {epochs}]@{(time.time() - start):.0f} sec. train loss: {train_loss:.4f} val_loss:{val_loss:.4f} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} \n')

        tb.add_scalar("loss/train", train_loss, epoch)
        tb.add_scalar("loss/val", val_loss, epoch)
        tb.add_scalar("accuracy/precision", precision, epoch)
        tb.add_scalar("accuracy/recall", recall, epoch)
        tb.add_scalar("accuracy/f1", f1, epoch)
    print(f'[DONE] @{time.time() - start:.0f} sec. Training achieved best precision: {best_precision:.4f} best recall: {best_recall:.4f}. This run data is at "{tb_log_path}"')
    tb.close()

# TODO: generate points by triades
# TODO: augment: rotate, flip, crop

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/ids128.cache', help='Path to ids.json or dataset.cache of dataset')
    parser.add_argument('--image-folder', type=str, default='data/images', help='Path to source images')
    parser.add_argument('--limit-number-of-records', type=int, default=None, help='Take only this maximum records from dataset')

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64, help='Size of batch')
    parser.add_argument('--lr', type=float, default=0.0002, help='Starting learning rate')

    parser.add_argument('--checkpoint-interval', type=int, default=40, help='Save checkpoint every n epoch')

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
        limit_number_of_records=opt['limit_number_of_records'])
