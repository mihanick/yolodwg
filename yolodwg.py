import os
from tqdm import tqdm
import time
import psutil
import json
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, SequentialSampler, SubsetRandomSampler


from plot import plot_batch_grid, plot_loader_predictions
import config
#------------------------------


def open_square(src_img_path, to_size=512):
    '''
    https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    Opens specified path. Gets image max_size
    Pastes opened at bottom-left of square max_size*max_size
    rescales image to to_size

    returns numpy array with 3 channels
    '''
    src = Image.open(src_img_path)
    max_size = max(src.size)

    # https://stackoverflow.com/questions/50898034/how-replace-transparent-with-a-color-in-pillow
    trg = Image.new(mode="RGB", size=(max_size, max_size), color=(255, 255, 255))

    # http://espressocode.top/python-pil-paste-and-rotate-method/
    # image is pasted using top left coords
    # while drawing coordinate system 0,0 is bottom left
    # so we have to shift by y size difference
    # 
    # remove black background
    # https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    trg.paste(src, box=(0, max_size - src.size[1]), mask=src.split()[3])

    trg = trg.convert('RGB')
    trg = trg.resize((to_size, to_size))

    #nsrc = np.array(src)
    ntrg = np.array(trg)

    #src.save('src.png')
    #trg.save('trg.png')

    return ntrg, max_size

class EntityDataset(Dataset):
    '''
    Dataset of images - labels (points of dimensions)
    '''
    def from_cache(self, cache_file):
        '''
        Reads from saved data using torch
        '''
        
        try: # file could not exist, or torch might not load it
            cached_data = torch.load(cache_file)
            # check cached validity:
            self.img_size = cached_data['img_size']
            self.max_labels = cached_data['max_labels']
            self.data = cached_data['data']

        except Exception as e:
            print(f'Could not load cache: {e}')

        print(f'Entity dataset. Images: {len(self.data)} Max points:{self.max_labels}.')

    def save_cache(self, save_path):
        # save creaed data as cache
        torch.save({
                    'img_size':self.img_size,
                    'max_labels': self.max_labels,
                    'data':self.data
                    }, 
                    save_path)

    def from_json_ids_pickle_labels_img_folder(self, ids_file='ids.json', image_folder='data/images'):
        '''
        Creates from pandas pickle reading img_ids using images in image_folder
        '''

        with open(ids_file, mode='r') as f:
            json_data = json.load(f)

        self.img_size = json_data['img_size']
        self.ids = json_data['ids']
        labels_pandas_file = json_data['labels_pandas_file']
        df = pd.read_pickle(labels_pandas_file)
        
        progress_bar = tqdm(enumerate(self.ids), total=len(self.ids))
        for group_no, group_id in progress_bar:
            source_image_annotated = f'{image_folder}/annotated_{group_id}.png'
            source_image_stripped = f'{image_folder}/stripped_{group_id}.png'
            img, max_size = open_square(source_image_stripped, to_size=self.img_size)
            # self.num_image_channels = max(self.num_image_channels, img.getchannel())

            for class_id, class_name in enumerate(self.classes):
                dims = df[(df['GroupId'] == group_id) & (df['ClassName'] == class_name)]
                dim_count = len(dims)

                # [class_id, dim_id, pnt_id, x, y, good?]
                labels = np.zeros([dim_count * len(self.pnt_classes), 1 + 1 + 1 + self.num_coordinates + 1], dtype=float)
                # labels = np.zeros([dim_count, self.num_classes, self.num_pnt_classes, self.num_coordinates], dtype=float)

                pnt_row_counter = 0
                for dim_no, dim_row in dims.iterrows():
                    for pnt_id, pnt_class in enumerate(self.pnt_classes):
                        labels[pnt_row_counter, 0] = dim_no + 1 # dimension number
                        labels[pnt_row_counter, 1] = pnt_id + 1 # point_id (nonzero)
                        for coord_id, coord_name in enumerate(self.coordinates):
                            coordval = dim_row[f'{pnt_class}.{coord_name}'] #  ['XLine1Point','XLine2Point','DimLinePoint'].[X,Y]
                            labels[pnt_row_counter, 1 + 1 + coord_id] = coordval
                        labels[pnt_row_counter, 1 + 1 + self.num_coordinates] = 0 # good
                        labels[pnt_row_counter, 1 + 1 + 1 + self.num_coordinates] = class_id + 1 # AlignedDimension (nonzero)
                        pnt_row_counter += 1

                # remember maximum number of points
                if pnt_row_counter > self.max_labels:
                    self.max_labels = pnt_row_counter

            labels[:, 2:4] /= self.img_size # scale coordinates to [0..1]
            #labels[:, 3] = labels[:, 3] # invert y coord as on the drawing it will be from top left, not from bottom left
            self.data.append((img, labels))

            progress_bar.set_description(f'Gathering dataset. max labels: {self.max_labels}. {group_id}: labels: {pnt_row_counter}')

        print(f'Entity dataset. Images: {len(self.data)} Max points:{self.max_labels}.')

    def __init__(self):
        '''

        '''
        
        self.max_labels = 0
        self.img_size = None
        self.num_image_channels = 3

        self.limt_number_of_records = None

        self.classes = ['AlignedDimension']
        self.pnt_classes = ['XLine1Point', 'XLine2Point', 'DimLinePoint']
        self.coordinates = ['X', 'Y']

        self.num_classes = len(self.classes)
        self.num_pnt_classes = len(self.pnt_classes)
        self.num_coordinates = len(self.coordinates)

        self.ids = []
        self.data = []

    def __len__(self):
        if self.limt_number_of_records is not None:
            return min(len(self.data), self.limt_number_of_records)
        return len(self.data)

    def __getitem__(self, index):
        img, lbl = self.data[index]
        return img, lbl

class DwgDataset:
    def __init__(self, 
                    entities,
                    batch_size=32):

        self.batch_size = batch_size
        self.entities = entities

        self.img_size = self.entities.img_size

        data_len = len(self.entities)

        validation_fraction = 0.1
        np.random.seed(42)

        val_split = int(np.floor(validation_fraction * data_len))
        indices = list(range(data_len))
        np.random.shuffle(indices)

        val_indices   = indices[:val_split]
        train_indices = indices[val_split:]

        # we need to keep order for debug purposes
        # for release could change to SubsetRandomSampler
        train_sampler = SequentialSampler(train_indices) 
        val_sampler   = SequentialSampler(val_indices)

        # https://stackoverflow.com/questions/64586575/adding-class-objects-to-pytorch-dataloader-batch-must-contain-tensors
        def custom_collate(sample):
            imgs = torch.zeros((self.batch_size, self.entities.num_image_channels, self.img_size, self.img_size), dtype=torch.float32)
            # batch, label_id, class_id, dim_id, x, y, good
            lbls = torch.zeros((self.batch_size, self.entities.max_labels, 1 + 1 + 1 + self.entities.num_coordinates + 1), dtype=torch.float32)

            for i in range(len(sample)):
                img, lbl = sample[i]
                ima = img / 255
                ima = np.transpose(ima, (2, 0, 1)) # x, y, channels -> channels, x, y
                ima = torch.from_numpy(ima)
                #ima = ima.unsqueeze(0)

                imgs[i] = ima
                num_labels = lbl.shape[0]
                lbls[i, :num_labels] = torch.from_numpy(lbl)

            return imgs, lbls

        self.train_loader = torch.utils.data.DataLoader(self.entities, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate, shuffle=False, drop_last=False)
        self.val_loader   = torch.utils.data.DataLoader(self.entities, batch_size=batch_size, sampler=val_sampler, collate_fn=custom_collate, shuffle=False, drop_last=False)

#------------------------------
class DwgKeyPointsResNet50(nn.Module):
    '''
    https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/
    '''
    def __init__(self, requires_grad=True, pretrained=True, max_points=100, num_coordinates=2, num_channels=3):
        super(DwgKeyPointsResNet50, self).__init__()
        self.max_points = max_points
        self.num_coordinates = num_coordinates
        self.max_coords = self.max_points * self.num_coordinates

        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, self.max_coords * 4)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.bn = nn.BatchNorm1d(1000)
        self.do = nn.Dropout()
        # add the final layer
        self.l0 = nn.Linear(100, self.max_coords)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model(x)
        x = x.reshape(batch, self.max_points, -1)
        x = F.adaptive_max_pool2d(x, (self.max_points, 2))
        x = x.reshape(batch, -1)
        
        # x = self.bn(x)
        # x = self.l0(x)
        # x = self.do(x)
        return x

#------------------------------
class DwgKeyPointsModel(nn.Module):
    def __init__(self, max_points=100, num_coordinates=2, num_channels=3):
        '''
        Regresses input images to
        flattened max_points*num_coordinates predictions of keypoints
        '''
        super(DwgKeyPointsModel, self).__init__()
        self.max_points = max_points
        self.num_coordinates = num_coordinates
        self.max_coords = self.max_points * self.num_coordinates
        self.num_channels = num_channels

        s = 16 #vanilla

        self.conv1 = nn.Conv2d(self.num_channels, s*2, kernel_size=5)
        self.conv2 = nn.Conv2d(s*2, s*4, kernel_size=3)
        self.conv3 = nn.Conv2d(s*4, s*8, kernel_size=3)

        #self.batchnorm1 = nn.BatchNorm2d(s*2)
        #self.batchnorm2 = nn.BatchNorm2d(s*4)
        #self.batchnorm3 = nn.BatchNorm2d(s*8)

        self.fc1 = nn.Linear(s*8, self.max_coords) # x, y for each point
        self.pool = nn.MaxPool2d(3, 3)

        self.dropout = nn.Dropout2d(p=0.2)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        #x = self.batchnorm1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)

        #x = self.batchnorm2(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        # x = self.batchnorm3(x)

        bs, _, _, _ = x.shape
        x = F.adaptive_max_pool2d(x, 1)
        x = x.reshape(bs, -1)
        x = self.dropout(x)

        x = self.fc1(x)

        return x
#------------------------------

def save_checkpoint(model, optimizer, loss, checkpoint_path, precision=0, recall=0, f1=0):
    
    folder = Path(checkpoint_path)
    folder.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'max_points':model.max_points,
        'num_coordinates':model.num_coordinates,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':loss,
        'precision': precision,
        'recall': recall,
        'f1':f1
    }, checkpoint_path)

def get_gpu_mem_usage():
    return torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else 0 #gb
def get_ram_mem_usage():
    # https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python
    #print(psutil.cpu_percent())
    #print(psutil.virtual_memory())  # physical memory usage
    return psutil.virtual_memory()[2]

class non_zero_loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mse = nn.SmoothL1Loss()

    def forward(self, outs, ground_truth):
        batch_size = ground_truth.shape[0]
        loss = 0
        counter = 0
        for i in range(batch_size):
            bgt = ground_truth[i]
            non_empty_targets = bgt[bgt[:, 5] > 0]
            n_true_pnts = non_empty_targets.shape[0]
            if n_true_pnts > 0:
                counter += 1
                outs_unf = outs[i].view(-1, 2) # unflatten
                outs_compared = outs_unf[:n_true_pnts] # only compare number of points 
                non_empty_compared = non_empty_targets[:, 2:4].to(self.device) # only compare coordinates
                c_loss = self.mse(outs_compared, non_empty_compared)

                loss += c_loss
        return loss / counter

def train_epoch(model, loader, device, criterion, optimizer, scheduler=None, epoch=0, epochs=0, plot_prediction=False, plot_folder='runs'):
    '''
    runs entire loader via model.train()
    calculates loss and precision metrics
    plots predictions and truth (time consuming)
    '''
    model.train()

    running_loss = 0.0
    counter = 0

    progress_bar = tqdm(enumerate(loader), total=len(loader))
    for batch_i, (imgs, targets) in progress_bar:
        counter += 1

        progress_bar.set_description(f'[{epoch} / {epochs}]Train. GPU:{get_gpu_mem_usage():.1f}G RAM:{get_ram_mem_usage():.0f}% Running loss: {running_loss / counter:.4f}')

        imgs = imgs.to(device)
        targets = targets.to(device)

        batch_size = targets.shape[0]

        optimizer.zero_grad()

        out = model(imgs)

        loss = criterion(out, targets)
        #loss = criterion(out, targets[:, :, 2:4].reshape(batch_size, -1))

        running_loss += loss.item()

        #debug plot
        if plot_prediction and plot_folder is not None:
            plot_batch_grid(
                        input_images=imgs,
                        true_keypoints=targets,
                        predictions=out,
                        plot_save_file=f'{plot_folder}/train_{epoch}_{batch_i}.png')
            plt.close()

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    
    return running_loss / counter

def val_epoch(model, loader, criterion, device, epoch=0, epochs=0, plot_prediction=False, plot_folder=None):
    '''
    runs entire loader via model.eval()
    calculates loss and precision metrics
    plots predictions and truth (time consuming)
    '''
    model.eval()

    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        precision = 0.0
        recall = 0.0
        f1 = 0.0

        progress_bar = tqdm(enumerate(loader), total=len(loader))
        for batch_i, (imgs, ground_truth) in progress_bar:
            counter += 1

            progress_bar.set_description(f"[{epoch} / {epochs}]Val  . Precision:{precision:.4f}. Recall:{recall:.4f}. F1:{f1:4f} Runnning loss: {running_loss / counter:.4f}")

            batch_size = ground_truth.shape[0]

            imgs = imgs.to(device)

            out = model(imgs)

            loss = criterion(out, ground_truth)
            #loss = criterion(out, ground_truth[:, :, 2:4].reshape(batch_size, -1).to(config.device))
            running_loss += loss.item()

            if plot_prediction and plot_folder is not None:
                plot_batch_grid(
                            input_images=imgs,
                            true_keypoints=ground_truth,
                            predictions=out,
                            plot_save_file=f'{plot_folder}/validation_{epoch}_{batch_i}.png')
                plt.close() # reduce memory

            predictions = out.reshape(batch_size, -1, model.num_coordinates)
            
            for i, points in enumerate(ground_truth):
                # https://stasiuk.medium.com/pose-estimation-metrics-844c07ba0a78
                # point is predicted correctly if it is within 0.05 of bound
                prediction = predictions[i]

                empty_points_count = 0
                for j, point in enumerate(points):
                    # number of filled in true points is generally
                    # less than max_poits, so some points are filled
                    # with class_id == 0
                    empty_true_point = point[5] == 0

                    
                    if not empty_true_point:
                        n_dim = point[0] # dimension identifier for this point
                        pnt_x, pnt_y = point[2], point[3]
                        points_of_same_dim = points[points[:, 0]==n_dim]
                        points_of_same_dim = points_of_same_dim[points_of_same_dim[:, 2]!=0]

                        # bound box (min and max coordinates of all points of this dimension):
                        min_x, min_y, max_x, max_y = torch.min(points_of_same_dim[:, 2]), torch.min(points_of_same_dim[:, 3]), torch.max(points_of_same_dim[:, 2]), torch.max(points_of_same_dim[:, 3])
                        boundbox_diagonal = torch.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
                        tolerance = 0.1 * boundbox_diagonal
                        #tolerance = 0.05

                        # find all distances from current point to predictions
                        distances_from_current_point = torch.sqrt((prediction[:, 0] - pnt_x) ** 2 + (prediction[:, 1] - pnt_y) **2)

                        # calculate min distance to compare with tolerance
                        min_distance_from_current_point = torch.min(distances_from_current_point)

                        if min_distance_from_current_point <= tolerance:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        empty_points_count += 1
                # https://stackoverflow.com/questions/60922782/how-can-i-count-the-number-of-1s-and-0s-in-the-second-dimension-of-a-pytorch-t
                predicted_empty_points = (prediction == 0).all(dim=1)
                # https://stackoverflow.com/questions/62150659/how-to-convert-a-tensor-of-booleans-to-ints-in-pytorch
                predicted_empty_points_count = predicted_empty_points.long().sum().item()
                tn += min(empty_points_count, predicted_empty_points_count)
                fn += empty_points_count - tn
        
        # TODO: class metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if tp + fp != 0:
            precision = tp / (tp + fp)
        if tp + fn != 0:
            recall = tp / (tp + fn)
        if precision + recall != 0.0:
            f1 = 2 * precision * recall / (precision + recall)
    return running_loss / counter, precision, recall, f1

def run(
        image_folder='data/images',
        data_file_path='data/ids128.json',
        
        batch_size=4,
        epochs=20,
        checkpoint_interval=10,
        lr=0.001
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
    entities = EntityDataset()
    if data_file_path.endswith('.json'):
        entities.from_json_ids_pickle_labels_img_folder(ids_file=data_file_path, image_folder=image_folder)
    elif data_file_path.endswith('.cache'):
        entities.from_cache(cache_file=data_file_path)
    
    # debug:
    # entities.limt_number_of_records = 50

    dwg_dataset = DwgDataset(entities=entities, batch_size=batch_size)
    #dwg_dataset.entities.save_cache('data/ids128.cache')
    train_loader = dwg_dataset.train_loader
    val_loader   = dwg_dataset.val_loader

    # create model
    #model = DwgKeyPointsModel(max_points=dwg_dataset.entities.max_labels, num_coordinates=dwg_dataset.entities.num_coordinates, num_channels=dwg_dataset.entities.num_image_channels)
    model = DwgKeyPointsResNet50(requires_grad=False, pretrained=True, max_points=dwg_dataset.entities.max_labels, num_coordinates=dwg_dataset.entities.num_coordinates, num_channels=dwg_dataset.entities.num_image_channels)
    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.85)
    scheduler = None 
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.99)

    #criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    criterion = non_zero_loss(config.device)

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
        checkpoint_is_better = recall != 0 and precision !=0 and (recall >= best_recall and precision >= best_precision)
        should_save_checkpoint = (checkpoint_interval is not None and epoch % checkpoint_interval == 0) or last_epoch or checkpoint_is_better
        if should_save_checkpoint:
            save_checkpoint(model, optimizer, criterion, checkpoint_path=f'{tb_log_path}/checkpoint{epoch}.weights', precision=precision, recall=recall, f1=f1)
            # Display generated figure in tensorboard
            figs = plot_loader_predictions(loader=val_loader, model=model, epoch=epoch, plot_folder=tb_log_path)
            for i, fig in enumerate(figs):
                tb.add_figure(tag=f'checkpoint_{epoch}', figure=fig, global_step=epoch, close=True)

            plt.close()

        # save checkpoint for best results
        if checkpoint_is_better:
            best_precision = precision
            best_recall = recall
            save_checkpoint(model, optimizer, criterion, checkpoint_path=f'{tb_log_path}/best.weights', precision=precision, recall=recall, f1=f1)
            # print(f"Best recall: {best_recall:.4f} Best precision: {best_precision:.4f}")

        print(f'[{epoch} / {epochs}]@{(time.time() - start):.0f} sec. train loss: {train_loss:.4f} val_loss:{val_loss:.4f} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} \n')

        tb.add_scalar("loss/train", train_loss, epoch)
        tb.add_scalar("loss/val", val_loss, epoch)
        tb.add_scalar("accuracy/precision", precision, epoch)
        tb.add_scalar("accuracy/recall", recall, epoch)
        tb.add_scalar("accuracy/f1", f1, epoch)
    print(f'[DONE] @{time.time() - start:.0f} sec. Training achieved best recall: {best_recall:.4f} best precision: {best_precision:.4f}. This run data is at "{tb_log_path}"')
    tb.close()

# TODO: generate points by triades
# TODO: calculate precision

# TODO: augment: rotate, flip, crop

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/ids128.cache', help='Path to ids.json or dataset.cache of dataset')
    parser.add_argument('--image-folder', type=str, default='data/images', help='Path to source images')

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=400, help='Size of batch. Keep as max as GPU mem allows')
    parser.add_argument('--lr', type=float, default=0.0025, help='Starting learning rate')

    parser.add_argument('--checkpoint-interval', type=int, default=50, help='Save checkpoint every n epoch')

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
        lr=opt['lr'])