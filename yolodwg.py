import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

from PIL import Image

from processing import build_data
from plot import plot_batch_grid, plot_loader_predictions
import config

#------------------------------
from torch.utils.data import Dataset, SubsetRandomSampler

def open_square(src_img_path):
    '''
    https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    Opens specified path. Gets image max_size
    Pastes opened at bottom-left of square max_size*max_size
    
    returns Image and its max_size

    '''
    src = Image.open(src_img_path)
    max_size = max(src.size)
    trg = Image.new("RGBA", (max_size, max_size))

    # http://espressocode.top/python-pil-paste-and-rotate-method/
    # image is pasted using top left coords
    # while drawing coordinate system 0,0 is bottom left
    # so we have to shift by y size difference
    trg.paste(src, (0, max_size - src.size[1]))
    return trg, max_size

# TODO: non-zero classes and dimnumbers
class EntityDataset(Dataset):
    '''
    Dataset of images - labels (points of dimensions) 
    '''

    def __init__(self, img_size=512, rebuild=False, limit_records=None, use_cache=False):
        '''
        Dataset of images and
        img_size - return images in specified square size
        rebuild - queries entities from mongodb and stores to pandas dataframe dataset{img_size}.pickle
        limit_records - limit max number of records queried from database
        use_cache - torch.save data to speed up loading small datasets
        '''

        self.max_labels = 0
        self.img_size = img_size
        self.num_image_channels = 4

        self.classes = ['AlignedDimension']
        self.pnt_classes = ['XLine1Point', 'XLine2Point', 'DimLinePoint']
        self.coordinates = ['X', 'Y']

        self.num_classes = len(self.classes)
        self.num_pnt_classes = len(self.pnt_classes)
        self.num_coordinates = len(self.coordinates)

        self.data = []

        recreate_cache = True
        self.cached_data_file = f'dataset{img_size}.cache'
        if use_cache:
            try: # file could not exist, or torch might not load it
                cached_data = torch.load(self.cached_data_file)
                # check cached validity:
                if cached_data['img_size'] == img_size:
                    self.max_labels = cached_data['max_labels']
                    self.data = cached_data['data']
                    recreate_cache = False
            except:
                recreate_cache = True

        if recreate_cache:
            df, ids = build_data(rebuild=rebuild, img_size=img_size, limit_records=limit_records)
            
            for group_no, group_id in tqdm(enumerate(ids)):
                source_image_annotated = f'./data/images/annotated_{group_id}.png'
                source_image_stripped = f'./data/images/stripped_{group_id}.png'
                img, max_size = open_square(source_image_stripped)
                img = img.resize((img_size, img_size))
                # self.num_image_channels = max(self.num_image_channels, img.getchannel())

                for class_id, class_name in enumerate(self.classes):
                    dims = df[(df['GroupId'] == group_id) & (df['ClassName'] == class_name)]
                    dim_count = len(dims)

                    # [class_id, dim_id, pnt_id, x, y, good?]
                    labels = np.zeros([dim_count * len(self.pnt_classes), 1 + 1 + self.num_coordinates + 1], dtype=float)
                    # labels = np.zeros([dim_count, self.num_classes, self.num_pnt_classes, self.num_coordinates], dtype=float)

                    label_row_counter = 0
                    for dim_no, dim_row in dims.iterrows():
                        for pnt_id, pnt_class in enumerate(self.pnt_classes):
                            labels[label_row_counter, 0] = class_id # AlignedDimension
                            labels[label_row_counter, 1] = pnt_id
                            for coord_id, coord_name in enumerate(self.coordinates):
                                coordval = dim_row[f'{pnt_class}.{coord_name}'] #  ['XLine1Point','XLine2Point','DimLinePoint'].[X,Y]
                                labels[label_row_counter, 1 + 1 + coord_id] = coordval
                            labels[label_row_counter, 1 + 1 + self.num_coordinates] = 0 # good
                            label_row_counter += 1

                    # remember maximum number of points
                    if label_row_counter > self.max_labels:
                        self.max_labels = label_row_counter

                labels[:, 2:4] /= img_size # scale coordinates to [0..1]
                labels[:, 3] = 1 - labels[:, 3] # invert y coord as on the drawing it will be from top left, not from bottom left
                self.data.append((np.array(img), labels))

            # save creaed data as cache
            if use_cache:
                torch.save({'img_size':self.img_size, 'max_labels': self.max_labels, 'data':self.data}, self.cached_data_file)

        print(f'Entity dataset. Images: {len(self.data)} Max points:{self.max_labels}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, lbl = self.data[index]
        return img, lbl

class DwgDataset:
    def __init__(self, batch_size=4, img_size=512, limit_records=None, rebuild=False, use_cache=False):
        self.batch_size = batch_size

        self.entities = EntityDataset(img_size=img_size, rebuild=rebuild, limit_records=limit_records, use_cache=use_cache)
        self.img_size = img_size

        data_len = len(self.entities)

        validation_fraction = 0.1
        np.random.seed(42)

        val_split = int(np.floor(validation_fraction * data_len))
        indices = list(range(data_len))
        np.random.shuffle(indices)

        val_indices   = indices[:val_split]
        train_indices = indices[val_split:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler   = SubsetRandomSampler(val_indices)

        # https://stackoverflow.com/questions/64586575/adding-class-objects-to-pytorch-dataloader-batch-must-contain-tensors
        def custom_collate(sample):
            imgs = torch.zeros((self.batch_size, self.img_size, self.img_size, self.entities.num_image_channels), dtype=torch.float32)
            # batch, label_id, class_id, dim_id, x, y, good
            lbls = torch.zeros((self.batch_size, self.entities.max_labels, 1 + 1 + self.entities.num_coordinates + 1), dtype=torch.float32)

            for i in range(len(sample)):
                img, lbl = sample[i]
                imgs[i] = torch.from_numpy(img)
                imgs /= 255
                num_labels = lbl.shape[0]
                lbls[i, :num_labels] = torch.from_numpy(lbl)
            return imgs, lbls

        self.train_loader = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = train_sampler, collate_fn=custom_collate, drop_last=False)
        self.val_loader   = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = val_sampler, collate_fn=custom_collate, drop_last=False)
        
#------------------------------

#------------------------------
class DwgKeyPointsModel(nn.Module):
    def __init__(self, max_points=100, num_coordinates=2):
        '''
        Regresses input images to
        flattened max_points*num_coordinates predictions of keypoints
        '''
        super(DwgKeyPointsModel, self).__init__()
        self.max_points = max_points
        self.num_coordinates = num_coordinates
        self.max_coords = self.max_points * self.num_coordinates

        self.conv1 = nn.Conv2d(4, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        self.fc1 = nn.Linear(128, self.max_coords) # x, y for each point
        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout2d(p=0.2)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 3) # batch, x, y, channels -> batch, channels, y, x
        x = torch.transpose(x, 2, 3) # batch, channels, y, x -> batch, channels, x, y

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        out = self.fc1(x)

        # TODO: force output to be in range [0..1]
        return out
#------------------------------

def save_checkpoint(model, optimizer, loss, checkpoint_path, precision=0, recall=0, f1=0):
    
    dir = Path(checkpoint_path)
    dir.parent.mkdir(parents=True, exist_ok=True)

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

def train_epoch(model, loader, device, criterion, optimizer, scheduler):
    model.train()

    running_loss = 0.0
    counter = 0

    # TODO: tqdm desc and total
    for _, (imgs, targets) in tqdm(enumerate(loader)):
        counter += 1

        imgs = imgs.to(device)
        targets = targets.to(device)

        # only coordinates, plus flatten
        targets = targets[:, :, -3:-1]
        targets = targets.reshape(targets.size(0), -1)

        optimizer.zero_grad()

        out = model(imgs)
        loss = criterion(out, targets)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return running_loss / counter

def val_epoch(model, loader, device, criterion, epoch=0, plot_prediction=False, plot_folder=None):
    model.eval()

    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        valid = []

        for batch_i, (imgs, ground_truth) in tqdm(enumerate(loader)):
            batch_size = ground_truth.shape[0]
            counter += 1

            imgs = imgs.to(device)
            targets = ground_truth[:, :, -3:-1]
            targets = targets.to(device)
            targets = targets.reshape(targets.size(0), -1)

            out = model(imgs)
            loss = criterion(out, targets)
            running_loss += loss.item()

            if plot_prediction and plot_folder is not None:
                plot_batch_grid(
                            input_images=imgs,
                            true_keypoints=targets,
                            predictions=out,
                            plot_save_file=f'{plot_folder}/prediction_{epoch}.png')

            predictions = out.reshape(batch_size, -1, model.num_coordinates)
            
            for i, points in enumerate(ground_truth):
                # https://stasiuk.medium.com/pose-estimation-metrics-844c07ba0a78
                # point is predicted correctly if it is within 0.05 of bound

                prediction = predictions[i]
                for j, point in enumerate(points):
                    # number of filled in true points is generally 
                    # less than max_poits, so some points are filled
                    # with zeroes
                    empty_true_point = torch.sum(point) == 0
                    prediction_correct = 0
                    
                    if not empty_true_point:
                        n_dim = point[1] # dimension identifier for this point
                        pnt_x, pnt_y = point[2], point[3]
                        points_of_same_dim = points[points[:, 1]==n_dim]
                        points_of_same_dim = points_of_same_dim[points_of_same_dim[:, 2]!=0]

                        # bound box (min and max coordinates of all points of this dimension):
                        min_x, min_y, max_x, max_y = torch.min(points_of_same_dim[:, 2]), torch.min(points_of_same_dim[:, 3]), torch.max(points_of_same_dim[:, 2]), torch.max(points_of_same_dim[:, 3])
                        boundbox_diagonal = torch.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
                        tolerance = 0.1 * boundbox_diagonal

                        # find all distances from current point to predictions
                        distances_from_current_point = torch.sqrt((prediction[:, 0] - pnt_x) ** 2 + (prediction[:, 1] - pnt_y) **2)

                        # calculate min distance to compare with tolerance
                        min_distance_from_current_point = torch.min(distances_from_current_point)

                        if min_distance_from_current_point <= tolerance:
                            prediction_correct = 1
                        valid.append(prediction_correct)

        # tp, fp, fn = 1, 1, 1
        # TODO: class metrics
        precision = np.mean(valid) # tp / (tp + fp)
        recall = 0 # tp / (tp + fn)
        f1 = 0 #2 * precision * recall /(precision + recall)
    return running_loss / counter, precision, recall, f1

def run(
        batch_size=4,
        epochs=20,
        img_size=512,
        rebuild=False,
        limit_records=None,
        use_cache=True,
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

    # enable gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataset
    dwg_dataset = DwgDataset(
                    batch_size=batch_size,
                    img_size=img_size, 
                    limit_records=limit_records,
                    rebuild=rebuild,
                    use_cache=use_cache,
                    )

    train_loader = dwg_dataset.train_loader
    val_loader   = dwg_dataset.val_loader

    # create model
    model = DwgKeyPointsModel(max_points=dwg_dataset.entities.max_labels, num_coordinates=dwg_dataset.entities.num_coordinates)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()

    start = time.time()

    best_recall = 0.0
    best_precision = 0.0

    for epoch in range(epochs):
        train_loss = train_epoch(
                                model=model,
                                loader=train_loader,
                                device=device,
                                criterion=criterion,
                                optimizer=optimizer,
                                scheduler=scheduler)

        val_loss, precision, recall, f1 = val_epoch(
                                model=model,
                                loader=val_loader,
                                device=device,
                                criterion=criterion)

        last_epoch = (epoch == epochs - 1)
        should_save_checkpoint = (checkpoint_interval is not None and epoch % checkpoint_interval == 0) or last_epoch
        if should_save_checkpoint:
            save_checkpoint(model, optimizer, criterion, checkpoint_path=f'{tb_log_path}/checkpoint{epoch}.weights', precision=precision, recall=recall, f1=f1)
            # Display generated figure in tensorboard
            fig = plot_loader_predictions(loader=val_loader, model=model, epoch=epoch, plot_folder=tb_log_path)
            tb.add_figure('predicted', fig, epoch)

        if recall >= best_recall and precision >= best_precision:
            best_precision = precision
            best_recall = recall
            save_checkpoint(model, optimizer, criterion, checkpoint_path=f'{tb_log_path}/best.weights', precision=precision, recall=recall, f1=f1)
            # print(f"Best recall: {best_recall:.4f} Best precision: {best_precision:.4f}")

        print(f'[{epoch}/{epochs}]@{(time.time() - start):.0f}s train loss: {train_loss:.4f} val_loss:{val_loss:.4f} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f}')

        tb.add_scalar("loss/train", train_loss, epoch)
        tb.add_scalar("loss/val", val_loss, epoch)
        tb.add_scalar("accuracy/precision", precision, epoch)
        #tb.add_scalar("accuracy/recall", recall, epoch)
        #tb.add_scalar("accuracy/f1", f1, epoch)
        
    tb.close()

def plot_val_dataset():
    dwg_dataset = DwgDataset(batch_size=4, img_size=128, limit_records=50, rebuild=False)

    train_loader = dwg_dataset.train_loader
    val_loader   = dwg_dataset.val_loader

    chp_path = 'runs/1/best.weights'
    checkpoint = torch.load(chp_path)
    max_points = checkpoint['max_points']
    num_coordinates = checkpoint['num_coordinates']
    model = DwgKeyPointsModel(max_points=max_points, num_coordinates=num_coordinates).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return plot_loader_predictions(val_loader, model)
# TODO: generate points by triades
# TODO: calculate precision
# TODO: parse arguments
# TODO: play with architecture, fully convolutional resnet50
# TODO: augment: rotate, flip, crop

if __name__ == "__main__":
    run(
        batch_size=32,
        img_size=64,
        limit_records=300,
        rebuild=False,
        use_cache=True,
        epochs=30,
        checkpoint_interval=None)