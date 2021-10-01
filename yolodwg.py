import enum
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

from PIL import Image

from processing import build_data

# https://stackoverflow.com/questions/63268967/configure-pycharm-debugger-to-display-array-tensor-shape
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info

#------------------------------
from torch.utils.data import Dataset, SubsetRandomSampler

class EntityDataset(Dataset):
    def __init__(self, img_size=512, limit_records=None):
        df, ids = build_data(rebuild=True, img_size=img_size, limit_records=limit_records)

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

        for group_no, group_id in tqdm(enumerate(ids)):
            source_image_annotated = f'./data/images/annotated_{group_id}.png'
            source_image_stripped = f'./data/images/stripped_{group_id}.png'
            img, max_size = self.open_square(source_image_stripped)
            img = img.resize((img_size,img_size))
            # self.num_image_channels = max(self.num_image_channels, img.getchannel())

            for class_id, class_name in enumerate(self.classes):
                dims = df[(df['GroupId'] == group_id) & (df['ClassName'] == class_name)]
                dim_count = len(dims)

                # [class_id, dim_id, pnt_id, x, y, good?]
                labels = np.zeros([dim_count, 1 + 1 + self.num_coordinates + 1], dtype=np.float)
                # labels = np.zeros([dim_count, self.num_classes, self.num_pnt_classes, self.num_coordinates], dtype=np.float)
                
                for dim_no, dim_row in dims.iterrows():
                    
                    labels[dim_no, 0] = class_id # AlignedDimension
                    for pnt_id, pnt_class in enumerate(self.pnt_classes):
                        
                        labels[dim_no, 1] = pnt_id
                        for coord_id, coord_name in enumerate(self.coordinates):
                            labels[dim_no, 1 + 1 + coord_id] = dim_row[f'{pnt_class}.{coord_name}'] #  ['XLine1Point','XLine2Point','DimLinePoint'].[X,Y]

                    labels[dim_no, 1 + 1 + self.num_coordinates] = 0 # good

            labels /= max_size
            if dim_count > self.max_labels:
                self.max_labels = dim_count
            self.data.append((np.array(img), labels))

    def open_square(self, src_img_path):
        '''
        https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, lbl = self.data[index]
        return img, lbl

class DwgDataset:
    def __init__(self, batch_size=4, img_size=512, limit_records=None):
        self.batch_size = batch_size

        self.entities = EntityDataset(img_size=img_size, limit_records=limit_records)
        self.img_size = img_size

        data_len = len(self.entities)

        validation_fraction = 0.1
        test_fraction       = 0.2
        np.random.seed(42)

        val_split  = int(np.floor(validation_fraction * data_len))
        test_split = int(np.floor(test_fraction * data_len))
        indices = list(range(data_len))
        np.random.shuffle(indices)

        val_indices   = indices[:val_split]
        test_indices  = indices[val_split:test_split]
        train_indices = indices[test_split:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler   = SubsetRandomSampler(val_indices)
        test_sampler  = SubsetRandomSampler(test_indices)

        # https://stackoverflow.com/questions/64586575/adding-class-objects-to-pytorch-dataloader-batch-must-contain-tensors
        def custom_collate(sample):
            imgs = torch.zeros((self.batch_size, self.img_size, self.img_size, self.entities.num_image_channels))
            # batch, label_id, class_id, dim_id, x, y, good
            lbls = torch.zeros((self.batch_size, self.entities.max_labels, 1 + 1 + self.entities.num_coordinates + 1))

            for i in range(len(sample)):
                img, lbl = sample[i]
                imgs[i] = torch.from_numpy(img)
                num_labels = lbl.shape[0]
                lbls[i, :num_labels] = torch.from_numpy(lbl)
            return imgs, lbls

        self.train_loader = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = train_sampler, collate_fn=custom_collate, drop_last=False)
        self.val_loader   = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = val_sampler, collate_fn=custom_collate, drop_last=False)
        self.test_loader  = torch.utils.data.DataLoader(self.entities, batch_size = batch_size, sampler = test_sampler, collate_fn=custom_collate)       
#------------------------------

#------------------------------
class DwgKeyPointsModel(nn.Module):
    def __init__(self, max_coords):
        super(DwgKeyPointsModel, self).__init__()
        self.max_coords = max_coords
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

        return out
#------------------------------

def run(
        batch_size=4,
        epochs=20,
        img_size=512,
        limit_records=None,
        checkpoint_interval=1,
        checkpoint_dir="checkpoints",
        lr=0.001
    ):

    runs_dir = 'runs'

    # autoincrement run
    run_number = 1
    for dir in os.walk(runs_dir):
        for dirno in dir[1]:
            if dirno.isdigit():
                no = int(dirno)
                if no > run_number:
                    run_number = no + 1
    tb_log_path = f'{runs_dir}/{run_number}'

    #https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tb = SummaryWriter(tb_log_path)

    # enable gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataset
    dwg_dataset = DwgDataset(batch_size=batch_size, img_size=img_size, limit_records=limit_records)

    train_loader = dwg_dataset.train_loader
    val_loader   = dwg_dataset.val_loader
    test_loader  = dwg_dataset.test_loader

    # create model
    model = DwgKeyPointsModel(max_coords=dwg_dataset.entities.max_labels * dwg_dataset.entities.num_coordinates)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()

    start = time.time()

    best_recall = 0
    best_precision = 0

    def train_epoch(model, loader):
        model.train()

        running_loss = 0.0
        counter = 0

        for batch_i, (imgs, targets) in tqdm(enumerate(loader)):
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

    def val_epoch(model, loader):
        model.eval()

        running_loss = 0.0
        counter = 0
        with torch.no_grad():
            # TODO: tqdm desc and total
            for batch_i, (imgs, targets) in tqdm(enumerate(loader)):
                counter += 1

                imgs = imgs.to(device)
                targets = targets.to(device)
                targets = targets[:, :, -3:-1]
                targets = targets.reshape(targets.size(0), -1)

                out = model(imgs)
                loss = criterion(out, targets)
                running_loss += loss.item()
        return running_loss / counter
        
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = val_epoch(model, val_loader)

        print(f'[{epoch}/{epochs}]@{time.time()-start:.0f}s train loss: {train_loss:.4f} val_loss:{val_loss:.4f}')

        tb.add_scalar("train loss", train_loss, epoch)
        tb.add_scalar("val loss", val_loss, epoch)
        # TODO: display images in tb

# TODO: save checkpoint
#        if epoch % checkpoint_interval == 0:
#            model.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))
#        if model.losses['recall'] > best_recall and model.losses['precision'] > best_precision:
#            best_precision = model.losses['precision']
#            best_recall = model.losses['recall']
#            model.save_weights(checkpoint_dir + "/best.weights")
#            print("best recall: {0:.4f} best precision: {1:.4f}".format(best_recall, best_precision))


# TODO: inference
# TODO: plots
# TODO: generate points by triades
# TODO: cache images in dataset
# TODO: calculate precision
# TODO: parse arguments
# TODO: play with architecture, fully convolutional resnet50

if __name__ == "__main__":
    run(batch_size=32, img_size=64, limit_records=300, epochs=300)