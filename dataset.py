import torch
from torch.utils.data import Dataset, SequentialSampler, SubsetRandomSampler

import json
import pandas as pd

import numpy as np

from tqdm import tqdm
from PIL import Image

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

        print(f'Entity dataset. Images: {len(self.data)} Max points:{self.max_labels}. Records limit:{self.limit_number_of_records}')

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

    def __init__(self, limit_number_of_records=None):
        '''

        '''
        
        self.max_labels = 0
        self.img_size = None
        self.num_image_channels = 3

        self.limit_number_of_records = limit_number_of_records

        self.classes = ['AlignedDimension']
        self.pnt_classes = ['XLine1Point', 'XLine2Point', 'DimLinePoint']
        self.coordinates = ['X', 'Y']

        self.num_classes = len(self.classes)
        self.num_pnt_classes = len(self.pnt_classes)
        self.num_coordinates = len(self.coordinates)

        self.ids = []
        self.data = []

    def __len__(self):
        if self.limit_number_of_records is not None:
            return min(len(self.data), self.limit_number_of_records)
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
            # batch * pnt_id * [dim_id, pnt_id, x, y, good, cls_id]
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
