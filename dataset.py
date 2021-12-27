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

    
    def __init__(self, limit_number_of_records=None):
        '''

        '''
        
        self.max_boxes = 1
        self.max_keypoints_per_box = 3
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

    def from_cache(self, cache_file):
        '''
        Reads from saved data using torch
        '''
        
        try: # file could not exist, or torch might not load it
            cached_data = torch.load(cache_file)
            # check cached validity:
            self.img_size = cached_data['img_size']
            self.max_boxes = cached_data['max_boxes']
            self.max_keypoints_per_box = cached_data['max_keypoints_per_box']
            self.data = cached_data['data']

        except Exception as e:
            print(f'Could not load cache: {e}')

        print(f'Entity dataset. Images: {len(self.data)} Max boxes:{self.max_boxes}. Max keypoints per box:{self.max_keypoints_per_box} Records limit:{self.limit_number_of_records}')

    def save_cache(self, save_path):
        # save creaed data as cache
        torch.save({
                    'img_size':self.img_size,
                    'max_boxes': self.max_boxes,
                    'max_keypoints_per_box':self.max_keypoints_per_box,
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

                # remember maximum number of boxes
                if dim_count > self.max_boxes:
                    self.max_boxes = dim_count

                # [class_id, dim_id, pnt_id, x, y, good?]
                keypoints = np.zeros([dim_count * len(self.pnt_classes), 1 + 1 + 1 + self.num_coordinates + 1], dtype=float)
                kp_counter = 0

                # boundboxes are calculated from keypoints
                boxes = np.zeros((dim_count, 5))
                
                for dim_no, dim_row in dims.iterrows():
                    for pnt_id, pnt_class in enumerate(self.pnt_classes):
                        keypoints[kp_counter, 0] = dim_no + 1 # dimension number
                        keypoints[kp_counter, 1] = pnt_id + 1 # point_id (nonzero)
                        for coord_id, coord_name in enumerate(self.coordinates):
                            coordval = dim_row[f'{pnt_class}.{coord_name}'] #  ['XLine1Point','XLine2Point','DimLinePoint'].[X,Y]

                            keypoints[kp_counter, 1 + 1 + coord_id] = coordval
                        keypoints[kp_counter, 1 + 1 + self.num_coordinates] = 0 # good
                        keypoints[kp_counter, 1 + 1 + 1 + self.num_coordinates] = class_id + 1 # AlignedDimension (nonzero)
                        kp_counter += 1

                    # bound box coordinates from min and max coords of keypoints
                    boxes[dim_no, :2] = np.min(keypoints[keypoints[:, 0] == dim_no + 1][:, 2:4], axis=0)
                    boxes[dim_no, 2:4] = np.max(keypoints[keypoints[:, 0] == dim_no + 1][:, 2:4], axis=0)
                    

                keypoints[:, 2:4] /= self.img_size # scale coordinates to [0..1]
                keypoints[:, 3] = 1 - keypoints[:, 3] #flip y

                boxes[:, :4] /= self.img_size # scale coordinates to [0..1]
                # as we're flipping min and max should be swapped
                flipped_max = 1 - boxes[:, 1] #flip y
                flipped_min = 1 - boxes[:, 3] #flip y
                boxes[:, 3] = flipped_max
                boxes[:, 1] = flipped_min
                assert (boxes[:, 3] >= boxes[:, 1]).all() , "Max coordinate less than min"
                
                boxes[:, 4] = class_id + 1

            self.data.append((img, boxes, keypoints))

            progress_bar.set_description(f'Gathering dataset. max boxes: {self.max_boxes}. {group_id}: labels: {kp_counter}')

        print(f'Entity dataset. Images: {len(self.data)} Max boxes:{self.max_boxes}.')

    def __len__(self):
        if self.limit_number_of_records is not None:
            return min(len(self.data), self.limit_number_of_records)
        return len(self.data)

    def __getitem__(self, index):
        img, boxes, keypoints = self.data[index]
        return img, boxes, keypoints

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
            keypoints = torch.zeros((self.batch_size, self.entities.max_boxes * self.entities.max_keypoints_per_box, 1 + 1 + 1 + self.entities.num_coordinates + 1), dtype=torch.float32)
            # batch * dim_id * [xmin ymin xmax ymax cls_id]
            boxes = torch.zeros((self.batch_size, self.entities.max_boxes , 5), dtype=torch.float32)

            for i in range(len(sample)):
                img, box, keypoint = sample[i]
                ima = img / 255
                ima = np.transpose(ima, (2, 0, 1)) # x, y, channels -> channels, x, y
                ima = torch.from_numpy(ima)
                imgs[i] = ima

                num_keypoints = keypoint.shape[0]
                keypoints[i, :num_keypoints] = torch.from_numpy(keypoint)

                num_boxes = box.shape[0]
                boxes[i, :num_boxes] = torch.from_numpy(box)

            assert (boxes[:, :, 2] - boxes[:, :, 0] >= 0).all()
            assert (boxes[:, :, 3] - boxes[:, :, 1] >= 0).all()
            return imgs, boxes, keypoints

        self.train_loader = torch.utils.data.DataLoader(self.entities, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate, shuffle=False, drop_last=False)
        self.val_loader   = torch.utils.data.DataLoader(self.entities, batch_size=batch_size, sampler=val_sampler, collate_fn=custom_collate, shuffle=False, drop_last=False)
