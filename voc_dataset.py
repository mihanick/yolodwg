# try to train on the data from
# https://github.com/argusswift/YOLOv4-pytorch

import torch
from torch.utils.data import Dataset, SequentialSampler, SubsetRandomSampler

import json
import pandas as pd
import cv2
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

    if src.mode=='RGBA':
        # http://espressocode.top/python-pil-paste-and-rotate-method/
        # image is pasted using top left coords
        # while drawing coordinate system 0,0 is bottom left
        # so we have to shift by y size difference
        # 
        # remove black background
        # https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
        trg.paste(src, box=(0, max_size - src.size[1]), mask=src.split()[3])
    else:
        trg.paste(src)

    trg = trg.convert('RGB')
    if to_size:
        trg = trg.resize((to_size, to_size))

    #nsrc = np.array(src)
    ntrg = np.array(trg)

    #src.save('src.png')
    #trg.save('trg.png')

    return ntrg, max_size

VOC_DATA = {
    "NUM": 20,
    "CLASSES": [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ],
}

class VocDataset(Dataset):
    '''
    Dataset of images - labels (points of dimensions)
    '''
    
    def load_voc_annotations(anno_path):
        with open(anno_path, "r") as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        assert len(annotations) > 0, "No images found in {}".format(anno_path)

        return annotations

    def parse_annotations(self, annotations):
        data = []
        for annotation in annotations:
            anno = annotation.strip().split(" ")

            img_path = anno[0]

            bboxes = np.array(
                [list(map(float, box.split(","))) for box in anno[1:]]
            )
            boxes = bboxes.reshape(-1,5)
            numboxes = boxes.shape[0]
            if numboxes > self.max_boxes:
                self.max_boxes = numboxes
            keypoints = np.zeros([numboxes * len(self.pnt_classes), 1 + 1 + 1 + self.num_coordinates + 1], dtype=float)

            # debug:
            # assert boxes[:,:4].max()>0.001

            data.append((img_path, boxes, keypoints))
        return data

    def __init__(self, data_label_txt, limit_number_of_records=None):
        '''

        '''
        
        self.max_boxes = 1
        self.max_keypoints_per_box = 1
        self.img_size = 256
        self.num_image_channels = 3

        self.limit_number_of_records = limit_number_of_records

        self.classes = VOC_DATA['CLASSES']
        self.pnt_classes = ['None']
        self.coordinates = ['X', 'Y']

        self.num_classes = len(self.classes)
        self.num_pnt_classes = len(self.pnt_classes)
        self.num_coordinates = len(self.coordinates)

        self.ids = []
        annotations = VocDataset.load_voc_annotations(data_label_txt)
        self.data = self.parse_annotations(annotations)

    def __len__(self):
        if self.limit_number_of_records is not None:
            return min(len(self.data), self.limit_number_of_records)
        return len(self.data)

    def __getitem__(self, index):
        img, boxes, keypoints = self.data[index]

        # assert boxes[:, :4].max()>1

        return img, boxes, keypoints

class VocDataloader:
    def __init__(self,
                    train_entities,
                    val_entities,
                    batch_size=32):

        self.batch_size = batch_size
        self.train_entities = train_entities
        self.val_entities = val_entities

        self.num_image_channels = self.train_entities.num_image_channels
        self.num_classes = self.train_entities.num_classes
        self.img_size = self.train_entities.img_size
        self.max_boxes = self.train_entities.max_boxes
        self.max_keypoints_per_box = self.train_entities.max_keypoints_per_box
        self.num_coordinates = self.train_entities.num_coordinates
        self.num_image_channels = self.train_entities.num_image_channels

        # https://stackoverflow.com/questions/64586575/adding-class-objects-to-pytorch-dataloader-batch-must-contain-tensors
        def custom_collate(sample):
            imgs = torch.zeros((self.batch_size, self.num_image_channels, self.img_size, self.img_size), dtype=torch.float32)

            # batch * pnt_id * [dim_id, pnt_id, x, y, good, cls_id]
            keypoints = torch.zeros((self.batch_size, self.train_entities.max_boxes * self.train_entities.max_keypoints_per_box, 1 + 1 + 1 + self.train_entities.num_coordinates + 1), dtype=torch.float32)
            # batch * dim_id * [xmin ymin xmax ymax cls_id]
            boxes = torch.zeros((self.batch_size, self.train_entities.max_boxes , 5), dtype=torch.float32)

            for i in range(len(sample)):
                img_path, box, keypoint = sample[i]

                img, this_img_max_size = open_square(img_path, to_size=self.img_size)

                ima = img / 255
                ima = np.transpose(ima, (2, 0, 1)) # x, y, channels -> channels, x, y
                ima = torch.from_numpy(ima)
                imgs[i] = ima

                num_keypoints = keypoint.shape[0]
                keypoints[i, :num_keypoints] = torch.from_numpy(keypoint)

                num_boxes = box.shape[0]
                boxes[i, :num_boxes] = torch.from_numpy(box)

                # scale coordinates to 0..1 this should be done on copy of data, as it causes data corruption each iteration
                # TODO: no scale
                boxes[i, :num_boxes, :4] *= self.img_size / this_img_max_size

            # max coords should be larger than min coords by dimension
            # assert (boxes[:, :, 2] - boxes[:, :, 0] >= 0).all()
            # assert (boxes[:, :, 3] - boxes[:, :, 1] >= 0).all()
            return imgs, boxes, None #keypoints=None for VOC

        self.train_loader = torch.utils.data.DataLoader(self.train_entities, batch_size=batch_size, collate_fn=custom_collate, shuffle=False, drop_last=False)
        self.val_loader   = torch.utils.data.DataLoader(self.val_entities, batch_size=batch_size, collate_fn=custom_collate, shuffle=False, drop_last=False)


if __name__ == "__main__":

    train_voc_dataset = VocDataset('data/voc/train_annotation.txt')
    val_voc_dataset = VocDataset('data/voc/test_annotation.txt')
    voc_dataloader = VocDataloader(train_voc_dataset,val_voc_dataset, batch_size=4)

    for i, (
        img,
        box,
        _
    ) in enumerate(voc_dataloader.train_loader):
        
        print(img.shape)
        print(box.shape)
        break