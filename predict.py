import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from plot import plot_batch_grid
from pathlib import Path

import config

from yolodwg import DwgKeyPointsModel
from dataset import open_square

def predict(img_path, weights='runs/4/best.weights'):

    checkpoint = torch.load(weights)
    max_points = checkpoint['max_points']
    num_coordinates = checkpoint['num_coordinates']
    model = DwgKeyPointsModel(max_points=max_points, num_coordinates=num_coordinates).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    img, max_size = open_square(img_path, to_size=128)
    img = np.array(img)

    with torch.no_grad():
        # TODO: write img_path to tensor function and use it in loader also
        img_name = Path(img_path).name

        imgs = torch.from_numpy(img) # x, y, channel
        imgs = imgs.float()
        imgs /= 255
        imgs = imgs.unsqueeze(0) # batch=1, x,y,channel
        imgs = imgs.permute(0,3,1,2)
        imgs = imgs.to(config.device)

        out = model(imgs)

        plot_batch_grid(
                        imgs,
                        true_keypoints=None,
                        predictions=out,
                        plot_save_file=f'runs/predict_{img_name}',
                        max_grid_size=1,
                        plot_labels=True)

        return out

if __name__ == "__main__":
    predict(img_path='/home/mk/yolodwg/data/images/stripped_0c2e092f-66b4-4a32-8bea-cfac3c1a4eaf.png')