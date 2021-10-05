import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from plot import plot_image_prediction_truth, plot_batch_grid, plot_loader_predictions
import config

from yolodwg import DwgKeyPointsModel, DwgDataset, open_square

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


# https://stackoverflow.com/questions/63268967/configure-pycharm-debugger-to-display-array-tensor-shape
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info


