'''
Functions to plot predictions and ground truth over input imagess
'''
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import config
from tqdm import tqdm

def plot_loader_predictions(loader, model, epoch=0, plot_folder=None):
    
    if plot_folder is None:
        return

    model.eval()
    figs = []
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(loader):

            imgs = imgs.to(config.device)
            targets = targets.to(config.device)
            targets = targets[:, :, -3:-1]

            out = model(imgs)
            out = out.view((out.shape[0], model.max_points, -1))

            fig = plot_batch_grid(
                        input_images=imgs,
                        true_keypoints=targets,
                        predictions=out,
                        plot_save_file=f'{plot_folder}/prediction_{epoch}_{i}.png')
            figs.append(fig)
            if i > 3:
                break
    return figs
    

def plot_batch_grid(input_images, true_keypoints=None, predictions=None, plot_save_file=None):
    '''
    input images torch.tensor(n_batches, img_size, img_size, channels)
    true_keypoints torch.tensor(n_batches, max_points * (classes + n_coords)
    predictions  torch.tensor(n_batches, max_points * (classes + n_coords)
    plot_save_dir - directory to save plots

    returns last plot
    '''
    batch_size = input_images.shape[0]
    grid_size = int(math.sqrt(batch_size))
    grid_size = min(3, grid_size)

    fig = plt.figure(figsize=(10, 10))

    #print(f'Plotting images grid:')
    for i, img in enumerate(input_images):
        if i + 1 > grid_size * grid_size:
            break
        np_img = img.detach().cpu().numpy()
        if true_keypoints is not None:
            tkp = true_keypoints[i].detach().cpu().numpy()
            tkp = np.reshape(tkp, (-1, 2))
        if predictions is not None:
            pred = predictions[i].detach().cpu().numpy()
            pred = np.reshape(pred, (-1, 2))

        plt.subplot(grid_size, grid_size, i + 1)
        plt.axis('off')

        plot_image_prediction_truth(np_img, pred, tkp)

    if plot_save_file is not None:
        plt.savefig(plot_save_file)
    return fig

def plot_image_prediction_truth(input_image, predicted_keypoints=None, true_keypoints=None):
    '''
    Plots the predicted keypoints and
    actual keypoints for first image from batch

    input_image single image np.array.size(image_size, image_size, num_channels=num_channels)
    predicted_keypoints keypoint predictions for image np.array.size(max_points, 2[x,y])
    true_keypoints torch.tensor ground truth for image np.array.size(max_points, 2[x,y])
    y coordinates are calculated from top left corner
    x,y are in [0..1] range
    optionally saves graph in specified plot_save_path

    returns pyplot graph
    '''

    img_size = input_image.shape[0]
    #input_image *= 255
    #input_image
    plt.imshow(input_image)

    if predicted_keypoints is not None:
        output_keypoint = predicted_keypoints * img_size
        for p in range(output_keypoint.shape[0]):
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')
            plt.text(output_keypoint[p, 0], output_keypoint[p, 1], f'{p}')

    if true_keypoints is not None:
        orig_keypoint = true_keypoints * img_size
        for p in range(orig_keypoint.shape[0]):
            plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'g.')
            plt.text(orig_keypoint[p, 0], orig_keypoint[p, 1], f'{p}')
