'''
Functions to plot predictions and ground truth over input imagess
'''
import numpy as np
import math
import torch
import config

import matplotlib.pyplot as plt
import matplotlib

from models.utils import nms_conf_suppression, plot_boxes_cv2
matplotlib.style.use('ggplot')

def plot_loader_predictions(loader, model, epoch=0, conf_thresh=0.1, nms_thresh=0.2, plot_folder=None, limit_number_of_plots=2):
    if plot_folder is None:
        return

    model.eval()
    figs = []

    with torch.no_grad():
        for i, (imgs, boxes, true_keypoints) in enumerate(loader):
            imgs = imgs.to(config.device)

            out = model(imgs)
            predicted_boxes = out[0]
            confidences = out[1]
            predictions = nms_conf_suppression(box_array=predicted_boxes, confs=confidences, conf_thresh=conf_thresh, nms_thresh=nms_thresh)

            fig = plot_batch_grid(
                        input_images=imgs,
                        true_boxes=boxes,
                        true_keypoints=true_keypoints,
                        predictions=predictions,
                        plot_save_file=f'{plot_folder}/checkpoint_{epoch}_{i}.png')
            figs.append(fig)
            plt.close()
            if limit_number_of_plots is not None:
                if i > limit_number_of_plots:
                    break
    return figs
    

def plot_batch_grid(input_images, true_boxes=None, true_keypoints=None, predictions=None, plot_save_file=None, max_grid_size=2, plot_labels=False):
    '''
    input images torch.tensor(n_batches, channels, img_size, img_size)
    true_keypoints torch.tensor(n_batches, max_points * (classes + n_coords)
    predictions  torch.tensor(n_batches, max_points * (classes + n_coords)
    plot_save_dir - directory to save plots

    returns last plot
    '''
    batch_size = input_images.shape[0]
    grid_size = int(math.sqrt(batch_size))
    grid_size = min(max_grid_size, grid_size)

    fig = plt.figure(figsize=(7, 7))

    #print(f'Plotting images grid:')
    for i, img in enumerate(input_images):
        # plot only number of images to place in grid
        if i + 1 > grid_size * grid_size:
            break
        # img = img.clip(min=0, max=1)
        input_image = img.detach().cpu().numpy()
        
        input_image = np.transpose(input_image, (1, 2, 0)) #channels,x,y -> x,y,channels
        img_size = input_image.shape[0]

        plt.subplot(grid_size, grid_size, i + 1)
        plt.axis('off')

        if predictions is not None:
            input_image = plot_boxes_cv2(img=input_image, boxes=predictions[i], color=(1,0,0)) #red
            plt.imshow(input_image)
        if true_boxes is not None:
            input_image = plot_boxes_cv2(img=input_image, boxes=true_boxes[i], color=(0,1,0)) #green
            plt.imshow(input_image)

        if true_keypoints is not None:
            tkp = true_keypoints[i, :, 2:4].detach().cpu().numpy()
            if (tkp.shape[1] != 0): # Handle plot of empty ground_truth
                # tkp = tkp * img_size # scale coordinates to img size
                # tkp[:, 1] = img_size - tkp[:, 1] # flip y

                for p in range(tkp.shape[0]):
                    true_pnt_cls = true_keypoints[i, p, 5]
                    if true_pnt_cls > 0:
                        plt.plot(tkp[p, 0], tkp[p, 1], 'g.')
                        if plot_labels:
                            plt.text(tkp[p, 0], tkp[p, 1], f'{true_pnt_cls}') # plt pnt class

    if plot_save_file is not None:
        plt.savefig(plot_save_file)
    return fig

