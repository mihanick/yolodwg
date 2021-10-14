'''
Functions to plot predictions and ground truth over input imagess
'''
import numpy as np
import math
import torch
import config

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

def plot_loader_predictions(loader, model, epoch=0, plot_folder=None, limit_number_of_plots=3):
    
    if plot_folder is None:
        return

    model.eval()
    figs = []
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(loader):

            imgs = imgs.to(config.device)
            targets = targets.to(config.device)

            out = model(imgs)

            fig = plot_batch_grid(
                        input_images=imgs,
                        true_keypoints=targets,
                        predictions=out,
                        plot_save_file=f'{plot_folder}/checkpoint_{epoch}_{i}.png')
            figs.append(fig)
            if limit_number_of_plots is not None:
                if i > limit_number_of_plots:
                    break
    return figs
    

def plot_batch_grid(input_images, true_keypoints=None, predictions=None, plot_save_file=None, max_grid_size=2, plot_labels=False):
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
        if i + 1 > grid_size * grid_size:
            break
        input_image = img.detach().cpu().numpy()
        input_image = np.transpose(input_image, (1, 2, 0)) #channels,x,y -> x,y,channels
        img_size = input_image.shape[0]

        plt.subplot(grid_size, grid_size, i + 1)
        plt.axis('off')

        plt.imshow(input_image)

        if true_keypoints is not None:
            tkp = true_keypoints[i, :, 2:4].detach().cpu().numpy()
            if (tkp.shape[1] != 0): # Handle plot of empty ground_truth
                tkp = tkp * img_size # scale coordinates to img size
                tkp[:, 1] = img_size - tkp[:, 1] # flip y

                for p in range(tkp.shape[0]):
                    plt.plot(tkp[p, 0], tkp[p, 1], 'g.')
                    if plot_labels:
                        plt.text(tkp[p, 0], tkp[p, 1], f'{true_keypoints[i, p, 5]}') # plt pnt class

        if predictions is not None:
            predicted_keypoints = predictions[i].detach().cpu().numpy()

            for p in range(predicted_keypoints.shape[0]):
                p_coords = predicted_keypoints[p,:2]
                p_coords *= img_size #scale coordinates to img_size
                p_coords[1] = img_size - p_coords[1] # flip y, as pyplot starts plotting from top lef, but coords are not from bottom left

                p_pnt_cls=predicted_keypoints[p, 2:]
                p_pnt_cls = np.exp(p_pnt_cls)/sum(np.exp(p_pnt_cls)) # softmax to sum up to 1
                predicted_class = np.argmax(p_pnt_cls)
                confidence = p_pnt_cls[predicted_class]

                if predicted_class > 0:
                    plt.plot( p_coords[0], p_coords[1], 'r.')
                    if plot_labels:
                        plt.text(p_coords[0], p_coords[1], f'{predicted_class}@{confidence * 100:.0f}%')
                

    if plot_save_file is not None:
        plt.savefig(plot_save_file)
    return fig

