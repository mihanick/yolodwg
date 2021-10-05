import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from plot import plot_image_prediction_truth
import config

from yolodwg import DwgKeyPointsModel, DwgDataset, open_square

def predict(img_path, weights='best.weights'):

    checkpoint = torch.load(weights)
    max_points = checkpoint['max_points']
    num_coordinates = checkpoint['num_coordinates']
    model = DwgKeyPointsModel(max_points=max_points, num_coordinates=num_coordinates).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    img, max_size = open_square(img_path)
    img = np.array(img)

    with torch.no_grad():
        # TODO: write img_path to tensor function and use it in loader also
        imgs = torch.from_numpy(img) # x, y, channel
        imgs = imgs.float()
        imgs /= 255
        imgs = imgs.unsqueeze(0) # batch=1, x,y,channel
        imgs = imgs.to(config.device)

        out = model(imgs)
        # unflatten output
        out = out.squeeze(0) # remove fake batch
        out = out.reshape(-1, 2)
        prediction = out.detach().cpu().numpy()
        # prediction = prediction.reshape(-1, 2)

        graphic = plot_image_prediction_truth(
                                    input_image=img,
                                    predicted_keypoints=prediction, 
                                    true_keypoints=None)

        graphic.savefig('predicted.png')

        return prediction

if __name__ == "__main__":
    predict(img_path='/home/mk/yolodwg/data/images/stripped_0c2e092f-66b4-4a32-8bea-cfac3c1a4eaf.png')