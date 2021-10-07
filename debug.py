import config
import torch
from yolodwg import DwgKeyPointsModel
from plot import plot_batch_grid

model = DwgKeyPointsModel(max_points=100)
checkpoint = torch.load('runs/best.weights', map_location=config.device)
max_points = checkpoint['max_points']
num_coordinates = checkpoint['num_coordinates']
model = DwgKeyPointsModel(max_points=max_points, num_coordinates=num_coordinates).to(config.device)
model.load_state_dict(checkpoint['model_state_dict'])

for i in range(10):
    imgs = torch.rand([4, 128, 128, 3])
    imgs = imgs.to(config.device)
    out = model(imgs)
    #print(torch.mean(out).item())
    plot_batch_grid(
            input_images=imgs,
            true_keypoints=None,
            predictions=out,
            plot_save_file=f'runs/debug_{i}.png')