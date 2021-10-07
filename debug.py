import config
import torch
from yolodwg import DwgKeyPointsModel, DwgKeyPointsResNet50
from plot import plot_batch_grid

#model = DwgKeyPointsModel(max_points=100)
#model = DwgKeyPointsModel(max_points=max_points, num_coordinates=num_coordinates).to(config.device)

checkpoint = torch.load('runs/best.weights', map_location=config.device)
max_points = checkpoint['max_points']
num_coordinates = checkpoint['num_coordinates']

model = DwgKeyPointsResNet50(requires_grad=True, max_points=max_points)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(config.device)
model.eval()

for i in range(10):
    imgs = torch.rand([4, 3, 128, 128])
    imgs = imgs.to(config.device)
    out = model(imgs)
    #print(torch.mean(out).item())
    plot_batch_grid(
            input_images=imgs,
            true_keypoints=None,
            predictions=out,
            plot_save_file=f'runs/debug_{i}.png')