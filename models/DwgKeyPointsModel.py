import torch
import torch.nn as nn
import torch.nn.functional as F

class DwgKeyPointsModel(nn.Module):
    def __init__(self, max_points=100, num_coordinates=2, num_pnt_classes=3, num_img_channels=3):
        '''
        Regresses input images to
        flattened max_points*num_coordinates predictions of keypoints
        '''
        super(DwgKeyPointsModel, self).__init__()
        self.max_points = max_points
        self.num_coordinates = num_coordinates
        self.num_pnt_classes = num_pnt_classes
        self.num_features = num_coordinates + num_pnt_classes + 1 # x, y, and each pnt cls and pnt_cls==0
        self.output_size = self.max_points * self.num_features
        self.num_channels = num_img_channels

        s = 16 #vanilla

        self.conv1 = nn.Conv2d(self.num_channels, s*2, kernel_size=5)
        self.conv2 = nn.Conv2d(s*2, s*4, kernel_size=3)
        self.conv3 = nn.Conv2d(s*4, s*8, kernel_size=3)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(128*14*14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 4) #ims*128*4*4
        x = x.reshape(bs, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(bs, self.max_points, -1)
        # scale class predictions to sum up to 1
        # x[:, :, 2:] = F.softmax(x[:, :, 2:])

        return x

def test_model_input_plot():
    import config
    import torch
    from yolodwg import DwgKeyPointsModel, DwgKeyPointsResNet50, non_zero_loss
    from plot import plot_batch_grid

    #model = DwgKeyPointsModel(max_points=max_points, num_coordinates=num_coordinates).to(config.device)

    checkpoint = torch.load('runs/2/best.weights', map_location=config.device)
    max_points = checkpoint['max_points']
    num_pnt_classes = checkpoint['num_pnt_classes']

    model = DwgKeyPointsModel(max_points=max_points, num_pnt_classes=num_pnt_classes)
    #model = DwgKeyPointsResNet50(requires_grad=True, max_points=max_points)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    res = None

    for i in range(100):
        imgs = torch.rand([4, 3, 128, 128])
        imgs = imgs.to(config.device)
        out = model(imgs)
        if res is None:
            res = out
        else:
            res = torch.cat([res, out])
        #plot_batch_grid(
        #        input_images=imgs,
        #        true_keypoints=None,
        #        predictions=out,
        #        plot_save_file=f'runs/debug_{i}.png')

    print(torch.min(res).item())
    print(torch.mean(res).item())
    print(torch.max(res).item(), '\n')

if __name__ == "__main__":
    test_model_input_plot()