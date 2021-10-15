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

        self.fc1 = nn.Linear(s*8, self.output_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(bs, -1)
        x = self.dropout(x)

        x = self.fc1(x)
        x = x.view(bs, self.max_points, -1)
        # scale class predicions to sum up to 1
        # x[:, :, 2:] = F.softmax(x[:, :, 2:])

        return x
