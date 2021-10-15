import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DwgKeyPointsResNet50(nn.Module):
    '''
    https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/
    '''
    def __init__(self, requires_grad=True, pretrained=True, max_points=100, num_coordinates=2, num_pnt_classes=3, num_img_channels=3):
        super(DwgKeyPointsResNet50, self).__init__()
        self.max_points = max_points
        self.num_coordinates = num_coordinates
        self.num_pnt_classes = num_pnt_classes
        self.num_features = num_coordinates + num_pnt_classes + 1 # x, y, and each pnt cls and pnt_cls==0
        self.output_size = self.max_points * self.num_features
        self.num_channels = num_img_channels
        

        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, self.output_size * 4)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.bn = nn.BatchNorm1d(1024)
        self.do = nn.Dropout()
        # add the final layer
        self.l0 = nn.Linear(100, self.output_size)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model(x)
        x = x.reshape(batch, self.max_points, -1)
        x = F.adaptive_max_pool2d(x, (self.max_points, self.num_features))
        return x
