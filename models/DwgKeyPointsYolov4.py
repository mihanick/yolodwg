'''
source from https://github.com/Tianxiaomo/pytorch-YOLOv4
'''
from numpy.lib.arraypad import _view_roi
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d
from .torch_utils import *
from .yolo_layer import YoloLayer
import config

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if self.training:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')
        else:
            #B = x.data.size(0)
            #C = x.data.size(1)
            #H = x.data.size(2)
            #W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self, size=32):
        super().__init__()
        n=size

        self.conv1 = Conv_Bn_Activation(3, n, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(n, 2*n, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(2*n, 2*n, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(2*n, 2*n, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(2*n, n, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(n, 2*n, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(2*n, 2*n, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(4*n, 2*n, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self, size=64):
        super().__init__()
        n = size
        self.conv1 = Conv_Bn_Activation(n, 2*n, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(2*n, n, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(2*n, n, 1, 1, 'mish')

        self.resblock = ResBlock(ch=n, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(n, n, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(2*n, 2*n, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        n=size
        self.conv1 = Conv_Bn_Activation(n, 2*n, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(2*n, n, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(2*n, n, 1, 1, 'mish')

        self.resblock = ResBlock(ch=n, nblocks=8)
        self.conv4 = Conv_Bn_Activation(n, n, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(2*n, 2*n, 1, 1, 'mish')
        
    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self, size=256):
        super().__init__()
        n=size
        self.conv1 = Conv_Bn_Activation(n, 2*n, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(2*n, n, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(2*n, n, 1, 1, 'mish')

        self.resblock = ResBlock(ch=n, nblocks=8)
        self.conv4 = Conv_Bn_Activation(n, n, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(2*n, 2*n, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self, size=512):
        super().__init__()
        n=size
        self.conv1 = Conv_Bn_Activation(n, 2*n, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(2*n, n, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(2*n, n, 1, 1, 'mish')

        self.resblock = ResBlock(ch=n, nblocks=4)
        self.conv4 = Conv_Bn_Activation(n, n, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(2*n, 2*n, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self, size=128):
        super().__init__()

        n=size

        self.conv1 = Conv_Bn_Activation(8*n, 4*n, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(4*n, 8*n, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(8*n, 4*n, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(16*n, 4*n, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(4*n, 8*n, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(8*n, 4*n, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(4*n, 2*n, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(4*n, 2*n, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(4*n, 2*n, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(2*n, 4*n, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(4*n, 2*n, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(2*n, 4*n, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(4*n, 2*n, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(2*n, n, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(2*n, n, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(2*n, n, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(n, 2*n, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(2*n, n, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(n, 2*n, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(2*n, n, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size())
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size())
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, size=128):
        super().__init__()
        n=size

        self.conv1 = Conv_Bn_Activation(n, 2*n, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(2*n, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
                                anchor_mask=[0, 1, 2], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(n, 2*n, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(4*n, 2*n, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(2*n, 4*n, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(4*n, 2*n, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(2*n, 4*n, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(4*n, 2*n, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(2*n, 4*n, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(4*n, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(2*n, 4*n, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(8*n, 4*n, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(4*n, 8*n, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(8*n, 4*n, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(4*n, 8*n, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(8*n, 4*n, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(4*n, 8*n, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(8*n, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo3 = YoloLayer(
                                anchor_mask=[6, 7, 8], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        
        if self.training:
            return [x2, x10, x18]
        else:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            #DEBUG:
            # assert not y1[0].mean().isnan()
            # assert not y2[0].mean().isnan()
            # assert not y3[0].mean().isnan()

            return get_region_boxes([y1, y2, y3])


class Yolov4(nn.Module):
    def __init__(self, n_classes=80, size=32):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3
        n=size

        # backbone
        self.down1 = DownSample1(size=n)
        self.down2 = DownSample2(size=2*n)
        self.down3 = DownSample3(size=4*n)
        self.down4 = DownSample4(size=8*n)
        self.down5 = DownSample5(size=16*n)
        # neck
        self.neck = Neck(size=4*n)

        # head
        self.head = Yolov4Head(output_ch, n_classes, size=4*n)


    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output

class DwgKeyPointsYolov4(nn.Module):
    def __init__(self,
            max_boxes=30,
            n_box_classes=1,
            num_coordinates=2,
            num_pnt_classes=3,
            num_img_channels=3,
            pretrained=True,
            requires_grad=False,
            size=32):
        '''
        
        '''
        super(DwgKeyPointsYolov4, self).__init__()

        self.max_boxes = max_boxes
        self.num_coordinates = num_coordinates
        self.num_pnt_classes = num_pnt_classes
        self.num_channels = num_img_channels
        self.n_box_classes = n_box_classes

        self.n_anchors = 3

        self.model = Yolov4(n_classes=n_box_classes, size=size)

        if pretrained:
            checkpoint = torch.load('yolov4.pth', map_location=config.device)
            # Pretrained can only work on 80 classes size 32
            self.model = Yolov4(n_classes=80, size=32)
            load_cp = {}
            for k in checkpoint:
                new_key = k
                if 'neek' in k:
                    new_key = k.replace('neek', 'neck')
                load_cp[new_key] = checkpoint[k]
            self.model.load_state_dict(load_cp)

            if not requires_grad:
                for param in self.model.parameters():
                    param.requires_grad = False

            # head will be trained anyways
            output_ch = (4 + 1 + self.n_box_classes) * 3
            self.model.head = Yolov4Head(output_ch, self.n_box_classes, size=4*size)

    def forward(self, x):
        xin = self.model(x)
        return xin

if __name__ == "__main__":
    pass