import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN

def get_keypoint_rcnn(pretrained=True, num_classes=2, box_detections_per_img=100, num_keypoints=17):

    # load a pre-trained model for classification and return
    # only the features
    backbone = mobilenet_v2(pretrained=pretrained).features
    
    # KeypointRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280
   
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
   
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be ['0']. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
   
    keypoint_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                             output_size=14,
                                                             sampling_ratio=2)
    # put the pieces together inside a KeypointRCNN model
    model = KeypointRCNN(backbone,
                         max_size=512,
                         num_classes=num_classes,
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         keypoint_roi_pool=keypoint_roi_pooler,
                         box_detections_per_img=box_detections_per_img,
                         num_keypoints=num_keypoints)

    return model



class DwgKeyPointsRcnn(nn.Module):
    def __init__(self, requires_grad=True, pretrained=True, max_labels=15, num_coordinates=2, num_pnt_classes=3, num_img_channels=3):
        '''
        max_labels max number of dimensions boxes per image

        '''
        super(DwgKeyPointsRcnn, self).__init__()

        self.max_labels = max_labels
        self.num_coordinates = num_coordinates
        self.num_pnt_classes = num_pnt_classes
        self.num_features = num_coordinates + num_pnt_classes + 1 # x, y, and each pnt cls and pnt_cls==0
        self.max_points = self.max_labels * self.num_pnt_classes
        self.num_channels = num_img_channels

        #from torchvision.models.detection import keypointrcnn_resnet50_fpn
        #self.model = keypointrcnn_resnet50_fpn(
        #                                        pretrained=False,  # Won't able to load pretrained, as it requires num_classes==17
        #                                        num_keypoints=num_pnt_classes, 
        #                                        num_classes=2,
        #                                        pretrained_backbone=True, 
        #                                        trainable_backbone_layers=None)

        self.model = get_keypoint_rcnn(
                                        pretrained=pretrained,
                                        num_classes=2, #label classes including background. We have only AlignedDimension, so its 2
                                        box_detections_per_img=max_labels,
                                        num_keypoints=num_pnt_classes)

        if False:
            if requires_grad == True:
                for param in self.model.parameters():
                    param.requires_grad = True
            elif requires_grad == False:
                for param in self.model.parameters():
                    print(param.requires_grad)
                    param.requires_grad = False

    def forward(self, x, targets=None):
        bs, _, img_size, _ = x.shape

        #expected input as list of tensor images
        xinput = [im for im in x]
        if self.training:
            # https://pytorch.org/vision/stable/models.html#keypoint-r-cnn
            all_targets = []
            for target in targets: # batches

                reshape_target = target.view(self.max_labels, self.num_pnt_classes, self.num_features)

                # number of labels is number of non-zero label classes
                valid_indices_this_image = reshape_target[:, 0, 5] > 0

                # everything is AlignedDimension
                labels = torch.ones(self.max_labels, device=target.device, dtype=torch.int64) 
                
                # take coordinates
                keypoints = reshape_target[:, :, 2:4] 
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
                # add 1 column (according to rcnn) it's keypoint visibility
                # should be 1, otherwise keypoint loss will be 0
                keypoints = F.pad(keypoints, pad=(0,1,0,0), value=1)
                # scale coordinates to image size
                keypoints[:, :, :2] *= img_size

                # get label bounds from keypoints
                #dim=4 is for coordinates x1y1x2y2
                boxes = torch.zeros(self.max_labels, 4, device=target.device)
                # x1y1 = xmin, ymin of all keypoints of one label (dim=1)
                boxes[:, :2] = torch.min(keypoints[:, :, 0:2], dim=1).values
                # x2y2 = xmax, ymax
                boxes[:, 2:] = torch.max(keypoints[:, :, 0:2], dim=1).values
                
                 #'rcnn will only accept positive width of bound'
                positive_bound_widths_indices =  (boxes[:, 2] - boxes[:, 0]) > 0
                 #'rcnn will only accept positive height of bound'
                positive_bound_heights_indices = (boxes[:, 3] - boxes[:, 1]) > 0
                valid_indices_this_image = torch.logical_and(valid_indices_this_image, positive_bound_heights_indices)
                valid_indices_this_image = torch.logical_and(valid_indices_this_image, positive_bound_widths_indices)

                # limit input to number of non-zero classvalid labels
                labels = labels[valid_indices_this_image]
                keypoints =  keypoints[valid_indices_this_image]
                boxes = boxes[valid_indices_this_image]

                all_targets.append({
                    'labels' : labels,
                    'keypoints' : keypoints,
                    'boxes' : boxes 
                })

            out = self.model(xinput, all_targets)

            return out
        else:
            out = self.model(xinput)

            catout = []
            for pred in out:
                #print(pred['keypoints'].shape)
                #print(pred['keypoints_scores'].shape)

                #print(pred['boxes'].shape)
                #print(pred['labels'].shape)
                #print(pred['scores'].shape)
                coords = pred['keypoints'][:, :, :2] / img_size
                coords = coords.reshape(-1, 2)

                # TODO: Keypoint class prediction, class is 1
                coord_scores = pred['keypoints_scores']
                cls_scores = torch.zeros( self.max_labels * self.num_pnt_classes, self.num_pnt_classes + 1, device=coords.device)
                cls_scores[:,1] = 1

                img_prd = torch.cat([coords, cls_scores], dim=1)

                catout.append(img_prd)

            return torch.stack(catout)

if __name__ == "__main__":
    model = DwgKeyPointsRcnn()
    device = torch.device('cuda')
    model.to(device)
    model.train()
    
    x = torch.rand(4, 3, 512, 512, device=device)
    target = torch.rand(4, 45, 6, device=device)
    predictions = model(x, target)
    print(predictions)