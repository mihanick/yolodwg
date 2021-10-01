'''
main runner from console
'''
# https://debuggercafe.com/getting-started-with-facial-keypoint-detection-using-pytorch/

from main import run
import argparse
import torch



# https://stackoverflow.com/questions/63268967/configure-pycharm-debugger-to-display-array-tensor-shape
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/dwg/images", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/dwg.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="config/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="config/dwg.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.01, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

run(
    use_cuda=opt.use_cuda,
    class_path=opt.class_path,
    data_config_path=opt.data_config_path,
    model_config_path=opt.model_config_path,
    weights_path=opt.weights_path,
    batch_size=opt.batch_size,
    epochs=opt.epochs,
    checkpoint_interval=opt.checkpoint_interval,
    checkpoint_dir=opt.checkpoint_dir,
    n_cpu=opt.n_cpu
)