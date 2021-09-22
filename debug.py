# https://stackoverflow.com/questions/63268967/configure-pycharm-debugger-to-display-array-tensor-shape
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info


from processing import build_data
#df, ids = build_data(rebuild=True, img_size=512, limit_records=100)
from create_yolo_dataset_files import create_yolo_dataset_files
create_yolo_dataset_files(rebuild=True, limit_records=200)