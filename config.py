import torch
from torch._C import device

# https://stackoverflow.com/questions/63268967/configure-pycharm-debugger-to-display-array-tensor-shape
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    shape = tensor.shape
    sh_str = repr(shape)[6:]
    # show value in one-item tensor
    if tensor.ndim == 0:
        sh_str = str(tensor.item())

    return sh_str + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info

use_cuda_if_available = True
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')