
from models import CalculatePrediction
from utils.datasets import ListDataset
import torch
from utils.utils import PlotImageAndPrediction
from IPython import display
from IPython.display import Image
from pathlib import Path
import os.path

# https://stackoverflow.com/questions/63268967/configure-pycharm-debugger-to-display-array-tensor-shape
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info

def test_draw():
    dataloader = torch.utils.data.DataLoader(
        ListDataset('./data/dwg/train.txt', max_objects=120), batch_size=4, shuffle=False
    )

    for batch_i, (file_names, imgs, targets) in enumerate(dataloader):
        dets = CalculatePrediction(model=None, batch_of_images=imgs)
        for i, _img in enumerate(imgs):
            file_id = os.path.splitext(os.path.split(file_names[i])[1])[0]
            #display.display(d)

            det = dets[i]
            trg = targets[i]


            display.display(Image("data/dwg/images/test/"+file_id+".png"))

            display.display(PlotImageAndPrediction(image=1 - _img, target=trg, detections=det))
            
        
        

if __name__ == "__main__":
    test_draw()