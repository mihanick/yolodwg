#git clone https://github.com/mihanick/yolodwg
#cd yolov5
pip install -r requirements.txt

# in case of ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# apt update
# apt-get install ffmpeg libsm6 libxext6  -y

# gdown --id 1xMpsc2M8JgN84nh5xBAZj5gNvNgmNrCO
# gdown --id 1zQNX6vJgnGT7h4TfeD__oGVzN8SMQNuA
python create_yolo_dataset_files.py
#tensorboard --logdir runs/train
#python train.py --img 512 --batch 16 --epochs 200 --data dwg.yaml --exist-ok --adam --weights yolov5x.pt
# python train.py --img 512 --batch 24 --epochs 200 --data dwg.yaml  --exist-ok --adam --weights yolov5m.pt
python train.py --img 512 --batch 64 --epochs 200 --data dwg.yaml --exist-ok --adam --weights yolov5s.pt 
python detect.py --weights runs/train/exp/weights/best.pt --source data/dwg/images/train --imgsz 512 --conf-thres 0.1 --iou-thres 0.1 --exist-ok
python debug.py

