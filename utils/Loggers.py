# YOLOv5 experiment logging utils

import os
import warnings
from threading import Thread

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.general import colorstr, emojis
from utils.plots import plot_images, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ('csv', 'tb')  # text-file, TensorBoard

class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, data_dict=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.data_dict = data_dict
        self.logger = logger  # for printing results to console
        self.include = include
        self.csv = True
        self.tb = None
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary

    def start(self):
        self.csv = True  # always log to csv

        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include:
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")

            #https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
            self.tb = SummaryWriter(str(s))

        return self

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots):
        # Callback runs on train batch end
        if plots:
            if ni == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')  # suppress jit trace warning
                    self.tb.add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()

    def on_train_val_end(self, mloss, results, lr, epoch, best_fitness, fi):
        # Callback runs on val end during training
        vals = list(mloss) + list(results) + lr
        keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                'x/lr0', 'x/lr1', 'x/lr2']  # params
        x = {k: v for k, v in zip(keys, vals)}  # dict

        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)  # TensorBoard

    def on_train_end(self, last, best, plots):
        # Callback runs on training end
        if plots:
            plot_results(dir=self.save_dir)  # save results.png
        files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter
