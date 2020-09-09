import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os
from tqdm import tqdm
import numpy as np
import cv2
import sys
import shutil

from models.model import *
from datasets.scenenet_rgbd_stereo_dataset import *

from utils import *
from visualize_utils import *
from evaluate_utils import *
from options import parse_args
from config import PlaneConfig


def train(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    config = PlaneConfig(options)
    dataset = ScenenetRgbdDataset(options, config, split='train', random=False)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = AnchorScores(options, config)
    trainer = pl.Trainer(gpus=1, )
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    args = parse_args()

    args.keyname = 'planercnn'

    args.keyname += '_' + args.anchorType
    if args.dataset != '':
        args.keyname += '_' + args.dataset
        pass
    if args.trainingMode != 'all':
        args.keyname += '_' + args.trainingMode
        pass
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass

    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = args.dataFolder + '/test/' + args.keyname

    if False:
        writeHTML(args.test_dir, ['image_0', 'segmentation_0', 'depth_0', 'depth_0_detection', 'depth_0_detection_ori'],
                  labels=['input', 'segmentation', 'gt', 'before', 'after'], numImages=20, image_width=160,
                  convertToImage=True)
        exit(1)

    os.system('rm ' + args.test_dir + '/*.png')
    print('keyname=%s task=%s started' % (args.keyname, args.task))

    train(args)