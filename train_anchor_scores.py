import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler

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
        os.system("mkdir -p %s" % options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s" % options.test_dir)
        pass

    config = PlaneConfig(options)
    dataset = ScenenetRgbdDataset(options, config, split='train', random=False, load_scores=True)
    # dataset = ScenenetRgbdDataset(options, config, split='train', random=False)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    dataset_test = ScenenetRgbdDataset(options, config, split='test', random=False, load_scores=True)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=8)

    # profiler = AdvancedProfiler()
    profiler = SimpleProfiler()
    model = AnchorScores(options, config)
    # limit_val_batches=10, val_check_interval=0,
    # trainer = pl.Trainer(gpus=1, limit_train_batches=20, max_epochs=1, limit_val_batches=1, profiler=profiler)
    trainer = pl.Trainer(gpus=1, max_epochs=12, limit_val_batches=1, val_check_interval=500)
    # trainer = pl.Trainer(gpus=1, max_epochs=10, limit_val_batches=1, val_check_interval=500,
    #                      resume_from_checkpoint='lightning_logs/version_1/checkpoints/epoch=9.ckpt')
    # trainer = pl.Trainer(gpus=1, limit_val_batches=10, val_check_interval=500,
    #                      resume_from_checkpoint='lightning_logs/version_3/checkpoints/epoch=6.ckpt')
    trainer.fit(model, train_loader, test_loader)


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