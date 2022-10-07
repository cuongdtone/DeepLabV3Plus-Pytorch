# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/7/2022

from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import SegmentationDataset, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob


model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)

network.convert_to_separable_conv(model.classifier)

utils.set_bn_momentum(model.backbone, momentum=0.01)

path = 'weights/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
checkpoint = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state"])
model = nn.DataParallel(model)
model.to('cpu')
print("Resume model from %s" % path)
del checkpoint


