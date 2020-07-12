#   Copyright (c) 2020 Yaoyao Liu. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from dataloader.samplers import CategoriesSampler
from utils.misc import *
from utils.gpu_tools import occupy_memory, set_gpu
from tensorboardX import SummaryWriter
import tqdm
import time
import importlib
from trainer.meta_trainer import MetaTrainer

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'tieredimagenet'])
parser.add_argument('-datadir', type=str, default=None)
parser.add_argument('-set',type=str,default='val',choices=['test','val'])
parser.add_argument('-mode',type=str,default='meta_train',choices=['pre_train', 'meta_train', 'meta_eval'])
parser.add_argument('-bs', type=int, default=1,help='batch size')
parser.add_argument('-max_epoch', type=int, default=100)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-temperature', type=float, default=8)
parser.add_argument('-step_size', type=int, default=10)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-val_frequency',type=int,default=100)
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=5)
parser.add_argument('-query', type=int, default=16)
parser.add_argument('-val_episode', type=int, default=3000)
parser.add_argument('-backbone', type=str, default='resnet12', choices=['wrn', 'resnet12'])
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-save_all',action='store_true',help='save models on each epoch')
parser.add_argument('-meta_update',type=str,default='mtl',choices=['ft','mtl'])
parser.add_argument('--hyperprior_init_mode', type=str, default='LAS', choices=['LAS', 'EQU'])
parser.add_argument('--hyperprior_combination_softweight', type=float, default=1e-4)
parser.add_argument('--hyperprior_basestep_softweight', type=float, default=1e-4)
parser.add_argument('-base_init',type=str,default='feature',choices=['feature'])
parser.add_argument('-base_epoch', type=int, default=100)
parser.add_argument('-base_lr', type=float, default=0.1)
parser.add_argument('-base_lr_encoder', type=float, default=0.01)
parser.add_argument('-lr_combination', type=float, default=1e-6)
parser.add_argument('-lr_combination_hyperprior', type=float, default=1e-6)
parser.add_argument('-lr_basestep', type=float, default=1e-6)
parser.add_argument('-lr_basestep_hyperprior', type=float, default=1e-6)
parser.add_argument('-sleep',type=float,help='hour',default=0.0)
parser.add_argument('-gpu', default='0')
parser.add_argument('-gpu_occupy',action='store_true')
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-num_workers', type=int, default=8)

args = parser.parse_args()
pprint(vars(args))

if args.seed==0:
    print ('Random mode.')
    torch.backends.cudnn.benchmark = True
else:
    import random
    print ('Fixed random seed:', args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_gpu=set_gpu(args)
args.num_gpu=num_gpu
if args.gpu_occupy:
    occupy_memory(args.gpu)
    print('Occupy GPU memory in advance.')

trainer = MetaTrainer(args)
if args.mode == 'meta_train':
    print('Start meta-train phase.')
    trainer.train()
    print('Start meta-test phase.')
    trainer.eval()
elif args.mode == 'meta_eval':
    print('Start meta-test phase.')
    trainer.eval()
elif args.mode == 'pre_train':
    print('Start pre-train phase.')
    trainer.pre_train()
