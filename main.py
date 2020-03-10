import argparse
import os.path as osp
import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.misc import pprint, ensure_path, create_dirs, get_logger, set_random_seed
from utils.gpu_tools import occupy_memory, set_gpu
from trainer.meta import MetaTrainer
from trainer.meta_sib import MetaTrainerSIB
from trainer.pre import PreTrainer
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### Basic Config
    parser.add_argument('--baseline', type=str, default='MTL', choices=['MTL', 'SIB'])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--train_query', type=int, default=15)
    parser.add_argument('--val_query', type=int, default=1)
    parser.add_argument('--label', type=str, default='Exp01')
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImageNet', 'FC100'])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu_occupy', type=bool, default='True')
    parser.add_argument('--hyperprior_combination_softweight', type=float, default=1e-4)
    parser.add_argument('--hyperprior_basestep_softweight', type=float, default=1e-4)
    parser.add_argument('--hyperprior_arch', type=str, default='FC', choices=['FC', 'LSTM'])
    parser.add_argument('--lr_combination', type=float, default=1e-6)
    parser.add_argument('--lr_combination_hyperprior', type=float, default=1e-6)
    parser.add_argument('--lr_basestep', type=float, default=1e-6)
    parser.add_argument('--lr_basestep_hyperprior', type=float, default=1e-6)
    parser.add_argument('--hyperprior_init_mode', type=str, default='LP', choices=['LP', 'EQU'])

    ### Config for MTL
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet'])
    parser.add_argument('--phase', type=str, default='meta_train', choices=['pre_train', 'meta_train', 'meta_eval'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--num_batch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_base', type=float, default=0.0001)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--update_step', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--eval_weights', type=str, default=None)
    parser.add_argument('--pre_max_epoch', type=int, default=100)
    parser.add_argument('--pre_batch_size', type=int, default=128)
    parser.add_argument('--pre_lr', type=float, default=0.1)
    parser.add_argument('--pre_gamma', type=float, default=0.2)
    parser.add_argument('--pre_step_size', type=int, default=30)
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9)
    parser.add_argument('--pre_custom_weight_decay', type=float, default=0.0005)

    ### Config for SIB
    parser.add_argument('--steps_sib', type=int, default=3)
    parser.add_argument('--seed_sib', type=int, default=100)
    parser.add_argument('--ckpt_sib', default=None)
    parser.add_argument('--batchsize_sib', type=int, default=1)
    parser.add_argument('--backbone_sib', type=str, default='WRN_28_10', choices=['WRN_28_10'])
    parser.add_argument('--lr_sib', type=float, default=0.001)
    parser.add_argument('--momentum_sib', type=float, default=0.9)
    parser.add_argument('--weight_decay_sib', type=float, default=0.0005)
    parser.add_argument('--num_iter_sib', type=int, default=50000)
    parser.add_argument('--val_episode_sib', type=int, default=100)
    parser.add_argument('--resume_path_sib', type=str, default='./ckpts/miniImageNet/netFeatBest.pth')
    parser.add_argument('--base_lr_sib', type=float, default=0.001)
    parser.add_argument('--sib_lr_mode', type=str, default='EBL', choices=['HPL', 'EBL'])

    args = parser.parse_args()
    pprint(vars(args))
    print('Experiment label: ' + args.label)
    set_gpu(args.gpu)

    occupy_memory(args.gpu)
    print('Occupy GPU memory in advance.')

    if args.baseline == 'MTL':
        if args.seed==0:
            torch.backends.cudnn.benchmark = True
        else:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if args.phase=='meta_train':
            trainer = MetaTrainer(args)
            trainer.train()
        elif args.phase=='meta_eval':
            trainer = MetaTrainer(args)
            trainer.eval()
        elif args.phase=='pre_train':
            trainer = PreTrainer(args)
            trainer.train()
        else:
            raise ValueError('Please set correct phase.')

    elif args.baseline == 'SIB':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        trainer = MetaTrainerSIB(args)
        trainer.train()

    else:
        raise ValueError('Please set correct phase.')

