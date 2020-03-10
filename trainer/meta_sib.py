import argparse
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from utils.misc import create_dirs, get_logger, set_random_seed
from dataloader.sib import BatchSampler, ValLoader, EpisodeSampler
from dataloader.dataset_setting_sib import dataset_setting
from networks.sib import get_featnet
from trainer.runner_sib import RunnerSIB
from models.sib import ClassifierSIB

class MetaTrainerSIB(object):
    def __init__(self, args):
        args.cache_dir = os.path.join("cache", '{}_{}shot_K{}_seed{}'.format(args.label, args.shot, args.steps_sib, args.seed_sib))
        args.log_dir = os.path.join(args.cache_dir, 'logs')
        args.out_dir = os.path.join(args.cache_dir, 'outputs')
        create_dirs([args.cache_dir, args.log_dir, args.out_dir])

        logger = get_logger(args.log_dir, args.label)
        set_random_seed(args.seed_sib)
        logger.info('Start experiment with random seed: {:d}'.format(args.seed_sib))
        logger.info(args)
        self.logger = logger

        trainTransform, valTransform, self.inputW, self.inputH, trainDir, valDir, testDir, episodeJson, nbCls = dataset_setting(args.dataset, args.way)

        self.trainLoader = BatchSampler(imgDir = trainDir, nClsEpisode=args.way, nSupport=args.shot, nQuery=args.train_query, transform=trainTransform, useGPU=True, inputW=self.inputW, inputH=self.inputH, batchSize=args.batchsize_sib)
        self.testLoader = EpisodeSampler(imgDir=testDir, nClsEpisode=args.way, nSupport=args.shot, nQuery=args.train_query, transform=valTransform, useGPU=True, inputW=self.inputW, inputH=self.inputH)
        self.valLoader = EpisodeSampler(imgDir=testDir, nClsEpisode=args.way, nSupport=args.shot, nQuery=args.train_query, transform=valTransform, useGPU=True, inputW=self.inputW, inputH=self.inputH)
        self.args = args

    def train(self):
        device = torch.device('cuda')
        args = self.args
        netFeat, args.nFeat = get_featnet(args.backbone_sib, self.inputW, self.inputH)

        netFeat = netFeat.to(device)
        netSIB = ClassifierSIB(args.way, args.nFeat, args.steps_sib, args)
        netSIB = netSIB.to(device)

        scale_vars, wnLayerFavg_parameters, dni_parameters, hyperprior_combination_initialization_vars, hyperprior_combination_mapping_vars, hyperprior_basestep_initialization_vars, hyperprior_basestep_mapping_vars = netSIB.get_sib_parameters()

        optimizer = torch.optim.SGD([{'params': scale_vars}, {'params': wnLayerFavg_parameters, 'lr': args.lr_sib}, {'params': dni_parameters, 'lr': args.lr_sib}, {'params': hyperprior_combination_initialization_vars, 'lr': args.lr_combination}, {'params': hyperprior_combination_mapping_vars, 'lr': args.lr_combination_hyperprior}, {'params': hyperprior_basestep_initialization_vars, 'lr': args.lr_basestep}, {'params': hyperprior_basestep_mapping_vars, 'lr': args.lr_basestep_hyperprior}], args.lr, momentum=args.momentum_sib, weight_decay=args.weight_decay_sib, nesterov=True)

        criterion = torch.nn.CrossEntropyLoss()

        runner_sib = RunnerSIB(args, self.logger, netFeat, netSIB, optimizer, criterion)

        bestAcc, lastAcc, history = runner_sib.train(self.trainLoader, self.valLoader)

        msg = 'mv {} {}'.format(os.path.join(args.out_dir, 'netSIBBest.pth'),
                                os.path.join(args.out_dir, 'netSIBBest{:.3f}.pth'.format(bestAcc)))
        logger.info(msg)
        os.system(msg)

        msg = 'mv {} {}'.format(os.path.join(args.out_dir, 'netSIBLast.pth'),
                                os.path.join(args.out_dir, 'netSIBLast{:.3f}.pth'.format(lastAcc)))
        logger.info(msg)
        os.system(msg)

        with open(os.path.join(args.out_dir, 'history.json'), 'w') as f :
            json.dump(history, f)

        msg = 'mv {} {}'.format(args.out_dir, '{}_{:.3f}'.format(args.out_dir, bestAcc))
        logger.info(msg)
        os.system(msg)


