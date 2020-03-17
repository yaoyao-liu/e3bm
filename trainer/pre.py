#   Copyright (c) 2020 Yaoyao Liu. All Rights Reserved.
#   Some files of this repository are modified from https://github.com/hushell/sib_meta_learn
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
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import BaseLearner, MtlLearner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter


class PreTrainer(object):
    def __init__(self, args):
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
            str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        self.args = args

        if self.args.dataset == 'MiniImageNet':
            from dataloader.mini_imagenet import MiniImageNet as Dataset
        elif self.args.dataset == 'TieredImageNet':
            from dataloader.tiered_imagenet import TieredImageNet as Dataset
        elif self.args.dataset == 'FC100':
            from dataloader.fewshotcifar import FewshotCifar as Dataset
        else:
            raise ValueError('Please set correct dataset.')

        self.trainset = Dataset('train', self.args, train_aug=True)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)

        self.valset = Dataset('test', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 600, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)

        num_class_pretrain = self.trainset.num_class
        
        self.model = MtlLearner(self.args, mode='pre', num_cls=num_class_pretrain)

        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr}, \
            {'params': self.model.pre_fc.parameters(), 'lr': self.args.pre_lr}], \
                momentum=self.args.pre_custom_momentum, nesterov=True, weight_decay=self.args.pre_custom_weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, \
            gamma=self.args.pre_gamma)        
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
        
    def save_model(self, name):
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))
        
    def train(self):
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        timer = Timer()
        global_count = 0
        writer = SummaryWriter(comment=self.args.save_path)
        
        for epoch in range(1, self.args.pre_max_epoch + 1):
            self.lr_scheduler.step()
            self.model.train()
            self.model.mode = 'pre'
            tl = Averager()
            ta = Averager()
                
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                tl.add(loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tl = tl.item()
            ta = ta.item()

            self.model.eval()
            self.model.mode = 'preval'

            vl = Averager()
            va = Averager()

            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            label_shot = torch.arange(self.args.way).repeat(self.args.shot)
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)
                
            print('Best Epoch {}, Best Val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                logits = self.model((data_shot, label_shot, data_query))
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()
            writer.add_scalar('data/val_loss', float(vl), epoch)
            writer.add_scalar('data/val_acc', float(va), epoch)       
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, vl, va))

            if va > trlog['max_acc']:
                trlog['max_acc'] = va
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            if epoch % 20 == 0:
                self.save_model('epoch'+str(epoch))

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl)
            trlog['val_acc'].append(va)

            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch > self.args.pre_max_epoch-2:
                self.save_model('epoch-last')
                torch.save(self.optimizer.state_dict(), osp.join(self.args.save_path,'optimizer_latest.pth'))

            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
        writer.close()
