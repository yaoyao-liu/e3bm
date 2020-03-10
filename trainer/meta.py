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

class MetaTrainer(object):
    def __init__(self, args):
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr' + str(args.lr) + '_lrbase' + str(args.lr_base) + '_lrc' + str(args.lr_combination) + '_lrch' + str(args.lr_combination_hyperprior) + '_lrbs' + str(args.lr_basestep) + '_lrbsh' + str(args.lr_basestep_hyperprior) + '_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) + '_csw' + str(args.hyperprior_combination_softweight) + '_cbsw' + str(args.hyperprior_basestep_softweight) + '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + '_stepsize' + str(args.step_size) + '_' + args.label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        self.args = args

        if self.args.dataset == 'MiniImageNet':
            from dataloader.mini_imagenet import MiniImageNet as Dataset
        elif self.args.dataset == 'TieredImageNet':
            from dataloader.tiered_imagenet import TieredImageNet as Dataset
        elif self.args.dataset == 'FC100':
            from dataloader.fewshotcifar import FewshotCifar as Dataset
        else:
            raise ValueError('Non-supported Dataset.')

        self.trainset = Dataset('train', self.args)
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way, self.args.shot + self.args.train_query)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=8, pin_memory=True)

        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 3000, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)
        
        self.model = MtlLearner(self.args)

        new_para = filter(lambda p: p.requires_grad, self.model.encoder.parameters())
        self.optimizer = torch.optim.Adam([{'params': new_para}, {'params': self.model.base_learner.parameters(), 'lr': self.args.lr_base}, {'params': self.model.get_hyperprior_combination_initialization_vars(), 'lr': self.args.lr_combination}, {'params': self.model.get_hyperprior_combination_mapping_vars(), 'lr': self.args.lr_combination_hyperprior}, {'params': self.model.get_hyperprior_basestep_initialization_vars(), 'lr': self.args.lr_basestep}, {'params': self.model.get_hyperprior_stepsize_mapping_vars(), 'lr': self.args.lr_basestep_hyperprior}], lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)        
        
        self.model_dict = self.model.state_dict()
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
            print(pretrained_dict.keys())
            self.model_dict.update(pretrained_dict) 

        self.model.load_state_dict(self.model_dict)    
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
        
    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))       

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
        writer = SummaryWriter(logdir=self.args.save_path)
        
        for epoch in range(1, self.args.max_epoch + 1):
            self.lr_scheduler.step()
            self.model.train()
            tl = Averager()
            ta = Averager()

            label = torch.arange(self.args.way).repeat(self.args.train_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            label_shot = torch.arange(self.args.way).repeat(self.args.shot)
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)
                
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                logits, combination_list, basestep_list = self.model((data_shot, label_shot, data_query))
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                writer.add_scalar('combination_value/0', float(combination_list[0][0]), global_count)
                writer.add_scalar('combination_value/24', float(combination_list[24][0]), global_count)
                writer.add_scalar('combination_value/49', float(combination_list[49][0]), global_count)
                writer.add_scalar('combination_value/74', float(combination_list[74][0]), global_count)
                writer.add_scalar('combination_value/99', float(combination_list[99][0]), global_count)

                writer.add_scalar('basestep_value/0', float(basestep_list[0][0]), global_count)
                writer.add_scalar('basestep_value/24', float(basestep_list[24][0]), global_count)
                writer.add_scalar('basestep_value/49', float(basestep_list[49][0]), global_count)
                writer.add_scalar('basestep_value/74', float(basestep_list[74][0]), global_count)
                writer.add_scalar('basestep_value/99', float(basestep_list[99][0]), global_count)
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                tl.add(loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tl = tl.item()
            ta = ta.item()

            self.model.eval()

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
                
            print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            tqdm_gen1 = tqdm.tqdm(self.val_loader)
            for i, batch in enumerate(tqdm_gen1, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                logits, _, _ = self.model((data_shot, label_shot, data_query))
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
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl)
            trlog['val_acc'].append(va)

            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            self.save_model('epoch-last')

        writer.close()
