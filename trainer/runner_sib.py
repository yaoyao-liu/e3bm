# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os
import itertools
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.outils import progress_bar, AverageMeter, accuracy, getCi
from utils.misc import to_device

class RunnerSIB:
    def __init__(self, args, logger, netFeat, netSIB, optimizer, criterion):
        self.netFeat = netFeat
        self.netSIB = netSIB
        self.optimizer = optimizer
        self.criterion = criterion

        self.nbIter = args.num_iter_sib
        self.nStep = args.steps_sib
        self.outDir = args.out_dir
        self.nFeat = args.nFeat
        self.batchSize = args.batchsize_sib
        self.nEpisode = args.val_episode_sib
        self.momentum = args.momentum_sib
        self.weightDecay = args.weight_decay_sib

        self.logger = logger
        self.device = torch.device('cuda')

        param = torch.load(args.resume_path_sib)
        self.netFeat.load_state_dict(param)
        msg = '\nLoading netFeat from {}'.format(args.resume_path_sib)
        self.logger.info(msg)


    def compute_grad_loss(self, clsScore, QueryLabel):
        # register hooks
        def require_nonleaf_grad(v):
            def hook(g):
                v.grad_nonleaf = g
            h = v.register_hook(hook)
            return h
        handle = require_nonleaf_grad(clsScore)

        loss = self.criterion(clsScore, QueryLabel)
        loss.backward(retain_graph=True) # need to backward again

        # remove hook
        handle.remove()

        gradLogit = self.netSIB.dni(clsScore) # B * n x nKnovel
        gradLoss = F.mse_loss(gradLogit, clsScore.grad_nonleaf.detach())

        return loss, gradLoss


    def validate(self, valLoader, lr=None, mode='val'):

        nEpisode = self.nEpisode
        self.logger.info('\n\nTest mode: randomly sample {:d} episodes...'.format(nEpisode))

        episodeAccLog = []
        top1 = AverageMeter()

        self.netFeat.eval()

        if lr is None:
            lr = self.optimizer.param_groups[0]['lr']

        for batchIdx in range(nEpisode):
            data = valLoader.getEpisode()
            data = to_device(data, self.device)

            SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                    data['SupportTensor'].squeeze(0), data['SupportLabel'].squeeze(0), \
                    data['QueryTensor'].squeeze(0), data['QueryLabel'].squeeze(0)

            with torch.no_grad():
                SupportFeat, QueryFeat = self.netFeat(SupportTensor), self.netFeat(QueryTensor)
                SupportFeat, QueryFeat, SupportLabel = \
                        SupportFeat.unsqueeze(0), QueryFeat.unsqueeze(0), SupportLabel.unsqueeze(0)

            clsScore = self.netSIB(SupportFeat, SupportLabel, QueryFeat, lr)
            clsScore = clsScore.view(QueryFeat.shape[0] * QueryFeat.shape[1], -1)
            QueryLabel = QueryLabel.view(-1)
            acc1 = accuracy(clsScore, QueryLabel, topk=(1,))
            top1.update(acc1[0].item(), clsScore.shape[0])

            msg = 'Top1: {:.3f}%'.format(top1.avg)
            progress_bar(batchIdx, nEpisode, msg)
            episodeAccLog.append(acc1[0].item())

        mean, ci95 = getCi(episodeAccLog)
        self.logger.info('Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95))
        return mean, ci95


    def train(self, trainLoader, valLoader, lr=None, coeffGrad=0.0) :
        bestAcc, ci = self.validate(valLoader, lr)
        self.logger.info('Acc improved over validation set from 0% ---> {:.3f} +- {:.3f}%'.format(bestAcc,ci))

        self.netSIB.train()
        self.netFeat.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        history = {'trainLoss' : [], 'trainAcc' : [], 'valAcc' : []}

        for episode in range(self.nbIter):
            data = trainLoader.getBatch()
            data = to_device(data, self.device)

            with torch.no_grad() :
                SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                        data['SupportTensor'], data['SupportLabel'], data['QueryTensor'], data['QueryLabel']
                nC, nH, nW = SupportTensor.shape[2:]

                SupportFeat = self.netFeat(SupportTensor.reshape(-1, nC, nH, nW))
                SupportFeat = SupportFeat.view(self.batchSize, -1, self.nFeat)

                QueryFeat = self.netFeat(QueryTensor.reshape(-1, nC, nH, nW))
                QueryFeat = QueryFeat.view(self.batchSize, -1, self.nFeat)

            if lr is None:
                lr = self.optimizer.param_groups[0]['lr']

            self.optimizer.zero_grad()

            clsScore = self.netSIB(SupportFeat, SupportLabel, QueryFeat, lr)
            clsScore = clsScore.view(QueryFeat.shape[0] * QueryFeat.shape[1], -1)
            QueryLabel = QueryLabel.view(-1)

            if coeffGrad > 0:
                loss, gradLoss = self.compute_grad_loss(clsScore, QueryLabel)
                loss = loss + gradLoss * coeffGrad
            else:
                loss = self.criterion(clsScore, QueryLabel)

            loss.backward()
            self.optimizer.step()

            acc1 = accuracy(clsScore, QueryLabel, topk=(1, ))
            top1.update(acc1[0].item(), clsScore.shape[0])
            losses.update(loss.item(), QueryFeat.shape[1])
            msg = 'Loss: {:.3f} | Top1: {:.3f}% '.format(losses.avg, top1.avg)
            if coeffGrad > 0:
                msg = msg + '| gradLoss: {:.3f}%'.format(gradLoss.item())
            progress_bar(episode, self.nbIter, msg)

            if episode % 1000 == 999 :
                acc, _ = self.validate(valLoader, lr)

                if acc > bestAcc :
                    msg = 'Acc improved over validation set from {:.3f}% ---> {:.3f}%'.format(bestAcc , acc)
                    self.logger.info(msg)

                    bestAcc = acc
                    self.logger.info('Saving Best')
                    torch.save({
                                'lr': lr,
                                'netFeat': self.netFeat.state_dict(),
                                'SIB': self.netSIB.state_dict(),
                                'nbStep': self.nStep,
                                }, os.path.join(self.outDir, 'netSIBBest.pth'))

                self.logger.info('Saving Last')
                torch.save({
                            'lr': lr,
                            'netFeat': self.netFeat.state_dict(),
                            'SIB': self.netSIB.state_dict(),
                            'nbStep': self.nStep,
                            }, os.path.join(self.outDir, 'netSIBLast.pth'))

                msg = 'Iter {:d}, Train Loss {:.3f}, Train Acc {:.3f}%, Val Acc {:.3f}%'.format(
                        episode, losses.avg, top1.avg, acc)
                self.logger.info(msg)
                history['trainLoss'].append(losses.avg)
                history['trainAcc'].append(top1.avg)
                history['valAcc'].append(acc)

                losses = AverageMeter()
                top1 = AverageMeter()

        return bestAcc, acc, history
