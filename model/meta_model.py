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

import  torch
import torch.nn as nn
from utils.misc import euclidean_metric
import torch.nn.functional as F

class BaseLearner(nn.Module):
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        net = F.linear(F.normalize(input_x, p=2, dim=1), F.normalize(fc1_w, p=2, dim=1))
        return net

    def parameters(self):
        return self.vars

class HyperpriorCombination(nn.Module):
    def __init__(self, args, update_step, z_dim):
        super().__init__()
        self.args = args
        self.hyperprior_initialization_vars = nn.ParameterList()
        if args.hyperprior_init_mode=='LAS':
            for idx in range(update_step-1):
                self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([0.0])))
            self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([1.0])))
        else:
            for idx in range(update_step):
                self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([1.0/update_step])))

        self.hyperprior_mapping_vars = nn.ParameterList()
        self.fc_w = nn.Parameter(torch.ones([update_step, z_dim*2]))
        torch.nn.init.kaiming_normal_(self.fc_w)
        self.hyperprior_mapping_vars.append(self.fc_w)
        self.fc_b = nn.Parameter(torch.zeros(update_step))
        self.hyperprior_mapping_vars.append(self.fc_b)
        self.hyperprior_softweight = args.hyperprior_combination_softweight

    def forward(self, input_x, grad, step_idx):
        mean_x = input_x.mean(dim=0)
        mean_grad = grad[0].mean(dim=0)
        net = torch.cat((mean_x, mean_grad), 0)
        net = F.linear(net, self.fc_w, self.fc_b)
        net = net[step_idx]
        net = self.hyperprior_initialization_vars[step_idx] + self.hyperprior_softweight*net
        return net

    def get_hyperprior_initialization_vars(self):
        return self.hyperprior_initialization_vars

    def get_hyperprior_mapping_vars(self):
        return self.hyperprior_mapping_vars

class HyperpriorBasestep(nn.Module):
    def __init__(self, args, update_step, update_lr, z_dim):
        super().__init__()
        self.args = args
        self.hyperprior_initialization_vars = nn.ParameterList()
        for idx in range(update_step):
            self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([update_lr])))

        self.hyperprior_mapping_vars = nn.ParameterList()
        self.fc_w = nn.Parameter(torch.ones([update_step, z_dim*2]))
        torch.nn.init.kaiming_normal_(self.fc_w)
        self.hyperprior_mapping_vars.append(self.fc_w)
        self.fc_b = nn.Parameter(torch.zeros(update_step))
        self.hyperprior_mapping_vars.append(self.fc_b)
        self.hyperprior_softweight = args.hyperprior_basestep_softweight


    def forward(self, input_x, grad, step_idx):
        mean_x = input_x.mean(dim=0)
        mean_grad = grad[0].mean(dim=0)
        net = torch.cat((mean_x, mean_grad), 0)
        net = F.linear(net, self.fc_w, self.fc_b)
        net = net[step_idx]
        net = self.hyperprior_initialization_vars[step_idx] + self.hyperprior_softweight*net
        return net

    def get_hyperprior_initialization_vars(self):
        return self.hyperprior_initialization_vars

    def get_hyperprior_mapping_vars(self):
        return self.hyperprior_mapping_vars

class MetaModel(nn.Module):
    def __init__(self, args, dropout=0.2, mode='meta'):
        super().__init__()
        self.args = args
        self.mode = mode

        self.init_backbone()
        self.base_learner = BaseLearner(args, self.z_dim)
        self.update_lr = self.args.base_lr
        self.update_step = self.args.base_epoch

        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            self.label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            self.label_shot = label_shot.type(torch.LongTensor)

        if self.mode == 'meta':
            self.hyperprior_combination_model = HyperpriorCombination(args, self.update_step, self.z_dim)
            self.hyperprior_basestep_model = HyperpriorBasestep(args, self.update_step, self.update_lr, self.z_dim)

    def init_backbone(self):
        if self.args.backbone == 'resnet12':
            if self.mode == 'pre':
                from model.resnet12 import ResNet
            else:
                if self.args.meta_update=='mtl':
                    from model.resnet12_mtl import ResNet
                else:
                    from model.resnet12 import ResNet
            self.encoder = ResNet()
            self.z_dim = 640
        elif self.args.backbone == 'wrn':
            if self.mode == 'pre':
                from Models.backbone.wrn import ResNet
            else:
                if self.args.meta_update=='mtl':
                    from model.wrn_mtl import ResNet
                else:
                    from model.wrn import ResNet
            self.encoder = ResNet()
            self.z_dim = 640
        else:
            raise ValueError('Please set the correct backbone')

        if self.mode == 'pre':
            self.fc = nn.Sequential(nn.Linear(self.z_dim, self.args.num_class))

    def forward(self, inputs):
        if self.mode=='pre':
            return self.pretrain_forward(inputs)
        elif self.mode=='meta':
            data_shot, data_query = inputs
            return self.meta_forward(data_shot, data_query)
        else:
            raise ValueError('Please set the correct mode')

    def pretrain_forward(self, input):
        return self.fc(self.encoder(input))

    def normalize_feature(self, x):
        x = x-x.mean(-1).unsqueeze(-1)
        return x

    def fusion(self, embedding):
        embedding = embedding.view(self.args.shot, self.args.way, -1)
        embedding = embedding.mean(0)
        return embedding

    def get_hyperprior_combination_initialization_vars(self):
        return self.hyperprior_combination_model.get_hyperprior_initialization_vars()

    def get_hyperprior_basestep_initialization_vars(self):
        return self.hyperprior_basestep_model.get_hyperprior_initialization_vars()

    def get_hyperprior_combination_mapping_vars(self):
        return self.hyperprior_combination_model.get_hyperprior_mapping_vars()

    def get_hyperprior_stepsize_mapping_vars(self):
        return self.hyperprior_basestep_model.get_hyperprior_mapping_vars()

    def meta_forward(self, data_shot, data_query):
        data_query=data_query.squeeze(0)
        data_shot = data_shot.squeeze(0)

        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        embedding_shot=self.normalize_feature(embedding_shot)
        embedding_query=self.normalize_feature(embedding_query)

        with torch.no_grad():
            if self.args.shot==1:
                proto = embedding_shot
            else:
                proto=self.fusion(embedding_shot)
            self.base_learner.fc1_w.data = proto

        fast_weights = self.base_learner.vars

        combination_value_list = []
        basestep_value_list = []
        batch_shot = embedding_shot
        batch_label = self.label_shot
        logits_q = self.base_learner(embedding_query, fast_weights)
        total_logits = 0.0 * logits_q

        for k in range(0, self.update_step):

            batch_shot = embedding_shot
            batch_label = self.label_shot
            logits = self.base_learner(batch_shot, fast_weights) * self.args.temperature
            loss = F.cross_entropy(logits, batch_label)
            grad = torch.autograd.grad(loss, fast_weights)
            generated_combination_weights = self.hyperprior_combination_model(embedding_shot, grad, k)
            generated_basestep_weights = self.hyperprior_basestep_model(embedding_shot, grad, k)
            fast_weights = list(map(lambda p: p[1] - generated_basestep_weights * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
            logits_q = logits_q * self.args.temperature
            total_logits += generated_combination_weights * logits_q
            combination_value_list.append(generated_combination_weights)
            basestep_value_list.append(generated_basestep_weights)  

        return total_logits