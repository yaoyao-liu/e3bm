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
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

class HyperpriorCombination(nn.Module):
    def __init__(self, args, update_step, z_dim):
        super().__init__()
        self.args = args
        self.hyperprior_initialization_vars = nn.ParameterList()
        if args.hyperprior_init_mode=='LP':
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

class HyperpriorCombinationLSTM(nn.Module):
    def __init__(self, args, update_step, z_dim):
        super().__init__()
        self.args = args
        self.hyperprior_initialization_vars = nn.ParameterList()
        if args.hyperprior_init_mode=='LP':
            for idx in range(update_step-1):
                self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([0.0])))
            self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([1.0])))
        else:
            for idx in range(update_step):
                self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([1.0/update_step])))
        self.hidden_dim = 1
        self.lstm = nn.LSTM(z_dim*2, self.hidden_dim)
        self.hidden = self.init_hidden()
        self.hyperprior_softweight = args.hyperprior_combination_softweight

    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, input_x, grad, step_idx):
        mean_x = input_x.mean(dim=0)
        mean_grad = grad[0].mean(dim=0)
        net = torch.cat((mean_x, mean_grad), 0)
        net, self.hidden = self.lstm(net.view(1, 1, -1), self.hidden)
        net = self.hyperprior_initialization_vars[step_idx] + self.hyperprior_softweight*net
        return net

    def get_hyperprior_initialization_vars(self):
        return self.hyperprior_initialization_vars

    def get_hyperprior_mapping_vars(self):
        return self.lstm.parameters()

class HyperpriorBasestepLSTM(nn.Module):
    def __init__(self, args, update_step, update_lr, z_dim):
        super().__init__()
        self.args = args
        self.hyperprior_initialization_vars = nn.ParameterList()
        for idx in range(update_step):
            self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([update_lr])))
        self.hidden_dim = 1
        self.lstm = nn.LSTM(z_dim*2, self.hidden_dim)
        self.hidden = self.init_hidden()
        self.hyperprior_softweight = args.hyperprior_combination_softweight

    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, input_x, grad, step_idx):
        mean_x = input_x.mean(dim=0)
        mean_grad = grad[0].mean(dim=0)
        net = torch.cat((mean_x, mean_grad), 0)
        net, self.hidden = self.lstm(net.view(1, 1, -1), self.hidden)
        net = self.hyperprior_initialization_vars[step_idx] + self.hyperprior_softweight*net
        return net

    def get_hyperprior_initialization_vars(self):
        return self.hyperprior_initialization_vars

    def get_hyperprior_mapping_vars(self):
        return self.lstm.parameters()

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

class MtlLearner(nn.Module):
    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        z_dim = 640
        self.base_learner = BaseLearner(args, z_dim)

        if self.mode == 'meta':
            from networks.resnet_mtl import ResNetMtl
            self.encoder = ResNetMtl()
            if args.hyperprior_arch == 'LSTM':
                self.hyperprior_combination_model = HyperpriorCombinationLSTM(args, self.update_step, z_dim)
                self.hyperprior_basestep_model = HyperpriorBasestepLSTM(args, self.update_step, self.update_lr, z_dim)
            else:
                self.hyperprior_combination_model = HyperpriorCombination(args, self.update_step, z_dim)
                self.hyperprior_basestep_model = HyperpriorBasestep(args, self.update_step, self.update_lr, z_dim)
        else:
            from networks.resnet import ResNet
            self.encoder = ResNet()
            self.pre_fc = nn.Sequential(nn.Linear(640, 512), nn.ReLU(), nn.Linear(512, num_cls))

    def forward(self, inp):
        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, input):
        return self.pre_fc(self.encoder(input))

    def meta_forward(self, data_shot, label_shot, data_query):
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        combination_value_list = []
        basestep_value_list = []
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        generated_combination_weights = self.hyperprior_combination_model(embedding_shot, grad, 0)
        generated_basestep_weights = self.hyperprior_basestep_model(embedding_shot, grad, 0)
        fast_weights = list(map(lambda p: p[1] - generated_basestep_weights * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)
        total_logits = generated_combination_weights * logits_q
        combination_value_list.append(generated_combination_weights)
        basestep_value_list.append(generated_basestep_weights)

        for k in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            generated_combination_weights = self.hyperprior_combination_model(embedding_shot, grad, k)
            generated_basestep_weights = self.hyperprior_basestep_model(embedding_shot, grad, k)
            fast_weights = list(map(lambda p: p[1] - generated_basestep_weights * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
            total_logits += generated_combination_weights * logits_q
            combination_value_list.append(generated_combination_weights)
            basestep_value_list.append(generated_basestep_weights)  
        return total_logits, combination_value_list, basestep_value_list

    def preval_forward(self, data_shot, label_shot, data_query):
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for k in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)         
        return logits_q

    def get_hyperprior_combination_initialization_vars(self):
        return self.hyperprior_combination_model.get_hyperprior_initialization_vars()

    def get_hyperprior_basestep_initialization_vars(self):
        return self.hyperprior_basestep_model.get_hyperprior_initialization_vars()

    def get_hyperprior_combination_mapping_vars(self):
        return self.hyperprior_combination_model.get_hyperprior_mapping_vars()

    def get_hyperprior_stepsize_mapping_vars(self):
        return self.hyperprior_basestep_model.get_hyperprior_mapping_vars()
