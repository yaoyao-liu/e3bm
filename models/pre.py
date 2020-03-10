import  torch
import torch.nn as nn
from utils.misc import euclidean_metric
import torch.nn.functional as F

class PretrainLearner(nn.Module):
    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        from mtl.networks.resnet import ResNet
        self.z_dim = 640
        self.encoder = ResNet()
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([num_class, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(num_class))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        net = self.encoder(input_x)
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(net, fc1_w, fc1_b)
        return net

    def get_add_parameters(self):
        return self.vars





