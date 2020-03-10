import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sib import label_to_1hot, dni_linear, LinearDiag, FeatExemplarAvgBlock

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
        mean_x = input_x.mean(dim=0).mean(dim=0)
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
        mean_x = input_x.mean(dim=0).mean(dim=0)
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
        mean_x = input_x.mean(dim=0).mean(dim=0)
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
        mean_x = input_x.mean(dim=0).mean(dim=0)
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

class ClassifierSIB(nn.Module):
    def __init__(self, nKnovel, nFeat, q_steps, args):
        super(ClassifierSIB, self).__init__()

        self.nKnovel = nKnovel
        self.nFeat = nFeat
        self.q_steps = q_steps

        self.scale_vars = nn.ParameterList()
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        self.scale_vars.append(self.bias)
        self.scale_vars.append(self.scale_cls)
        z_dim = 640

        if args.hyperprior_arch == 'LSTM':
            self.hyperprior_combination_model = HyperpriorCombinationLSTM(args, q_steps, z_dim)
            self.hyperprior_basestep_model = HyperpriorBasestepLSTM(args, q_steps,  args.base_lr_sib, z_dim)
        else:
            self.hyperprior_combination_model = HyperpriorCombination(args, q_steps, z_dim)
            self.hyperprior_basestep_model = HyperpriorBasestep(args, q_steps,  args.base_lr_sib, z_dim)

        # init_net lambda(d_t^l)
        self.favgblock = FeatExemplarAvgBlock(self.nFeat)
        self.wnLayerFavg = LinearDiag(self.nFeat)

        # grad_net (aka decoupled network interface) phi(x_t)
        self.dni = dni_linear(self.nKnovel, dni_hidden_size=self.nKnovel*8)

    def get_sib_parameters(self):
        return self.scale_vars, self.wnLayerFavg.parameters(), self.dni.parameters(), self.hyperprior_combination_model.get_hyperprior_initialization_vars(), self.hyperprior_combination_model.get_hyperprior_mapping_vars(), self.hyperprior_basestep_model.get_hyperprior_initialization_vars(), self.hyperprior_basestep_model.get_hyperprior_mapping_vars()

    def apply_classification_weights(self, features, cls_weights):
        features = F.normalize(features, p=2, dim=features.dim()-1, eps=1e-12)
        cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)

        cls_scores = self.scale_cls * torch.baddbmm(1.0, self.bias.view(1, 1, 1), 1.0,
                                                    features, cls_weights.transpose(1,2))
        return cls_scores

    def init_theta(self, features_supp, labels_supp_1hot):
        theta = self.favgblock(features_supp, labels_supp_1hot) # B x nKnovel x nFeat
        batch_size, nKnovel, num_channels = theta.size()
        theta = theta.view(batch_size * nKnovel, num_channels)
        theta = self.wnLayerFavg(theta) # weight each feature differently
        theta = theta.view(-1, nKnovel, num_channels)
        return theta

    def run_classification(self, features_supp, labels_supp_1hot, features_query, lr):

        features_supp = F.normalize(features_supp, p=2, dim=features_supp.dim()-1, eps=1e-12)

        theta = self.init_theta(features_supp, labels_supp_1hot)

        batch_size, num_examples = features_query.size()[:2]
        new_batch_dim = batch_size * num_examples

        total_scores = 0

        for t in range(self.q_steps):
            cls_scores = self.apply_classification_weights(features_query, theta)
            cls_scores = cls_scores.view(new_batch_dim, -1)
            grad_logit = self.dni(cls_scores)
            grad = torch.autograd.grad([cls_scores], [theta],
                                       grad_outputs=[grad_logit],
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True)[0] 

            generated_combination_weights = self.hyperprior_combination_model(features_supp, grad, t)
            generated_basestep_weights = self.hyperprior_basestep_model(features_supp, grad, t)
            theta = theta - generated_basestep_weights * grad
            total_scores += generated_combination_weights * self.apply_classification_weights(features_query, theta)

        return total_scores


    def forward(self, features_supp, labels_supp, features_query, lr):
        labels_supp_1hot = label_to_1hot(labels_supp, self.nKnovel)
        cls_scores = self.run_classification(features_supp, labels_supp_1hot, features_query, lr)

        return cls_scores

