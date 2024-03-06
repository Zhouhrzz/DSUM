import itertools
from itertools import chain
import torch
import torch.nn as nn
import utils
from utils import LRN
import math
import torch.nn.functional as F
from torch.nn.modules.distance import PairwiseDistance
from scipy.spatial.distance import cdist
import numpy as np
from grl import WarmStartGradientReverseLayer
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class AlexNet(nn.Module):

    def __init__(self, cudable, n_class, batch_size, centroid=None):
        super(AlexNet, self).__init__()
        self.cudable = cudable
        self.n_class = n_class
        self.s_centroid = torch.zeros(self.n_class, 256)
        self.t_centroid = torch.zeros(self.n_class, 256)
        self.decay = 0.3
        self.batch_size = batch_size
        self.CEloss, self.BCEloss = \
            nn.CrossEntropyLoss().cuda(), nn.BCEWithLogitsLoss(reduction='mean').cuda()
        self.MSEloss = nn.MSELoss().cuda()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=5000, auto_step=True)

        if self.cudable:
            self.s_centroid = self.s_centroid.cuda()
            self.t_centroid = self.t_centroid.cuda()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            # nn.LocalResponseNorm(3, alpha=1e-5),
            LRN(local_size=5, alpha=1e-4, beta=0.75),
            nn.Conv2d(96, 256, 5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            LRN(local_size=5, alpha=1e-4, beta=0.75),
            # nn.LocalResponseNorm(3, alpha=1e-5),
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096, 256),
        )

        self.fc9 = nn.Sequential(
            nn.Linear(256, self.n_class)
        )

        self.softmax = nn.Softmax(dim=1)
        self.D = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1)
        )
        self.init()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=5000, auto_step=True)


    def init(self):
        self.init_linear(self.fc8[0])
        self.init_linear(self.fc9[0], std=0.005)
        self.init_linear(self.D[0], D=True)
        self.init_linear(self.D[3], D=True)
        self.init_linear(self.D[6], D=True, std=0.3)


    def init_linear(self, m, std=0.01, D=False):
        # nn.init.normal_(m.weight.data, 0, std)
        # nn.init.xavier_normal_(m.weight)
        utils.truncated_normal_(m.weight.data, 0, std)
        if m.bias != None:
            if D:
                m.bias.data.fill_(0)
            else:
                m.bias.data.fill_(0.1)

    def forward(self, x, bs=None, smooth=9., Instance=False):
        view4_mu = []
        view4_var = []
        if not bs:
            bs = self.batch_size
        x = self.features(x)
        if Instance==True:
            global_x = x.view(x.size()[0], -1)
        else:
            global_x = x.view(bs, x.size(0) // bs, x.size()[1], x.size()[2], x.size()[3])
            global_x = torch.max(global_x, 1)[0].view(bs, -1)
        global_x = self.classifier(global_x)
        feature = self.fc8(global_x)
        if Instance==True:
            view4_flattened = feature.view(bs, int(feature.size()[0] / bs), feature.size()[-1])
            view4_mu = torch.mean(view4_flattened, dim=1)
            view4_var = torch.var(view4_flattened, dim=1, keepdim=False, unbiased=False)
            view4_std = torch.sqrt(view4_var + 1e-9)
            epsilon = torch.zeros_like(view4_std, dtype=torch.float)                       
            for index in range(0, bs): 
                epsilon[index] = torch.randn_like(epsilon[index])
            feature = view4_mu + epsilon * view4_std
            score = self.fc9(feature)
            pred = self.softmax(score)
        else:               
            score = self.fc9(feature)
            pred = self.softmax(score)

        return feature, score, pred, view4_mu, view4_var

    def centroid(self, st_feature, label, batchsize):
        ones = torch.ones_like(label, dtype=torch.float)
        zeros = torch.zeros(40).cuda()
        n_classes = zeros.scatter_add(0, label, ones)
        ones = torch.ones_like(n_classes)
        n_classes = torch.max(n_classes, ones)

        zeros = torch.zeros(40, 256).cuda()
        sum_feature = zeros.scatter_add(0, torch.transpose(label.repeat(256, 1), 1, 0), st_feature)
        centroid = torch.div(sum_feature, n_classes.view(40, 1))
        std = torch.zeros_like(st_feature, dtype=torch.float)
        centriid_sum = torch.zeros_like(centroid, dtype=torch.float)
        epsilon = torch.zeros_like(centroid, dtype=torch.float)
        for index in range(0 , batchsize):
            lable_index = label[index]
            centroid_index = centroid[lable_index]
            epsilon[lable_index] = torch.randn_like(centroid_index)
            up = pow((st_feature[index] - centroid_index),2)
            centriid_sum[lable_index] += up
            index += 1
        var = (torch.div(centriid_sum, n_classes.view(40, 1)))
        std = torch.sqrt(var + 1e-9)
        st_centroid = centroid + epsilon * std
        return centroid, var , st_centroid

    def forward_D(self, feature):
        logit = self.D(feature)
        return logit

    def closs(self, y_pred, y):
        C_loss = self.CEloss(y_pred, y)
        return C_loss

    def adloss(self, s_logits, t_logits):
        # sigmoid binary cross entropy with reduce mean
        D_real_loss = self.BCEloss(t_logits, torch.ones_like(t_logits))
        D_fake_loss = self.BCEloss(s_logits, torch.zeros_like(s_logits))
        D_loss = (D_real_loss + D_fake_loss) * 0.1
        G_loss = -D_loss
        return G_loss, D_loss

    def ad(self, fs, ft):
        fs_grl = self.grl(fs)
        ft_grl = self.grl(ft)
        feature = torch.cat((fs_grl, ft_grl), dim=0)
        ad_out = self.D(feature)
        ad_out = nn.Sigmoid()(ad_out)
        batch_size = fs.size(0)
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
        return nn.BCELoss()(ad_out, dc_target)

    # To some extent, can be replaced by weight_decay param in optimizer.
    def regloss(self):
        Dregloss = [torch.sum(layer.weight ** 2) / 2 for layer in self.D if type(layer) == nn.Linear]
        layers = chain(self.features, self.classifier, self.fc8, self.fc9)
        Gregloss = [torch.sum(layer.weight ** 2) / 2 for layer in layers if
                    type(layer) == nn.Conv2d or type(layer) == nn.Linear]
        mean = lambda x: 0.0005 * torch.mean(torch.stack(x))
        return mean(Dregloss), mean(Gregloss)

    def klloss(mean, var):
        kl_loss = ((var + mean**2 - torch.log(var) - 1) * 0.5).sum(dim=1).mean()
        return kl_loss

    def get_optimizer(self, init_lr, lr_mult):
        w_finetune, b_finetune, w_train, b_train, w_D, b_D = [], [], [], [], [], []

        finetune_layers = itertools.chain(self.features, self.classifier)
        train_layers = itertools.chain(self.fc8, self.fc9, self.D)
        for layer in finetune_layers:
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                w_finetune.append(layer.weight)
                b_finetune.append(layer.bias)
        for layer in train_layers:
            if type(layer) == nn.Linear:
                w_train.append(layer.weight)
                if layer.bias != None:
                    b_train.append(layer.bias)
        for layer in self.D:
            if type(layer) == nn.Linear:
                w_D.append(layer.weight)
                b_D.append(layer.bias)

        opt = torch.optim.SGD([{'params': w_finetune, 'lr': init_lr * lr_mult[0]},
                               {'params': b_finetune, 'lr': init_lr * lr_mult[1]},
                               {'params': w_train, 'lr': init_lr * lr_mult[2]},
                               {'params': b_train, 'lr': init_lr * lr_mult[3]}],
                              lr=init_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        return opt
