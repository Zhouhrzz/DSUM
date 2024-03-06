import torch
import numpy as np
import scipy.io as scio
import pdb
import time
import torch.nn as nn
import argparse
import torch.nn.functional as F

class Performance:
    # The input must be tensor, not be numpy !!!
    # s_feature, t_feature: x*256
    # s_label, t_lable: [1, label_num] or [label_num, 1] or [label_num] is all ok

    # The compute time with cpu will be faster than with cuda.
    # The speed with cpu is about 2 times than with cuda.
    def __init__(self, s_feature, s_label, t_feature, t_label, cuda=False):
        self.s_feature, self.t_feature = self.norm_feature(s_feature), self.norm_feature(t_feature)
        # self.s_feature, self.t_feature = F.normalize(s_feature), F.normalize(t_feature)
        self.s_label, self.t_label = s_label.view(-1,1), t_label.view(-1,1)
        self.cuda = cuda
        self.similarity_f = nn.CosineSimilarity(dim=2)
        if self.cuda:
            self.s_feature, self.t_feature, self.s_label, self.t_label = \
                self.s_feature.cuda(), self.t_feature.cuda(), self.s_label.cuda(), self.t_label.cuda()

        self.final_adj = self.nearest_index()
        self.rresult = self.retrieval_matrix()
        self.s_class_num, self.t_class_num = self.class_num()
        self.s_num = self.rresult.shape[0]


        if self.cuda:
            self.rresult, self.s_class_num, \
            self.t_class_num, self.similarity_f = self.rresult.cuda(), \
                                                  self.s_class_num.cuda(), \
                                                  self.t_class_num.cuda(), self.similarity_f.cuda()

    def test_performance(self):
        NN = self.top_k(k=1)
        FT, ST, DCG = self.ft_st_dcg()
        F_measure = self.measure(k=20)
        ANMRR = self.anmrr()
        return NN, FT, ST, DCG, F_measure, ANMRR

    def t_SNE_retrieval(self):
        t_SNE_retrieval_matrix = self.final_adj
        return t_SNE_retrieval_matrix   

    def top_k(self, k):
        return (torch.max(self.rresult[:, 0:k].view(-1, k), dim=1)[0].sum() / self.rresult.shape[0]).cpu().item()

    def ft_st_dcg(self):
        FT, ST, DCG = [], [], []
        coef = torch.tensor([1.0 / np.log2(i) for i in range(2, int(self.t_class_num.max()) + 2)])
        if self.cuda:
            coef = coef.cuda()

        for i in range(self.s_num):
            label_num = int(self.t_class_num[self.s_label[i]])
            FT.append(self.rresult[i, 0:label_num].sum()/label_num)
            ST.append(self.rresult[i, 0:(2*label_num-1)].sum()/label_num)
            DCG.append((self.rresult[i, 0:label_num] * coef[0:label_num]).sum() / coef[0:label_num].sum())

        FT, ST, DCG = torch.tensor(FT), torch.tensor(ST), torch.tensor(DCG)
        return (FT.sum()/len(FT)).item(), (ST.sum()/len(ST)).item(), (DCG.sum()/len(DCG)).item()

    def measure(self, k=20):
        temp = self.rresult.sum()
        n = self.rresult[:, 0:k]
        s = n.sum()
        p = s/(self.s_num*k)
        rr = s/temp
        return (2/(1/p + 1/rr)).cpu().item()

    def AUC(self):
        ETH_p, ETH_rr = [], []
        rr_ = self.rresult.sum()
        for i in range(self.rresult.shape[1]):
            s = (self.rresult[:, 0:(i+1)]).sum()
            ETH_p.append(s/(self.s_num*(i+1)))
            ETH_rr.append(s/rr_)
        ETH_p, ETH_rr = torch.tensor(ETH_p), torch.tensor(ETH_rr)
        pr_cure = [ETH_rr.view(-1,1), ETH_p.view(-1,1)]
        if self.cuda:
            ETH_p, ETH_rr = ETH_p.cuda(), ETH_rr.cuda()
        AUC = torch.trapz(ETH_p, ETH_rr)
        # torch >= 1.2.0, else np.trapz(np.asarray(ETH_p), np.asarray(ETH_rr))
        return pr_cure, AUC.cpu()

    def anmrr(self):
        T_max = self.t_class_num.max()
        NMRR = []
        for i in range(self.s_num):
            label_num = int(self.t_class_num[self.s_label[i]])
            S_k = min(4*label_num, 2*T_max)
            r = torch.tensor([k+1 if self.rresult[i, k]==1 else S_k+1 for k in range(label_num)])
            NMRR.append(((r.sum()/float(label_num)) - label_num/2.0 - 0.5) / (S_k - label_num/2.0 + 0.5))
        return torch.tensor(NMRR).sum()/self.s_num

    def norm_feature(self, feature):
        feature[feature < 0] = 0
        square_feature = feature.pow(2).sum(1, keepdim=True).pow(0.5)
        norm_feature = torch.zeros_like(feature)
        for i in range(feature.shape[0]):
            if square_feature[i] != 0:
                norm_feature[i] = torch.div(feature[i], square_feature[i])
        # norm_feature = torch.div(feature, square_feature)
        return norm_feature

    def nearest_index(self):
        dist = torch.cdist(self.s_feature, self.t_feature)
        _, final_adj = torch.sort(dist, dim=1, descending=False)
        return final_adj

    def retrieval_matrix(self):
        rresult = torch.zeros(len(self.s_label), len(self.t_label))
        for i in range(rresult.shape[0]):
            pre_labels = torch.index_select(self.t_label, dim=0, index=self.final_adj[i, :])
            rresult[i, :][pre_labels.squeeze()==self.s_label[i]] = 1
        return rresult

    def class_num(self):
        max_class = self.s_label.max()+1
        s_class_num, t_class_num = [], []
        for i in range(max_class):
            s_class_num.append((self.s_label == i).float().sum())
            t_class_num.append((self.t_label == i).float().sum())
        return torch.tensor(s_class_num), torch.tensor(t_class_num)

def view_pool(feature, view_num=4):
    feature = feature.view(int(feature.shape[0]/float(view_num)), int(view_num), feature.shape[1])
    feature = torch.max(feature, dim=1)[0].view(feature.shape[0], feature.shape[2])
    return feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieval')
    parser.add_argument('--s', type=str)
    parser.add_argument('--t', type=str)
    args = parser.parse_args()
    source_root, target_root = args.s, args.t
    source = scio.loadmat(source_root)
    target = scio.loadmat(target_root)
    source_feature, source_label = torch.from_numpy(source['feat1']), torch.from_numpy(source['label'])
    target_feature, target_label = torch.from_numpy(target['feat1']), torch.from_numpy(target['label'])

    start = time.time()
    validation = Performance(source_feature, source_label, target_feature, target_label, cuda=False)
    NN, FT, ST, DCG, F_measure, ANMRR = validation.test_performance()
    _, AUC = validation.AUC()
    now = time.time()
    print('              计算用时：{}s      '.format(now-start))
    print("NN: {:.4f}\tFT: {:.4f}\tST: {:.4f}\tF: {:.4f}\tDCG: {:.4f}\tANMRR: {:.4f}\tAUC: {:.4f}\t".
          format(NN, FT, ST, F_measure, DCG, ANMRR, AUC))