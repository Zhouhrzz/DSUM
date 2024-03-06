from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import numpy as np
from utils import *
import torch.nn.functional as F
import os
import scipy.io as scio
# Training settings
from model import AlexNet
import dataset
from feature_alex2 import test_pre
from performance import Performance

def pred_t(t_feature, s_centroid):
    similarity_f = nn.CosineSimilarity(dim=2)
    sim = similarity_f(t_feature.unsqueeze(1), s_centroid.unsqueeze(0)) / 0.5  # 80,21
    sim = nn.Softmax(dim=-1)(sim)
    return sim

def update_centroid(feature, label, n_class):
    n, d = feature.shape
    ones = torch.ones_like(label, dtype=torch.float) # .cuda()
    zeros = torch.zeros(n_class) # .cuda()
    n_classes = zeros.scatter_add(0, label, ones)

    ones = torch.ones_like(n_classes) # .cuda()
    n_classes = torch.max(n_classes, ones)

    zeros = torch.zeros(n_class, d) # .cuda()
    sum_feature = zeros.scatter_add(0, torch.transpose(label.repeat(d, 1), 1, 0), feature.float())
    current_centroid = torch.div(sum_feature, n_classes.view(n_class, 1))
    return current_centroid# .cuda()

def fine_feature(feature, centroid, label, select=True):
    feature2 = torch.index_select(centroid, dim=0, index=label.squeeze().long())
    if select:
        y_pred2 = pred_t(feature, centroid)
        y_pred2 = nn.Softmax(dim=-1)(y_pred2)
        y_pred2 = torch.max(y_pred2, dim=1)[1]
        mask = torch.eq(label.squeeze(), y_pred2).bool()
        feature2[(1 - mask.float()).bool(), :] = 0
    feature = F.relu((feature + feature2), inplace=True)
    return feature

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model_path', default="", type=str)
    parser.add_argument('--data', default="test", type=str)
    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--save_feature', default=False, type=bool)
    parser.add_argument('--fine_feature', default=False, type=bool)


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    
    source_view = 1
    target_view = 12
    data_root = args.data_root
    s_loader, t_loader, n_class = dataset.data_loader(data_root, args.data, source_view, target_view,
                                                                         64, 64)

    model = AlexNet(cudable=cuda, n_class=n_class, batch_size=1)
    pretrain_path = args.model_path
    pretrain = torch.load(pretrain_path)
    model.load_state_dict(pretrain['model'])
    model.cuda()
    model.eval()

    s_val_feature, s_val_label, s_val_pred_label, s_val_acc = test_pre(model, s_loader)
    t_val_feature, t_val_label, t_val_pred_label, t_val_acc = test_pre(model, t_loader, Instances=True)
    print('\t\t\tTest:  S_validation: {:.4f}\tT_validation: {:.4f}'.format(s_val_acc, t_val_acc))

    # 重新计算类别中心特征
    # s_test_centroid = update_centroid(s_val_feature, s_val_label.squeeze(), n_class)
    # t_test_centroid = update_centroid(t_val_feature, t_val_pred_label.squeeze(), n_class)
    if args.fine_feature:
        # assert os.path.exists(os.path.join(os.path.dirname(args.model_path), 'test_train_s_centroid.npy'))
        # assert os.path.exists(os.path.join(os.path.dirname(args.model_path), 'test_train_t_centroid.npy'))
        # s_train_centroid = torch.from_numpy(np.load(os.path.join(os.path.dirname(args.model_path), 'test_train_s_centroid.npy')))
        # t_train_centroid = torch.from_numpy(np.load(os.path.join(os.path.dirname(args.model_path), 'test_train_t_centroid.npy')))

        s_val_feature = fine_feature(s_val_feature, s_centroid.cpu(), s_val_label, select=False)
        t_val_feature = fine_feature(t_val_feature, s_centroid.cpu(), t_val_pred_label, select=False)


    if args.save_feature:
        save_feature_source = os.path.join(os.path.dirname(args.model_path), args.data + "_source.mat")
        save_feature_target = os.path.join(os.path.dirname(args.model_path), args.data + "_target.mat")
        scio.savemat(save_feature_target, {'feature': t_val_feature.numpy(), 'label': t_val_label.numpy()})
        scio.savemat(save_feature_source, {'feature': s_val_feature.numpy(), 'label': s_val_label.numpy()})
        # np.save(os.path.join(os.path.dirname(args.model_path), args.data + "_s_centroid.npy"), s_test_centroid.numpy())
        # np.save(os.path.join(os.path.dirname(args.model_path), args.data + "_t_centroid.npy"), t_test_centroid.numpy())
    
    
    validation = Performance(s_val_feature, s_val_label, t_val_feature, t_val_label, cuda=False)
    NN, FT, ST, DCG, F_measure, ANMRR = validation.test_performance()
    _, AUC = validation.AUC()
    print("\t\tFirst:  NN: {:.4f}\tFT: {:.4f}\tST: {:.4f}\tF: {:.4f}\tDCG: {:.4f}\t"
           "ANMRR: {:.4f}\tAUC: {:.4f}\t".format(NN, FT, ST, F_measure, DCG, ANMRR, AUC))
    
