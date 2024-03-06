import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import dataset
from model import AlexNet, l2norm
import utils
from torch.autograd import Variable
import torch.nn.functional as F
from feature_alex2 import test_pre
from performance import Performance
import random
import time
from torch.utils.tensorboard import SummaryWriter


def lr_schedule(opt, itera, mult):
    lr = init_lr / pow(1 + 0.001 * itera, 0.75)
    for ind, param_group in enumerate(opt.param_groups):
        param_group['lr'] = lr * mult[ind]
    return lr


def adaptation_factor(x):
    if x>= 1.0:
        return 1.0
    den = 1.0 + math.exp(-10 * x)
    lamb = 2.0 / den - 1.0
    return lamb


def output(mes):
    print(mes)
    log.write(mes)
    log.write('\n')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_board(writer, infos, itera):
    '''
    write loss curve
    writer: tensorboard writer
    infos: dict. {"loss_name": loss_value}
    itera: iterations
    '''
    for loss_name, loss in infos.items():
        writer.add_scalar(loss_name, loss, global_step=itera)


parser = argparse.ArgumentParser()
parser.add_argument('--s', default=0, type=int)
parser.add_argument('--t', default=1, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--da', default=1, type=int)                # 1 for doing domain adaptation
parser.add_argument("--data_root", default="")
parser.add_argument("--init_lr", default=5e-3, type=float)
parser.add_argument("--num_itera", default=10000, type=int)
parser.add_argument("--seed", default=666, type=int)
parser.add_argument("--name", default="", type=str)
parser.add_argument('--kl_scale', type=float, default=0.005)

args = parser.parse_args()
resume = args.resume
da = args.da
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = torch.cuda.is_available()
seed = args.seed
kl_scale = args.kl_scale
init_lr = args.init_lr
batch_size_s = 128
batch_size_t = 128
lr_mult = [0.1, 0.2, 1, 2]

save_root = os.path.join("results", args.name)
os.makedirs(save_root, exist_ok=True)
log = open(os.path.join(save_root, 'output.log'), 'w')
writer = SummaryWriter(os.path.join(save_root, "tensorboard")) #zhr
pretrain_path = resume
checkpoint_save_path = os.path.join(save_root, 'model.pth')

output('GPU: {}'.format(args.gpu))
output('name:{}, batch_size: {}, init_lr: {}'.format(args.name, batch_size_s, init_lr))
output('lr_mult: {}, resume: {}, da: {}'.format(lr_mult, resume, da))

if seed:
    output('Fix seed :{}'.format(seed))
    setup_seed(seed)

data_root = args.data_root
print(data_root)
source_view = 1
target_view = 12

s_loader, t_loader, s_val_loader, t_val_loader, n_class = dataset.data_loader(data_root, 'train', source_view, target_view,
                                                                         batch_size_s, batch_size_t, 64, 64)

s_loader_len, t_loader_len, s_val_len, t_val_len = len(s_loader), len(t_loader), len(s_val_loader), len(t_val_loader)
output('batchsize: {}, s_loader:{}, t_batch:{}, s_val_batch:{}, t_val_batch:{}'.format(
        batch_size_s, s_loader_len, t_loader_len, s_val_len, t_val_len))

model = AlexNet(cudable=cuda, n_class=n_class,batch_size=batch_size_s)

if cuda:
    model.cuda()

opt = model.get_optimizer(init_lr, lr_mult)
# resume or init
if not resume == '':
    pretrain = torch.load(pretrain_path)
    model.load_state_dict(pretrain['model'])
    opt.load_state_dict(pretrain['opt'])
    itera = pretrain['itera'] # need change to 0 when DA
else:
    model.load_state_dict(utils.load_pth_model(), strict=False)
    itera = 0

output('    \t\t\t=======    START TRAINING    =======    ')
old_correct=0
start = time.time()
if not resume == '':
    t_loader_itera = iter(t_loader)
    s_loader_itera = iter(s_loader)
with torch.autograd.set_detect_anomaly(True):
    for itera in range(itera, args.num_itera):
        model.train()
        lamb = adaptation_factor(itera * 1.0 / args.num_itera)
        current_lr= lr_schedule(opt, itera, lr_mult)

        if itera % s_loader_len == 0:
            s_loader_itera = iter(s_loader)
        if itera % t_loader_len == 0:
            t_loader_itera = iter(t_loader)
        
        xs, ys = s_loader_itera.next()
        N, V, C, H, W = xs.size()
        xs = Variable(xs).view(-1, C, H, W).cuda()   
        xt, yt = t_loader_itera.next()
        N, V, C, H, W = xt.size()
        xt = Variable(xt).view(-1, C, H, W).cuda()
        
        if cuda:
            xs, ys = xs.cuda(), ys.cuda()
            xt, yt = xt.cuda(), yt.cuda()
            
        xs_feat, xs_score, xs_pred, xs_mean, xs_var = model.forward(xs.float().squeeze(), bs=batch_size_s, Instance=True)
        C_loss = model.closs(xs_score, ys)

        if da:
            xt_feat, _, xt_pred, xt_mean, xt_var = model.forward(xt.squeeze(), bs=batch_size_t, Instance=True)
            feat_klloss = (model.klloss(xs_mean, xs_var) + model.klloss(xt_mean, xt_var))/2

            ad_loss = model.ad(l2norm(xs_feat), l2norm(xt_feat))

            xt_pred_onehot = torch.max(xt_pred, 1)[1]
            s_c_mean, s_c_var, s_centroid = model.centroid(xs_feat, ys, batch_size_s)
            t_c_mean, t_c_var, t_centroid = model.centroid(xt_feat, xt_pred_onehot, batch_size_t)
            centroid_klloss = (model.klloss(s_c_mean, s_c_var) + model.klloss(t_c_mean, t_c_var))/2

            kl_loss = feat_klloss + centroid_klloss

            semantic_loss = model.MSEloss(s_centroid, t_centroid)

            Dregloss, Gregloss = model.regloss()
            Reg_loss = Dregloss + Gregloss

            F_loss = C_loss + ad_loss + kl_scale * kl_loss + lamb * semantic_loss + Reg_loss

            # log loss
            loss_dict = {
                "Reg_loss": Reg_loss, 
                "C_loss": C_loss,
                "ad_loss": ad_loss,
                "kl_loss": kl_loss,
                "semantic_loss": semantic_loss
            }
            write_board(writer, loss_dict, itera)

            opt.zero_grad()
            F_loss.backward()
            opt.step()
        else:
            F_loss = C_loss + Reg_loss
            opt.zero_grad()
            F_loss.backward()
            opt.step()

            # log loss
            loss_dict = {
                "C_loss": C_loss,
            }
            write_board(writer, loss_dict, itera) #zhr
    
        if itera % 50 == 0:
            s_pred_label = torch.max(xs_pred, 1)[1]
            s_correct = torch.sum(torch.eq(s_pred_label, ys).float())
            s_acc = torch.div(s_correct, ys.size(0))
            now = time.time()
            output('itera: {}, lr: {:.4f}'.format(itera, current_lr))

            if da:
                output('time: {:.2f}s, s_acc: {:.4f}, '
                        'C_loss: {:.4f}, ad_loss: {:.4f}, '
                       'kl_loss: {:.4f}, semantic_loss: {:.4f}, Reg_loss: {:.4f}, '
                       'Totol: {:.4f}'.format(
                        (now - start), s_acc.item(),
                        C_loss.item(), ad_loss, 
                        kl_loss.item(), lamb*semantic_loss, Reg_loss,
                        F_loss.item()))
            else:
                output('time: {:.2f}s, s_acc: {:.4f}, C_loss: {:.4f}, Totol: {:.4f}'.format(
                        (now - start), s_acc.item(), C_loss.item(), F_loss.item()))
            start = time.time()
    
        
        # validation
        if itera % 500 == 0 and itera != 0:
            output('    \t\t\t=======    START VALIDATION    =======    ')
            model.eval()
            s_val_feature, s_val_label, _, s_val_acc = test_pre(model, s_val_loader)
            t_val_feature, t_val_label, _, t_val_acc = test_pre(model, t_val_loader, Instances=True)
            output('\t\t\tTrain S_validation_1:{:.4f}  T_validation_2:{:.4f}'.format(s_val_acc, t_val_acc))

            validation = Performance(s_val_feature, s_val_label, t_val_feature, t_val_label, cuda=False)
            NN = validation.top_k(k=1)
            FT, ST, DCG = validation.ft_st_dcg()
            output("\t\tFirst: NN: {:.4f}\tFT: {:.4f}\tST: {:.4f}\tDCG: {:.4f}".
                   format(NN, FT, ST, DCG))
            v_acc = (NN+FT+ST+DCG)/4.0
            output('\t\t\titera: {}, v_acc: {:.4f}'.format(itera, v_acc))


        # save model
        if (itera % 500 == 0 and itera != 0 and v_acc>old_correct):
            output('   \t\t\t =======    SAVE MODEL    =======    ')
            old_correct = v_acc
            torch.save({
                'itera': itera + 1,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
            }, checkpoint_save_path)
        itera += 1

