# coding: utf-8
import argparse, os #pickle, os, #math, random, sys, time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from dataset.loader import CausalityInTrafficAccident
from tensorboardX import SummaryWriter
from models import TSN

parser = argparse.ArgumentParser(description='Training Framework for Cause and Effect Event Classification')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--feature', type=str, default="i3d-rgb-x8")

parser.add_argument('--input_size', type=int, default=1024)
parser.add_argument('--hidden_size', type=int, default=256)

parser.add_argument('--loss_type', type=str, default='CrossEntropy')
parser.add_argument('--num_experiments', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=2000)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--use_dropout', type=float, default=0.5)

parser.add_argument('--architecture_type', type=str, default='TSN')
parser.add_argument('--consensus_type', type=str, default='average')
parser.add_argument('--num_segments', type=int, default=4)
parser.add_argument('--new_length', type=int, default=1)

parser.add_argument('--dataset_ver', type=str, default='Mar9th')
parser.add_argument('--feed_type', type=str, default='classification')
parser.add_argument('--logdir', type=str, default='runs')

parser.add_argument("--random_seed", type=int, default=0)

args = parser.parse_args()

if(args.random_seed > 0):
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
p = vars(args)
print(args)

p['device'] = 0

dataset_train = CausalityInTrafficAccident(p, split='train')
dataset_val   = CausalityInTrafficAccident(p, split='val', test_mode=True)
dataset_test  = CausalityInTrafficAccident(p, split='test', test_mode=True)

device = p['device']
dataloader_train = DataLoader(dataset_train, batch_size=p['batch_size'], shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=p['batch_size'])
dataloader_test = DataLoader(dataset_test, batch_size=p['batch_size'])

print("train/validation/test dataset size", \
        len(dataset_train), len(dataset_val), len(dataset_test))


#################################
# logging directory
#################################
expdir = '%s-%s-batch%d-embed-%d' % \
        (p['architecture_type'], p['feature'], p['batch_size'], p['hidden_size'])

if(p['use_dropout'] > 0.0):
    expdir = expdir + '-dropout%.1f' % p['use_dropout']

logdir = './%s/%s/' % (args.logdir, expdir)

ei = 0
while(os.path.exists(logdir + '/%d/' % ei)):
    ei = ei + 1

#################################
# main loop
#################################

for di in range(0, args.num_experiments):
    p['logdir'] = './%s/%s/%d/%d/' % (args.logdir, expdir, ei, di)
    if(not os.path.exists(p['logdir'])):
        os.makedirs(p['logdir'])

    model = []
    model = TSN(p, dataset_train)
    model = model.cuda(device)

    optim = get_optimizer(args, model)

    max_perf_val = 0.0    
    max_perf_aux = 0.0
    for epoch in range(0, args.num_epochs):
        stats_train = process_epoch('train', epoch, p, dataloader_train, model, optim)
        stats_val = process_epoch('val', epoch, p, dataloader_val, model)
        
        perf_val = stats_val['top1.cause'] + stats_val['top1.effect']
        perf_val_aux = stats_val['top2.cause'] + stats_val['top2.effect']
        if(perf_val >= max_perf_val):
            if(perf_val_aux >= max_perf_aux):
                max_perf_val = perf_val
                max_perf_aux = perf_val_aux
                torch.save(model.state_dict(), p['logdir'] + 'model_max.pth')        

    stats_test = process_epoch('test', epoch, p, dataloader_test, model)
    print(stats_test)