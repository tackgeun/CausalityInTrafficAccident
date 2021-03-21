# coding: utf-8
import argparse, pickle, os, math, random, sys, time
from timeit import default_timer as timer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from dataset.loader import CausalityInTrafficAccident

from utils import *
from tensorboardX import SummaryWriter
from models import *

import pdb

parser = argparse.ArgumentParser(description='Training Framework for Temporal Cause and Effect Localization')

# Dataloader
parser.add_argument('--dataset_ver', type=str, default='Mar9th')
parser.add_argument('--use_flip', type=bool, default=True)
parser.add_argument('--feature', type=str, default="i3d-rgb-x8")
parser.add_argument('--input_size', type=int, default=1024)

# Architecture
parser.add_argument('--architecture_type', type=str, default='forward-SST', choices=['forward-SST', 'backward-SST', 'bi-SST', 'SSTCN-SST', 'SSTCN-R-C3D', 'SSTCN-Segmentation', 'MSTCN-Segmentation'])
#parser.add_argument('--feed_type', type=str, default='detection')
parser.add_argument('--prediction_type', type=str, default="both")

parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--loss_type', type=str, default='CrossEntropy')

# Action Detection (SST)
parser.add_argument('--positive_thres', type=float, default=0.4)
parser.add_argument('--sst_K', type=int, default=64)
#parser.add_argument('--sst_rnn_type', type=str, default='GRU')

# Action Segmentation (SSTCN, MSTCN)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_stages', type=int, default=2)
parser.add_argument('--w1', type=float, default=1.0)
parser.add_argument('--w2', type=float, default=1.0)
parser.add_argument('--w3', type=float, default=1.0)
parser.add_argument('--w4', type=float, default=1.0)
parser.add_argument('--mse_tau', type=float, default=4.0)

# Optimization
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument('--num_experiments', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=100)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--use_dropout', type=float, default=0.5)

parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--weight_decay', type=float, default=1e-2)

# Logging and Display
parser.add_argument('--display_period', type=int, default=101)
parser.add_argument('--logdir', type=str, default='runs')


args = parser.parse_args()

p = vars(args)

p['len_sequence'] = 208
p['fps'] = 25
p['vid_length'] = p['len_sequence'] * 8 / p['fps']

if('Segmentation' in p['architecture_type']):
    p['feed_type'] = 'multi-label'
elif('SST' in p['architecture_type']):
    p['feed_type'] = 'detection'

if('SST' in p['architecture_type']):
    p['sst_dt'] = p['vid_length'] / p['len_sequence']
    p["sst_K"] = args.sst_K
    p['proposal_scales'] = [float(i+1) * p['sst_dt'] for i in range(0, p["sst_K"])] # in seconds

if('MSTCN' in p['architecture_type']):
    p['config_layers'] = [args.num_layers for _ in range(0, args.num_stages)]

p['device'] = 0

print(p)

# Dataset
dataset_train = CausalityInTrafficAccident(p, split='train')
dataset_val   = CausalityInTrafficAccident(p, split='val', test_mode=True)
dataset_test  = CausalityInTrafficAccident(p, split='test', test_mode=True)

device = p['device']
dataloader_train = DataLoader(dataset_train, batch_size=p['batch_size'], shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=p['batch_size'])
dataloader_test = DataLoader(dataset_test, batch_size=p['batch_size'])

print("train/validation/test dataset size", \
        len(dataset_train), len(dataset_val), len(dataset_test))

# Logging
arch_name = p['architecture_type']
expdir = '%s-%s-batch%d-layer%d-embed-%d' % \
        (arch_name, p['feature'], p['batch_size'], p['num_layers'], p['hidden_size'])

if(p['use_dropout'] > 0.0):
    expdir = expdir + '-dropout%.1f' % p['use_dropout']

if(p['use_randperm'] > 0):
    expdir = expdir + '-randperm%d' % p['use_randperm']

logdir = './%s/%s/' % (args.logdir, expdir)

ei = 0
while(os.path.exists(logdir + '/%d/' % ei)):
    ei = ei + 1

exp_stats = dict()
for key in ['cause-thr-test', 'effect-thr-test', 'cause-thr-val', 'effect-thr-val']:
    exp_stats[key] = []

###################################
# Main Training Loop
###################################

for di in range(0, args.num_experiments):
    # Reproducibility
    if(args.random_seed > 0):
        torch.manual_seed(args.random_seed + di)
        np.random.seed(args.random_seed + di)
        random.seed(args.random_seed + di)
        torch.cuda.manual_seed(args.random_seed + di)
        torch.cuda.manual_seed_all(args.random_seed + di)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
    model = []

    if('Segmentation' in p['architecture_type']):
        if('SSTCN' in p['architecture_type']):
            model = SSTCN(p)
        elif('MSTCN' in p['architecture_type']):
            model = MSTCN(p)        
    elif('SST' in p['architecture_type']):
        if('SSTCN' in p['architecture_type']):
            model = SSTCNSequenceEncoder(p)
        else:
            model = SSTSequenceEncoder(p)
    elif('trivial' in p['architecture_type']):
        model = Trivial(p)
    model = model.cuda(device)

    logdir = './%s/%s/%d/%d/' % (args.logdir, expdir, ei, di)

    # tensorboard, stats
    stats = dict()
    stats['max-cause-iou-mean-val'] = 0
    stats['max-effect-iou-mean-val'] = 0
    stats['max-cause-iou-mean-test'] = 0
    stats['max-effect-iou-mean-test'] = 0
    writer = SummaryWriter(logdir)

    max_perf_val = 0.0

    # loss function
    if(args.loss_type == 'CrossEntropy'):
        p['criterion'] = CrossEntropy().cuda(device)
    elif(args.loss_type == 'WeightedCE'):
        p['criterion'] = WeightedCE().cuda(device)
        set_loss_weights(p['criterion'], labels, p['positive_thres'])

    if(args.optimizer == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif(args.optimizer == 'adamw'):
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # main loop
    for epoch in range(0, args.num_epochs):
        train_stats, train_loss = iterate_epoch(p, dataloader_train, model, optimizer)
        val_stats,   val_loss   = iterate_epoch(p, dataloader_val, model)

        perf_train, stats = update_epoch_stats(p, 'train', epoch, writer, stats, train_stats, train_loss)
        perf_val,   stats = update_epoch_stats(p, 'val', epoch, writer, stats, val_stats, val_loss)

        # update the validation best statistics and model
        if(perf_val >= max_perf_val):
            torch.save(model.state_dict(), logdir + 'model_max.pth')
            max_perf_val = perf_val

            if((p['prediction_type'] == 'cause' or p['prediction_type'] == 'both')):
                stats['max-cause-iou-thr-val'] = copy.deepcopy(stats['cause-iou-thr-val'])
                stats['max-cause-iou-mean-val'] = copy.deepcopy(stats['cause-iou-mean-val'])

            if((p['prediction_type'] == 'effect' or p['prediction_type'] == 'both')):
                stats['max-effect-iou-thr-val'] = copy.deepcopy(stats['effect-iou-thr-val'])
                stats['max-effect-iou-mean-val'] = copy.deepcopy(stats['effect-iou-mean-val'])

        if((epoch % args.display_period == 0) and (epoch != 0)):
            print('[epoch %d]' % epoch)
            if(p['prediction_type'] == 'cause' or p['prediction_type'] == 'both'):
                print('[cause] train/val/val max acc tIoU@0.5 : %.4f / %.4f / %.4f' % (stats['cause-iou-thr-train'][4], stats['cause-iou-thr-val'][4], stats['max-cause-iou-thr-val'][4]))

            if(p['prediction_type'] == 'effect' or p['prediction_type'] == 'both'):
                print('[effect] train/val/val max acc tIoU@0.5 : %.4f / %.4f / %.4f' % (stats['effect-iou-thr-train'][4], stats['effect-iou-thr-val'][4], stats['max-effect-iou-thr-val'][4]))

            if(p['prediction_type'] == 'both'):
                print('[both] train/val/val max acc tIoU@0.5 : %.4f / %.4f / %.4f' % ( (stats['cause-iou-thr-train'][4]+stats['effect-iou-thr-train'][4])/2,
                                                                                (stats['cause-iou-thr-val'][4]+stats['effect-iou-thr-val'][4])/2,
                                                                                (stats['max-cause-iou-thr-val'][4]+stats['max-effect-iou-thr-val'][4])/2
                                                                            ))
            #print('train/val loss %.4f %.4f' % (float(train_loss['w_all']), float(val_loss['w_all'])))
            print('train/val loss %.4f %.4f' % (float(train_loss['loss']), float(val_loss['loss'])))

    # evaluated the best validation model on test set.
    state_dict = torch.load(logdir + 'model_max.pth')
    model.load_state_dict(state_dict)
    test_stats, test_losses = iterate_epoch(p, dataloader_test, model)
    perf_test, stats = update_epoch_stats(p, 'test', epoch, writer, stats, test_stats, test_losses)

    exp_stats['cause-thr-val'].append(stats['max-cause-iou-thr-val'])
    exp_stats['cause-thr-test'].append(stats['cause-iou-thr-test'])

    exp_stats['effect-thr-val'].append(stats['max-effect-iou-thr-val'])
    exp_stats['effect-thr-test'].append(stats['effect-iou-thr-test'])

if(p['prediction_type'] == 'both'):
    cause_thr_test = torch.stack(exp_stats['cause-thr-test'], dim=0)
    effect_thr_test = torch.stack(exp_stats['effect-thr-test'], dim=0)
    both_thr_test = (cause_thr_test + effect_thr_test) / 2

    if(args.num_experiments > 1):
        print("cause/effect/both test max performance mean/std @ IoU=0.5")
        print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
                float(torch.mean(cause_thr_test[:, 4])),
                float(torch.std(cause_thr_test[:, 4])),
                float(torch.mean(effect_thr_test[:, 4])),
                float(torch.std(effect_thr_test[:, 4])),
                float(torch.mean(both_thr_test[:, 4])),
                float(torch.std(both_thr_test[:, 4])),
            ))
    else:
        print("cause/effect/both test max performance mean @ IoU=0.5")
        print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
                float(torch.mean(cause_thr_test[:, 4])),
                float(torch.mean(effect_thr_test[:, 4])),
                float(torch.mean(both_thr_test[:, 4])),
            ))

    print('Accuracy of Cause Localization @ IoU=[0.1:0.9]')
    print(torch.mean(cause_thr_test, dim=0))
    if(args.num_experiments > 1):
        print(torch.std(cause_thr_test, dim=0))
    torch.save(cause_thr_test.cpu(), './%s/%s/%d/cause.pth' % (args.logdir, expdir, ei))

    print('Accuracy of Effect Localization @ IoU=[0.1:0.9]')
    print(torch.mean(effect_thr_test, dim=0))
    if(args.num_experiments > 1):
        print(torch.std(effect_thr_test, dim=0))
    torch.save(effect_thr_test.cpu(), './%s/%s/%d/effect.pth' % (args.logdir, expdir, ei))

    print('Accuracy of Mean of Cause and Effect Localization @ IoU=[0.1:0.9]')
    print(torch.mean(both_thr_test, dim=0))
    if(args.num_experiments > 1):
        print(torch.std(both_thr_test, dim=0))
    torch.save(both_thr_test.cpu(), './%s/%s/%d/both.pth' % (args.logdir, expdir, ei))

    if(p['feed_type'] == 'detection'):
        pred = infer_epoch(p, dataloader_test, model, dataset_test.boxes)    
        torch.save(pred, './%s/%s/%d/prediction.pth' % (args.logdir, expdir, ei))
        print('file path')
        print('./%s/%s/%d/prediction.pth' % (args.logdir, expdir, ei))
    
    
