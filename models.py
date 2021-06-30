import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import pdb

################################################################
# TSN
# the model code is borrowed from the following repository.
# https://github.com/yjxiong/tsn-pytorch
################################################################
class Consensus(nn.Module):
    def __init__(self, p, dataset):
        super(Consensus, self).__init__()
        def get_score_module(class_dim):
            #return torch.nn.Conv2d(self.hidden_dim, class_dim,1)
            return torch.nn.Linear(self.hidden_dim, class_dim)
        self.consensus_type = p['consensus_type']
        self.num_segments = p['num_segments']

        self.hidden_dim = p["hidden_size"]    # default 128
        self.num_causes = dataset.num_causes
        self.num_effects= dataset.num_effects
        self.score_c = get_score_module(self.num_causes)
        self.score_e = get_score_module(self.num_effects)

        if(self.consensus_type == 'linear'):
            self.layer = torch.nn.Linear(self.hidden_dim * self.num_segments, self.hidden_dim)

    def forward(self, feat):
        if(self.consensus_type == 'average'):
            probs_c = F.softmax(self.score_c(feat), dim=2)
            probs_e = F.softmax(self.score_e(feat), dim=2)

            logit_c = torch.log(probs_c.mean(dim=1))
            logit_e = torch.log(probs_e.mean(dim=1))

        elif(self.consensus_type == 'linear'):
            feat = feat.view(feat.size(0), -1)
            feat_trans = self.layer(feat)
            logit_c = self.score_c(feat_trans)
            logit_e = self.score_e(feat_trans)
        else:
            assert(False)

        return [logit_c, logit_e]


class TSN(nn.Module):
    def __init__(self, p, dataset):
        super(TSN, self).__init__()

        # get options for TSN
        self.video_dim = p["input_size"]      # default 1024
        self.hidden_dim = p["hidden_size"]    # default 128
        
        self.dropout = p["use_dropout"]       # default 0.5

        self.num_causes = dataset.num_causes
        self.num_effects = dataset.num_effects
        
        self.consensus_type = p['consensus_type'] # ['avg', 'linear']
        self.num_segments = p['num_segments']

        def get_feature_module():
            return nn.Sequential(torch.nn.Linear(self.video_dim, self.hidden_dim),nn.Dropout(self.dropout), nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim), nn.Dropout(self.dropout), nn.ReLU())

        self.feat = get_feature_module()
        self.consensus = Consensus(p, dataset)

    def forward(self, feat):
        
        embed_feat = self.feat(feat)
        logit_c, logit_e = self.consensus(embed_feat)

        return [logit_c, logit_e]


    # loss function
    def loss(self, logits, labels):
        if(self.consensus_type == 'average'):
            loss_cause = F.nll_loss(logits[0], labels[0])
            loss_effect = F.nll_loss(logits[1], labels[1])
        elif(self.consensus_type == 'linear'):
            loss_cause  = F.cross_entropy(logits[0], labels[0])
            loss_effect = F.cross_entropy(logits[1], labels[1])
        return loss_cause + loss_effect

    def forward_all(self, feat, labels):
        logits = self.forward(feat)
        loss = self.loss(logits, labels)

        return loss, logits

########################################################################
# SSTCN and MSTCN
# https://github.com/yabufarha/ms-tcn
########################################################################

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, dropout):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        if(mask == None):
            return (x + out)
        else:
            return (x + out) * mask[:, 0:1, :]

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, dropout):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, dropout)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        
    def forward(self, x, mask=None):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)

        if(mask == None):
            out = self.conv_out(out)
        else:
            out = self.conv_out(out) * mask[:, 0:1, :]
        return out

class MultiStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, dropout):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers[0], num_f_maps, dim, num_classes, dropout)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(s, num_f_maps, num_classes, num_classes, dropout)) for s in num_layers[1:]])

    def forward(self, x, mask=None):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for sidx, s in enumerate(self.stages):
            if(mask==None):
                out = s(F.softmax(out, dim=1))
            else:
                out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

########################################################################
#
#    Container module with 1D convolutions to generate proposals
#    This code is from https://github.com/ranjaykrishna/SST/blob/master/models.py
#    and modified for integration.
# 
########################################################################
class SSTSequenceEncoder(nn.Module):
    def __init__(self, p):
        super(SSTSequenceEncoder, self).__init__()

        # get options for SST
        self.rnn_type = 'GRU'
        self.video_dim = p["input_size"] # 500
        self.hidden_dim = p["hidden_size"] # hdim == 512
        self.K = p["sst_K"] # 64 # number of proposals
        self.arch_type = p['architecture_type'] # "GRU"
        self.rnn_num_layers = p["num_layers"] # 2
        self.rnn_dropout = p["use_dropout"]  # 0.2

        # get layers of SST
        if('forward' in self.arch_type):
            self.rnn = getattr(nn, self.rnn_type)(self.video_dim, self.hidden_dim,
                    self.rnn_num_layers, batch_first=True, dropout=self.rnn_dropout, bidirectional=False)
        else:
            self.rnn = getattr(nn, self.rnn_type)(self.video_dim, self.hidden_dim,
                    self.rnn_num_layers, batch_first=True, dropout=self.rnn_dropout, bidirectional=True)                
        
        if('bi' in self.arch_type):
            self.scores = torch.nn.Linear(self.hidden_dim*2, self.K * 3) # 3 = bg + cause + effect
        else:
            self.scores = torch.nn.Linear(self.hidden_dim, self.K * 3) # 3 = bg + cause + effect

    def forward(self, features):

        # dealing with batch size 1
        if len(features.size()) == 2:
            features = torch.unsqueeze(features, 0)
        B, L, _ = features.size()

        rnn_output, _ = self.rnn(features) # [B, L, hdim]

        if('forward' in self.arch_type):
            rnn_output = rnn_output.contiguous().view(-1, self.hidden_dim) # [B*L, hdim]
        else:
            rnn_output = rnn_output.contiguous().view(-1, self.hidden_dim*2) # [B*L, hdim]

        if('backward' in self.arch_type):
            rnn_output = rnn_output[:, 128:]

        outputs = self.scores(rnn_output) # [B*L, K*3]
        outputs = outputs.view(-1, 3) # [B*L*K, 3]
        
        return outputs     

class SSTCNSequenceEncoder(nn.Module):
    """
    Container module with 1D convolutions to generate proposals
    This code is from https://github.com/ranjaykrishna/SST/blob/master/models.py
    and modified for integration.
    """

    def __init__(self, p):
        super(SSTCNSequenceEncoder, self).__init__()

        # get options for SST
        self.video_dim = p["input_size"] # 500
        self.hidden_dim = p["hidden_size"] # hdim == 512
        self.K = p["sst_K"] # 64 # number of proposals
        self.dropout_rate = p["use_dropout"]  # 0.2
        self.num_layers = p["num_layers"]

        # get layers of SST
        self.layers = SingleStageModel(self.num_layers, self.hidden_dim, self.video_dim, self.K * 3, self.dropout_rate)

    def forward(self, features):
        #pdb.set_trace()
        # dealing with batch size 1
        if len(features.size()) == 2:
            features = torch.unsqueeze(features, 0)
        B, L, _ = features.size()

        features = features.transpose(1,2)
        outputs = self.layers(features) # [B, L, hdim]
        outputs = outputs.transpose(1,2)
        outputs = outputs.reshape(-1, 3) # [B*L*K, 3]

        return outputs

class CrossEntropy(nn.Module):
    """
    Weighted CE is adopted Weighted BCE from https://github.com/ranjaykrishna/SST/blob/master/models.py
    and modified for integration.
    """

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, outputs, labels):
        # logsoftmax = F.log_softmax(outputs, dim=1)
        # onehot = labels.new_zeros((labels.size(0),3))
        # onehot.scatter_(1,labels.unsqueeze(1),1)
        # loss = - (torch.sum(logsoftmax * onehot)).mean() / outputs.size(0)
        loss = F.cross_entropy(outputs, labels)

        return loss        




###########


class SSTCN(nn.Module):
    def __init__(self, p):
        super(SSTCN, self).__init__()
        
        hidden_size = p['hidden_size']
        num_layers = p['num_layers']
        len_sequence = p['len_sequence']
        num_preds = 3 # Backward, Cause, Effect

        if('i3d' in p['feature']):
            if('both' in p['feature']):
                self.use_rgb = True                
                self.use_flow = True
                
            elif('flow' in p['feature']):
                self.use_rgb = False
                self.use_flow = True
                
            elif('rgb' in p['feature']):
                self.use_rgb = True
                self.use_flow = False
                
            else:
                assert(False)
        else:            
            assert(False)

        ## BiLSTM for temporally-aware feature
        if(self.use_flow and self.use_rgb):
            input_size = p['input_size']*2
        else:
            input_size = p['input_size']

        # self.use_bn_input = p['use_bn_input']
        # if(p['use_bn_input']):
        #     self.bn = nn.BatchNorm1d(input_size, affine=False)

        self.layers = SingleStageModel(num_layers, hidden_size, input_size, num_preds, p['use_dropout'])

    def forward(self, rgb, flow):
        if (self.use_flow and self.use_rgb):
            inputs = torch.cat([rgb, flow], dim=2)
        elif (self.use_rgb):
            inputs = rgb
        elif (self.use_flow):
            inputs = flow

        inputs = inputs.transpose(1,2)

        # if(self.use_bn_input):
        #     inputs = self.bn(inputs)

        logits = self.layers.forward(inputs)

        return logits

class MSTCN(nn.Module):
    def __init__(self, p):
        super(MSTCN, self).__init__()
        
        hidden_size = p['hidden_size']
        num_layers = p['num_layers']
        # num_stages = p['num_stages']
        # num_output = p['num_output']
        len_sequence = p['len_sequence']
        stage_config = p['mstcn_stage_config']
        num_preds = 3 # Backward, Cause, Effect
        
        if('i3d' in p['feature']):
            if('both' in p['feature']):
                self.use_rgb = True                
                self.use_flow = True
            elif('flow' in p['feature']):
                self.use_rgb = False
                self.use_flow = True
            elif('rgb' in p['feature']):
                self.use_rgb = True
                self.use_flow = False
            else:
                assert(False)
        else:            
            assert(False)

        ## BiLSTM for temporally-aware feature
        if(self.use_flow and self.use_rgb):
            input_size = p['input_size']*2
        else:
            input_size = p['input_size']

        self.layers = MultiStageModel(stage_config, hidden_size, input_size, num_preds, p['use_dropout'])

    def forward(self, rgb, flow):
        if (self.use_flow and self.use_rgb):
            inputs = torch.cat([rgb, flow], dim=2)
        elif (self.use_rgb):
            inputs = rgb
        elif (self.use_flow):
            inputs = flow

        inputs = inputs.transpose(1,2)

        logits = self.layers.forward(inputs)

        return logits      


################################################################
# IRM
################################################################

class TSN_IRM(TSN):
    def __init__(self, p, dataset, irm_source, irm_target):
        super(TSN, self).__init__(p, dataset)
        self.irm_source = irm_source
        self.irm_target = irm_target

    def loss(self, logits, labels):
        logit_s = logits[0]
        logit_t = logits[1]
        if(self.consensus_type == 'average'):
            loss_cause = F.nll_loss(logits_s, labels[0])
            loss_effect = F.nll_loss(logits_t, labels[1])
        elif(self.consensus_type == 'linear'):
            loss_cause  = F.cross_entropy(logits_s, labels[0])
            loss_effect = F.cross_entropy(logits_t, labels[1])
        
def penalty(logits, y, criterion_fun, inv_val=1.0):
    scale = torch.tensor(inv_val).cuda().requires_grad_()
    loss = criterion_fun(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)
