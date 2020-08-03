import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

################################################################
# TSN
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
            loss_effect = F.nll_loss(logits[0], labels[0])
        elif(self.consensus_type == 'linear'):
            loss_cause  = F.cross_entropy(logits[0], labels[0])
            loss_effect = F.cross_entropy(logits[1], labels[1])
        return loss_cause + loss_effect

    def forward_all(self, feat, labels):
        logits = self.forward(feat)
        loss = self.loss(logits, labels)

        return loss, logits