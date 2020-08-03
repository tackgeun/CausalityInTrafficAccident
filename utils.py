import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_optimizer(args, model):
    if(args.optimizer == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        assert(False)

    return optimizer

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

################################################################
# process_epoch
################################################################
def process_epoch(phase, _epoch, p, _dataloader, _model, _optim=None):

    losses = AverageMeter()
    top1_c = AverageMeter()
    top5_c = AverageMeter()
    top1_e = AverageMeter()
    top2_e = AverageMeter()

    if(phase == 'train'):
        _model.train()
    elif(phase == 'val'):
        _model.eval()
    elif(phase == 'test'):
        _model.eval()
        state_dict = torch.load(p['logdir'] + 'model_max.pth')
        _model.load_state_dict(state_dict)

    for iter, _data in enumerate(_dataloader):
        feat_rgb, label_cause, label_effect = _data
        batch_size = feat_rgb.size(0)
        if(phase=='train'):
            _optim.zero_grad()

        loss, logits = _model.forward_all(feat_rgb.cuda(), [label_cause.cuda(), label_effect.cuda()])

        if(phase=='train'):
            loss.backward()
            _optim.step()

        # measure accuracy and record loss
        prec1_c, prec5_c = accuracy(logits[0], label_cause.cuda(), topk=(1,2))
        prec1_e, prec2_e = accuracy(logits[1], label_effect.cuda(), topk=(1,2))

        losses.update(loss.item(), batch_size)
        top1_c.update(prec1_c.item(), batch_size)
        top5_c.update(prec5_c.item(), batch_size)
        top1_e.update(prec1_e.item(), batch_size)
        top2_e.update(prec2_e.item(), batch_size)

        stats = dict()
        stats['loss'] = losses.avg
        stats['top1.cause'] = top1_c.avg
        stats['top5.cause'] = top5_c.avg
        stats['top1.effect'] = top1_e.avg
        stats['top2.effect'] = top2_e.avg
        return stats