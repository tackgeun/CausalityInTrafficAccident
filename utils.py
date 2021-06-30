import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math

import pdb

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

#####################################################################
# process_epoch
#####################################################################
def process_epoch(phase, _epoch, p, _dataloader, _model, _optim=None):
    losses = AverageMeter()
    top1_c = AverageMeter()
    top2_c = AverageMeter()
    top1_e = AverageMeter()
    top2_e = AverageMeter()
    top1_all = AverageMeter()
    
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
        prec1_c, prec2_c = accuracy(logits[0], label_cause.cuda(), topk=(1,2))
        prec1_e, prec2_e = accuracy(logits[1], label_effect.cuda(), topk=(1,2))

        losses.update(loss.item(), batch_size)
        top1_c.update(prec1_c.item(), batch_size)
        top2_c.update(prec2_c.item(), batch_size)
        top1_e.update(prec1_e.item(), batch_size)
        top2_e.update(prec2_e.item(), batch_size)

        stats = dict()
        stats['loss'] = losses.avg
        stats['top1.cause'] = top1_c.avg
        stats['top2.cause'] = top2_c.avg
        stats['top1.effect'] = top1_e.avg
        stats['top2.effect'] = top2_e.avg
        return stats


def compute_exact_overlap(logits, cause_gt, effect_gt, pred_type='both'):
    # logits: prediction (B, C, T)
    # gt: ground truth (B, 4) - cause start/end effect start/end
    
    _, _label = torch.max(logits, dim=1, keepdim=False)

    #print("compute_exact_overlap:", logits.size(), _label.size(), _label.min(),_label.max())

    def _count_iou(pred_label, _cls, cls_gt):
        B = pred_label.size(0)
        T = pred_label.size(1)

        dt = 1/float(T)

        _gt = torch.zeros(B,T)

        for b in range(0, B):
            _inter = 0
            t1, t2 = float(cls_gt[b,0])*float(T), float(cls_gt[b,1])*float(T)

            s_t1, e_t1 = math.floor(t1), math.ceil(t1)
            s_t2, e_t2 = math.floor(t2), math.ceil(t2)

            if(s_t1 == s_t2):
                _gt[b, s_t1] = t2-t1
            else:
                _gt[b, s_t1] = e_t1 - t1
                _gt[b, s_t2] = t2 - s_t2
                _gt[b, e_t1:s_t2] = 1
        
        inter = torch.sum(_gt * (pred_label == _cls).float(), dim=1, keepdim=False)
        union = torch.sum((pred_label == _cls).float(), dim=1, keepdim=False) + (cls_gt[:, 1] - cls_gt[:, 0])*float(T) - inter
            
        return inter/union

    if(pred_type == 'both'):
        return _count_iou(_label, 1, cause_gt), _count_iou(_label, 2, effect_gt)
    elif(pred_type == 'cause'):
        return _count_iou(_label, 1, cause_gt), []
    elif(pred_type == 'effect'):
        return [], _count_iou(_label, 1, effect_gt)

def compute_temporalIoU(iou_set):
    cnt = torch.zeros(9) # [0.1 ~ 0.9]
    for bi in range(0, len(iou_set)):
        for thr in range(1, 10):
            if(iou_set[bi] >= float(thr)/10.0):
                cnt[thr-1] = cnt[thr-1] + 1
    cnt = cnt / len(iou_set)

    return cnt

def compute_topk(logits, ious, topk=1):
    # logits: prediction (B*L*S, Class)
    # gt: ground truth (Batch, Len, Scales, Class) - cause start/end effect start/end
    # print('compute topk', logits.size(), ious.size())

    B, C, L, S = ious.size()

    if(C == 3): # bg, cause, effect
        logits = logits.view(B,L*S,C)
        max_val, max_idx = torch.max(logits, dim=2)
    elif(C == 2): # bg, prop
        logits = logits.view(B,L*S,1)
    
    def get_iou_from_top1(val, idx, _cls, ious):
        cls_val = val * (idx == _cls).float()
        lin_idx = torch.argmax(cls_val,dim=1)
        #print('get_iou_from_top1', lin_idx, _cls)
        res_iou = []
        for bi, idx in enumerate(lin_idx):
            _ious = ious[bi][_cls].view(-1)
            #print('class-%d batch-%d' % (idx, bi), _ious[lin_idx])
            res_iou.append(float(_ious[lin_idx[bi]]))
        return res_iou

    def get_iou_from_topk(val, idx, _cls, ious, topk=1):
        cls_val = val * (idx == _cls).float()
        max_val, lin_idx = torch.sort(cls_val,dim=1, descending=True)
        #print('get_iou_from_top1', lin_idx, _cls)
        res_iou = []
        for bi, idx in enumerate(lin_idx):
            _ious = ious[bi][_cls].view(-1)
            #print('class-%d batch-%d' % (idx, bi), _ious[lin_idx])
            res_iou.append(float(_ious[lin_idx[bi]]))
        return res_iou

    if(C == 3):
        top1_iou_cause = get_iou_from_top1(max_val, max_idx, 1, ious)
        top1_iou_effect= get_iou_from_top1(max_val, max_idx, 2, ious)
        return top1_iou_cause, top1_iou_effect

        return top1_iou_prop, 0

def add_loss(w1, loss1, train_loss):
    loss1 = float(loss1.cpu())
    if w1 in train_loss:
        train_loss[w1].append(loss1)
    else:
        train_loss[w1] = [loss1]

def write_loss(losses, epoch, prefix, writer):
    for k in losses.keys():
        losses[k] = torch.mean(torch.FloatTensor(losses[k]))
        writer.add_scalar('loss/%s/%s' % (prefix,k ), losses[k], epoch)
    #writer.add_scalars('loss/%s' % prefix, losses, epoch)

#####################################################################
# iterate_epoch
#####################################################################
def iterate_epoch(p, dataloader, model, optimizer=None):
    if(optimizer == None):
        model.eval()
    else:
        model.train()

    stats = dict()
    stats['cause-iou-set'] = []
    stats['effect-iou-set'] = []

    losses = dict()

    num_samples = 0
    for i_batch, v in enumerate(dataloader):

        if('Segmentation' in p['architecture_type']):
            (rgb, flow, causality_mask, cause_reg, effect_reg) = v
            causality_mask = causality_mask.cuda(p['device'])
        elif('SST' in p['architecture_type']):
            (rgb, flow, cause_reg, effect_reg, labels, ious) = v
            ious = ious.cuda(p['device'])
            labels = labels.cuda(p['device'])
        else:
            (rgb, flow, cause_reg, cause_mask, effect_reg, effect_mask, causality_mask) = v
            cause_mask = cause_mask.cuda(p['device'])
            effect_mask = effect_mask.cuda(p['device'])
            causality_mask = causality_mask.cuda(p['device'])

        cause_reg = cause_reg.cuda(p['device'])
        effect_reg = effect_reg.cuda(p['device'])

        # data to gpu
        rgb = rgb.cuda(p['device'])
        flow = flow.cuda(p['device'])

        if(optimizer != None):
            optimizer.zero_grad()

        # forward
        if('Segmentation' in p['architecture_type']):
            if 'MSTCN' in p['architecture_type']:
                loss1 = 0
                loss2 = 0
                
                _logits = model(rgb, flow)
                for logits in _logits:
                    loss1 += F.cross_entropy(logits, causality_mask, reduction='mean')
                    loss2 += torch.mean(torch.clamp(F.mse_loss(F.log_softmax(logits[:, :, 1:], dim=1), F.log_softmax(logits.detach()[:, :, :-1], dim=1),reduction="none"), min=0, max=p['mse_tau']*p['mse_tau']))
            else:
                logits = model(rgb, flow)
                loss1 = F.cross_entropy(logits, causality_mask, reduction='mean')
                loss2 = torch.mean(torch.clamp(F.mse_loss(F.log_softmax(logits[:, :, 1:], dim=1), F.log_softmax(logits.detach()[:, :, :-1], dim=1),reduction="none"), min=0, max=p['mse_tau']*p['mse_tau']))

            loss = loss1 * p['w1'] + loss2 * p['w2']

        elif('SST' in p['architecture_type']):
            if('both' in p['feature']):
                inputs = torch.cat([rgb, flow], dim=2)
            elif('rgb' in p['feature']):
                inputs = rgb
            logits = model(inputs)
            
            # print('ssd forward', logits.size(), labels.size())
            if('MSTCN' in p['architecture_type']):
                loss = 0
                for _logit in logits:
                    loss += p['criterion'](_logit, labels.view(-1))
            else:
                loss = p['criterion'](logits, labels.view(-1))

        # backward & training
        if(optimizer != None):
            loss.backward()
            optimizer.step()

        # accumulate tIoU
        if('Segmentation' in p['architecture_type']):
            cause_iou, effect_iou = compute_exact_overlap(logits.cpu(), cause_reg.cpu(), effect_reg.cpu(), p['prediction_type'])
            if(p['prediction_type'] == 'cause' or p['prediction_type'] == 'both'):
                for bi in range(0, cause_iou.size(0)):
                    stats['cause-iou-set'].append(float(cause_iou[bi].item()))

            if(p['prediction_type'] == 'effect' or p['prediction_type'] == 'both'):
                for bi in range(0, effect_iou.size(0)):
                    stats['effect-iou-set'].append(float(effect_iou[bi].item()))

        elif('SST' in p['architecture_type']):
            if('MSTCN' in p['architecture_type']):
                logits = logits[-1] # take the prediction from the last stage

            if('SST' in p['architecture_type']):
                cause_iou, effect_iou = compute_topk(logits, ious, 1)

            if(p['prediction_type'] == 'cause' or p['prediction_type'] == 'both'):
                stats['cause-iou-set'] = stats['cause-iou-set'] + cause_iou
                # for bi in range(0, cause_iou.size(0)):
                #   stats['cause-iou-set'].append(float(cause_iou[bi].item()))

                #print("cause_iou", cause_iou)

            if(p['prediction_type'] == 'effect' or p['prediction_type'] == 'both'):
                stats['effect-iou-set'] = stats['effect-iou-set'] + effect_iou
                # for bi in range(0, effect_iou.size(0)):
                #     stats['effect-iou-set'].append(float(effect_iou[bi].item()))
                #print("effect_iou", effect_iou)

        else:
            if(p['prediction_type'] == 'cause' or p['prediction_type'] == 'both'):
                for bi in range(0, cause_loc.size(0)):
                    stats['cause-iou-set'].append(iouloc(cause_loc[bi,:],cause_reg[bi,:]))

            if(p['prediction_type'] == 'effect' or p['prediction_type'] == 'both'):
                for bi in range(0, effect_loc.size(0)):
                    stats['effect-iou-set'].append(iouloc(effect_loc[bi,],effect_reg[bi,]))

        add_loss('loss', loss, losses)
        if('Segmentation' in p['architecture_type']):
            add_loss('w1_cnt', loss1, losses)
            add_loss('w1_mse', loss2, losses)
        elif('SST' in p['architecture_type']):
            # add_loss('w_cause', loss_c, losses)
            # add_loss('w_effect', loss_e, losses)
            add_loss('w_all', loss, losses)
        else:
            if(p['prediction_type'] == 'both'):
                add_loss('w1_c', loss1_c, losses)
                add_loss('w1_e', loss1_e, losses)
                if(p['use_calibration_loss']):
                    add_loss('w3_c', loss3_cause, losses)
                    add_loss('w3_e', loss3_effect, losses)
            else:
                add_loss('w1', loss1, losses)

    return stats, losses        


def update_epoch_stats(p, split, epoch, writer, stats, stats_epoch, loss_train):
    # update train stats
    write_loss(loss_train, epoch, split, writer)
    if(p['prediction_type'] == 'cause' or p['prediction_type'] == 'both'):
        cause_iou_thr = compute_temporalIoU(stats_epoch['cause-iou-set'])
        cause_iou_mean = float(torch.mean(cause_iou_thr[4:]))
        writer.add_scalar('IoU-cause/%s0.5-0.9'%split, cause_iou_mean, epoch)
        writer.add_scalar('IoU-cause/%s0.5'%split, float(cause_iou_thr[4]), epoch)

        stats['cause-iou-thr-%s' % split] = cause_iou_thr
        stats['cause-iou-mean-%s' % split] = cause_iou_mean

    if(p['prediction_type'] == 'effect' or p['prediction_type'] == 'both'):
        effect_iou_thr = compute_temporalIoU(stats_epoch['effect-iou-set'])
        effect_iou_mean = float(torch.mean(effect_iou_thr[4:]))
        writer.add_scalar('IoU-effect/%s0.5-0.9'%split, float(torch.mean(effect_iou_thr[4:])), epoch)
        writer.add_scalar('IoU-effect/%s0.5'%split, float(effect_iou_thr[4]), epoch)

        stats['effect-iou-thr-%s' % split] = effect_iou_thr
        stats['effect-iou-mean-%s' % split] = effect_iou_mean

    if(p['prediction_type'] == 'both'):
        writer.add_scalar('IoU-both/%s0.5-0.9'%split, (cause_iou_mean + effect_iou_mean) / 2, epoch)
        writer.add_scalar('IoU-both/%s0.5'%split, float((cause_iou_thr[4]+effect_iou_thr[4])/2), epoch)

    if(p['prediction_type'] == 'cause'):
        return cause_iou_mean, stats
    elif(p['prediction_type'] == 'effect'):
        return effect_iou_mean, stats
    elif(p['prediction_type'] == 'both'):
        return (cause_iou_mean + effect_iou_mean) / 2, stats    


def infer_top1(logits, ious, locs):
    # logits: prediction (B*L*S, Class)
    # gt: ground truth (Batch, Len, Scales, Class) - cause start/end effect start/end
    # print('compute topk', logits.size(), ious.size())

    B, C, L, S = ious.size()

    # locs : [2, 208, 64]

    if(C == 3): # bg, cause, effect
        logits = logits.view(B,L*S,C)
        max_val, max_idx = torch.max(logits, dim=2)
    else:
        assert(False)
    
    def get_loc_from_top1(val, idx, _cls, locs):
        #pdb.set_trace()
        cls_val = val * (idx == _cls).float()
        lin_idx = torch.argmax(cls_val,dim=1)
        #print('get_iou_from_top1', lin_idx, _cls)
        res_loc = []
        for bi, idx in enumerate(lin_idx):
            xs = locs[0].view(-1)
            ys = locs[1].view(-1)
            #print('class-%d batch-%d' % (idx, bi), _ious[lin_idx])
            res_loc.append((float(xs[lin_idx[bi]]), float(ys[lin_idx[bi]])))
        return res_loc

    top1_loc_cause = get_loc_from_top1(max_val, max_idx, 1, locs)
    top1_loc_effect= get_loc_from_top1(max_val, max_idx, 2, locs)
    
    return top1_loc_cause, top1_loc_effect

def infer_epoch(p, dataloader, model, boxes):   
    model.eval()

    preds = dict()
    preds['cause-loc-set'] = []
    preds['effect-loc-set'] = []

    num_samples = 0
    for i_batch, v in enumerate(dataloader):

        if('SST' in p['architecture_type']):
            (rgb, flow, cause_reg, effect_reg, labels, ious) = v
            ious = ious.cuda(p['device'])
            labels = labels.cuda(p['device'])
        else:
            assert(False)

        cause_reg = cause_reg.cuda(p['device'])
        effect_reg = effect_reg.cuda(p['device'])

        # data to gpu
        rgb = rgb.cuda(p['device'])
        flow = flow.cuda(p['device'])

        # forward
        if('ProbLocalization' in p['architecture_type']):
            if(p['prediction_type'] == 'cause'):
                prob = model(rgb, flow)
                loss1, cause_loc, cause_check = softmax_loc_loss(p, prob, cause_reg)
                loss = loss1

            elif(p['prediction_type'] == 'effect'):
                prob = model(rgb, flow)
                loss1, effect_loc, effect_check = softmax_loc_loss(p, prob, effect_reg)
                loss = loss1

            else:
                logits = model(rgb, flow)
                if(p['loss_type'] == 'crossentropy'):
                    loss1_c, loss1_e, cause_loc, effect_loc, cause_check, effect_check = \
                            softmax_both_loc_loss(p, logits, cause_reg, effect_reg)
                    loss = p['w1'] * loss1_c + p['w2'] * loss1_e

        elif('MultiLabel' in p['architecture_type']):
            if 'MultiStage' in p['architecture_type']:
                loss1 = 0
                loss2 = 0
                _logits = model(rgb, flow)
                for logits in _logits:
                    loss1 += F.cross_entropy(logits, causality_mask, reduction='mean')
                    loss2 += torch.mean(torch.clamp(F.mse_loss(F.log_softmax(logits[:, :, 1:], dim=1), F.log_softmax(logits.detach()[:, :, :-1], dim=1),reduction="none"), min=0, max=p['mse_tau']*p['mse_tau']))
            else:
                logits = model(rgb, flow)
                loss1 = F.cross_entropy(logits, causality_mask, reduction='mean')
                loss2 = torch.mean(torch.clamp(F.mse_loss(F.log_softmax(logits[:, :, 1:], dim=1), F.log_softmax(logits.detach()[:, :, :-1], dim=1),reduction="none"), min=0, max=p['mse_tau']*p['mse_tau']))

            loss = loss1 * p['w1'] + loss2 * p['w2']

        elif('SST' in p['architecture_type']):
            #print(">>> forward in SST")

            if('both' in p['feature']):
                inputs = torch.cat([rgb, flow], dim=2)
            elif('rgb' in p['feature']):
                inputs = rgb
            logits = model(inputs)

            
            # print('ssd forward', logits.size(), labels.size())
            if('MSTCN' in p['architecture_type']):
                loss = 0
                for _logit in logits:
                    loss += p['criterion'](_logit, labels.view(-1))
            else:
                loss = p['criterion'](logits, labels.view(-1))
            # print("labels: {}".format(labels.size()))
            # print("logits: {}".format(logits.size()))
            # print("loss: {}".format(loss))
            # pdb.set_trace()

            # set_loss_weights(p['criterion_cause'], cause_label)
            # set_loss_weights(p['criterion_effect'], effect_label)

            # loss_c = p['criterion_cause'](logits_cause, cause_label)
            # loss_e = p['criterion_effect'](logits_effect, effect_label)

            # loss = loss_c + loss_e

        # accumulate tIoU
        if('SST' in p['architecture_type']):
            if('MSTCN' in p['architecture_type']):
                logits = logits[-1] # take the prediction from the last stage
            if('SST' in p['architecture_type']):
                cause_loc, effect_loc = infer_top1(logits, ious, boxes)

            preds['cause-loc-set'] = preds['cause-loc-set'] + cause_loc
            preds['effect-loc-set'] = preds['effect-loc-set'] + effect_loc

    return preds        