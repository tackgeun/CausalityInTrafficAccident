import argparse, pickle, os, math, random, sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# The entire size = 1896
# # train 70% validation 15% test 15%
# # (0 ~ 1326) (1326 ~ 1611) (1611 ~ 1896)
# parser.add_argument('--dataset_ver', type=str, default='Mar9th')
# parser.add_argument('--train_start', type=int, default=0)
# parser.add_argument('--train_end', type=int, default=1355)
# parser.add_argument('--val_start', type=int, default=1355)
# parser.add_argument('--val_end', type=int, default=1355+290)
# parser.add_argument('--test_start', type=int, default=1355+290)
# parser.add_argument('--test_end', type=int, default=1355+290+290)

# parser.add_argument('--use_randperm', type=int, default=7802)

# parser.add_argument('--use_flip', type=bool, default=True)

# parser.add_argument('--num_causes', type=int, default=18)
# parser.add_argument('--num_effects', type=int, default=7)

# if(args.dataset_ver == 'Nov3th' or args.dataset_ver == 'Mar9th'):
#     args.train_start = 0
#     args.train_end   = 1355
#     args.val_start   = args.train_end
#     args.val_end     = args.train_end + 290
#     args.test_start  = args.val_end
#     args.test_end    = args.val_end + 290

class CausalityInTrafficAccident(Dataset):
    """Causality In Traffic Accident Dataset."""
    
    def __init__(self, p, split, test_mode=False):
        DATA_ROOT = '../causal-sst/dataset/'
        self.feature = p['feature']
        self.split = split
        if split == 'train':
            data_length = (0, 1355)
        elif split == 'val':
            data_length = (1355, 1355 + 290)
        elif split == 'test':
            data_length = (1355 + 290, 1355 + 290 + 290)
        p['use_randperm'] = 7802

        self.feed_type = p['feed_type']

        self.use_flip = True

        self.seq_length = 208
        self.fps = 25
        self.vid_length = self.seq_length * 8 / self.fps
        
        if(self.feed_type == 'classification'):
            self.num_segments = p["num_segments"] # default 3
            self.new_length = p['new_length']
            self.num_causes = 18
            self.num_effects = 7

        self.test_mode = test_mode
        self.random_shift = False

        if('both' in self.feature):
            self.use_flow = True
        elif('rgb' in self.feature):
            self.use_flow = False

        self.stride = 'x8'
        self.root_dir = DATA_ROOT + 'i3d-feature-fps25-shot/%s/' % self.stride
        
        dv = 'Mar9th'
        self.anno_dir = DATA_ROOT + 'annotation-%s-25fps.pkl' % dv
        
        with open(self.anno_dir, 'rb') as f:
            self.annos = pickle.load(f)
           
            feat_rgb = torch.load(DATA_ROOT + 'i3d-rgb-fps25-%s.pt' % dv)
            if(self.use_flow):
                feat_flow= torch.load(DATA_ROOT + 'i3d-flow-fps25-%s.pt' % dv)

            if(self.use_flip):
                feat_rgb_flip = torch.load(DATA_ROOT + 'i3d-rgb-flip-fps25-%s.pt' % dv)
                if(self.use_flow):
                    feat_flow_flip = torch.load(DATA_ROOT + 'i3d-flow-flip-fps25-%s.pt' % dv)

            start_idx = data_length[0]
            end_idx = data_length[1]

            if(p['use_randperm'] > 0):
                torch.manual_seed(p['use_randperm'])
                indices = torch.randperm(len(self.annos))
                L = indices.numpy().tolist()
                if(dv == 'Nov3rd' or dv == 'Nov3th'):
                    feat_rgb = feat_rgb[indices, :]
                    if(self.use_flow):
                        feat_flow = feat_flow[indices, :]
                    
                    if(self.use_flip):
                        feat_rgb_flip = feat_rgb_flip[indices, :]
                        if(self.use_flow):
                            feat_flow_flip = feat_flow_flip[indices, :]

                elif(dv == 'Mar9th'):
                    indices = indices.tolist()
                    remap = lambda I,arr: [arr[i] for i in I]
                    feat_rgb = remap(indices, feat_rgb)
                    if(self.use_flow):
                        feat_flow = remap(indices, feat_flow)
                    
                    if(self.use_flip):
                        feat_rgb_flip = remap(indices, feat_rgb_flip)
                        if(self.use_flow):
                            feat_flow_flip = remap(indices, feat_flow_flip)
                else:
                    assert(False)
                
                self.annos = [self.annos[L[l]] for l in range(0, len(self.annos))]

            self.annos = self.annos[start_idx:end_idx]
            self.feat_rgb = feat_rgb[start_idx:end_idx]
            if(self.use_flow):
                self.feat_flow = feat_flow[start_idx:end_idx]
            
            if(self.use_flip):
                self.feat_rgb_flip = feat_rgb_flip[start_idx:end_idx]
                if(self.use_flow):
                    self.feat_flow_flip = feat_flow_flip[start_idx:end_idx]

            # self.feat_rgb = feat_rgb[start_idx:end_idx, :, :]
            # if(self.use_flow):
            #     self.feat_flow = feat_flow[start_idx:end_idx, :, :]
            
            # if(self.use_flip):
            #     self.feat_rgb_flip = feat_rgb_flip[start_idx:end_idx, :, :]
            #     if(self.use_flow):
            #         self.feat_flow_flip = feat_flow_flip[start_idx:end_idx, :, :]

        if(self.feed_type == 'detection' or self.feed_type == 'proposal'):
            self.positive_thres = p['positive_thres']
            scales = torch.Tensor((p['proposal_scales'])).unsqueeze(0).unsqueeze(1) # 1 x scale x 1
            scales = scales / self.seq_length * self.vid_length
            
            boxes = torch.Tensor([j for j in range(0, self.seq_length)]).unsqueeze(0).unsqueeze(2)
            boxes = boxes / self.seq_length * self.vid_length
            boxes = boxes.repeat(2, 1, len(p['proposal_scales'])) # start/end, num_scales, temporal_length

            #print('ssd size', scales.size(), boxes.size())

            boxes[0, :, :] = boxes[0, :, :] - scales/2 # start time
            boxes[1, :, :] = boxes[1, :, :] + scales/2 # end time

            self.boxes = boxes.cuda(p['device'])

            iou_bg = torch.ones(self.boxes.size(1), self.boxes.size(2)) * self.positive_thres
            self.iou_bg = iou_bg.cuda(p['device'])


    def __len__(self):
            return len(self.annos)
        
    def compute_ious(self, boxes, gt):
        t1 = self.boxes[0, :, :]
        t2 = self.boxes[1, :, :]

        inter_t1 = torch.clamp(t1, min=gt[0]) # torch.cmax(t1, gt[0])
        inter_t2 = torch.clamp(t2, max=gt[1]) # torch.cmin(t2, gt[1])

        union_t1 = torch.clamp(t1, max=gt[0])
        union_t2 = torch.clamp(t2, min=gt[1])

        _inter = F.relu(inter_t2 - inter_t1)
        _union = F.relu(union_t2 - union_t1) + 1e-5

        return _inter / _union

    def __getitem__(self, idx):
        if self.feed_type == 'detection':
            return self.feed_detections(idx)
        elif self.feed_type == 'proposal':
            return self.feed_proposals(idx)
        elif self.feed_type == 'classification':
            return self.feed_classification(idx)

    def get_feature(self, idx):
        if(self.use_flip and random.random() > 0.5):
            rgb_feat = self.feat_rgb_flip[idx, :, :]
            if(self.use_flow):
                flow_feat = self.feat_flow_flip[idx, :, :]
            else:
                flow_feat = torch.zeros(0)
        else:
            rgb_feat = self.feat_rgb[idx, :, :]
            if(self.use_flow):
                flow_feat = self.feat_flow[idx, :, :]
            else:
                flow_feat = torch.zeros(0)

        return rgb_feat, flow_feat      

    def get_det_labels(self, idx):
        annos = self.annos[idx]
        cause_loc = torch.Tensor([annos[1][1], annos[1][2]])
        effect_loc = torch.Tensor([annos[2][1], annos[2][2]])

        cause_loc = cause_loc / annos[3]
        effect_loc = effect_loc / annos[3]

        iou_cause = self.compute_ious(self.boxes, annos[1][1:3])
        iou_effect = self.compute_ious(self.boxes, annos[2][1:3])

        ious = torch.stack([self.iou_bg, iou_cause, iou_effect], dim=0)
        _, labels = torch.max(ious, dim=0)               

        return cause_loc, effect_loc, ious, labels

    # # Binary IoUs
    # def get_prop_labels(self, idx):
    #     annos = self.annos[idx]
    #     cause_loc = torch.Tensor([annos[1][1], annos[1][2]])
    #     effect_loc = torch.Tensor([annos[2][1], annos[2][2]])

    #     iou_cause = self.compute_ious(self.boxes, annos[1][1:3])
    #     iou_effect = self.compute_ious(self.boxes, annos[2][1:3])

    #     iou_cat = torch.stack([iou_cause, iou_effect], dim=0)
    #     iou_cat, _ = torch.max(iou_cat, dim=0)
    #     ious = torch.stack([self.iou_bg, iou_cat], dim=0)
    #     _, labels = torch.max(ious, dim=0)

    #     return cause_loc, effect_loc, ious, labels

    # [BG, cause, effect] IoU
    def get_prop_labels(self, idx):
        annos = self.annos[idx]
        cause_loc = torch.Tensor([annos[1][1], annos[1][2]])
        effect_loc = torch.Tensor([annos[2][1], annos[2][2]])

        iou_cause = self.compute_ious(self.boxes, annos[1][1:3])
        iou_effect = self.compute_ious(self.boxes, annos[2][1:3])

        ious = torch.stack([self.iou_bg, iou_cause, iou_effect], dim=0)
        _, labels = torch.max(ious, dim=0)               
        labels = (labels > 0.5).long()

        return cause_loc, effect_loc, ious, labels

    # construct labels for SSD detector
    def feed_detections(self, idx):
        rgb_feat, flow_feat = self.get_feature(idx)
        cause_loc, effect_loc, ious, labels = self.get_det_labels(idx)

        return rgb_feat, flow_feat, cause_loc, effect_loc, labels, ious

    # construct labels for SST proposal
    def feed_proposals(self, idx):
        rgb_feat, flow_feat = self.get_feature(idx)
        cause_loc, effect_loc, ious, labels = self.get_prop_labels(idx)

        return rgb_feat, flow_feat, cause_loc, effect_loc, labels, ious

    def _sample_indices(self, num_frames):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, num_frames):
        if num_frames > self.num_segments + self.new_length - 1:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, num_frames):

        tick = (num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1


    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def feed_classification(self, idx):
        annos = self.annos[idx]
        
        if(self.use_flip and random.random() > 0.5):
            rgb_feat = self.feat_rgb_flip[idx]
        else:
            rgb_feat = self.feat_rgb[idx]
        
        num_frames = rgb_feat.size(0)

        cause_label = annos[1][3] - 1# - 1 (no background label)
        effect_label = annos[2][3] - self.num_causes - 1  # - 1 (no background label)
        

        if not self.test_mode:
            segment_indices = self._sample_indices(num_frames) if self.random_shift else self._get_val_indices(num_frames)
        else:
            segment_indices = self._get_test_indices(num_frames)

        #return self.get(record, segment_indices)
        segment_indices = segment_indices - 1

        rgb_feat = rgb_feat[segment_indices, :]
        #label = dict()
        #label['cause'] = annos[1][3]
        #label['effect'] = annos[2][3]

        #feat = dict()
        #feat['cause'] = rgb_feat
        #feat['effect'] = flow_feat

        return rgb_feat, cause_label, effect_label
        #return feat, label

