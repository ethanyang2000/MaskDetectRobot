import numpy as np
import torch
import cv2
from timeit import default_timer as timer
import torch.nn as nn
import torch.nn.functional as F
import math
import serial

class Args():
    def __init__(self):
        self.device = 'cpu'
        self.model_path = '/home/pi/Desktop/my_model.pkl'
        self.feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
        self.anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        self.anchor_ratios = [[1, 0.62, 0.42]] * 5
        self.img_path = '/home/pi/Desktop/test2.jpg'    

args = Args()

def decode_bbox(anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
    
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
    return predict_bbox

def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):

    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2* num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0] # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0] # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.expand_dims(np.concatenate(anchor_bboxes, axis=0), axis=0)
    return anchor_bboxes


def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.7, iou_thresh=0.5, keep_top_k=-1):

    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    return conf_keep_idx[pick]

class KitModel(nn.Module):

    def __init__(self):
        super(KitModel, self).__init__()

        self.conv2d_0 = self.__conv(2, name='conv2d_0', in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_0_bn = self.__batch_normalization(2, 'conv2d_0_bn', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_1 = self.__conv(2, name='conv2d_1', in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_1_bn = self.__batch_normalization(2, 'conv2d_1_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_2_bn = self.__batch_normalization(2, 'conv2d_2_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_3 = self.__conv(2, name='conv2d_3', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_3_bn = self.__batch_normalization(2, 'conv2d_3_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.cls_0_insert_conv2d = self.__conv(2, name='cls_0_insert_conv2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.loc_0_insert_conv2d = self.__conv(2, name='loc_0_insert_conv2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_4 = self.__conv(2, name='conv2d_4', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.cls_0_insert_conv2d_bn = self.__batch_normalization(2, 'cls_0_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.loc_0_insert_conv2d_bn = self.__batch_normalization(2, 'loc_0_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_4_bn = self.__batch_normalization(2, 'conv2d_4_bn', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.cls_0_conv = self.__conv(2, name='cls_0_conv', in_channels=64, out_channels=8, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loc_0_conv = self.__conv(2, name='loc_0_conv', in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.cls_1_insert_conv2d = self.__conv(2, name='cls_1_insert_conv2d', in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.loc_1_insert_conv2d = self.__conv(2, name='loc_1_insert_conv2d', in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_5 = self.__conv(2, name='conv2d_5', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.cls_1_insert_conv2d_bn = self.__batch_normalization(2, 'cls_1_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.loc_1_insert_conv2d_bn = self.__batch_normalization(2, 'loc_1_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_5_bn = self.__batch_normalization(2, 'conv2d_5_bn', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.cls_1_conv = self.__conv(2, name='cls_1_conv', in_channels=64, out_channels=8, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loc_1_conv = self.__conv(2, name='loc_1_conv', in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.cls_2_insert_conv2d = self.__conv(2, name='cls_2_insert_conv2d', in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.loc_2_insert_conv2d = self.__conv(2, name='loc_2_insert_conv2d', in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_6 = self.__conv(2, name='conv2d_6', in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.cls_2_insert_conv2d_bn = self.__batch_normalization(2, 'cls_2_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.loc_2_insert_conv2d_bn = self.__batch_normalization(2, 'loc_2_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_6_bn = self.__batch_normalization(2, 'conv2d_6_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.cls_2_conv = self.__conv(2, name='cls_2_conv', in_channels=64, out_channels=8, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loc_2_conv = self.__conv(2, name='loc_2_conv', in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2d_7 = self.__conv(2, name='conv2d_7', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.cls_3_insert_conv2d = self.__conv(2, name='cls_3_insert_conv2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.loc_3_insert_conv2d = self.__conv(2, name='loc_3_insert_conv2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2d_7_bn = self.__batch_normalization(2, 'conv2d_7_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.cls_3_insert_conv2d_bn = self.__batch_normalization(2, 'cls_3_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.loc_3_insert_conv2d_bn = self.__batch_normalization(2, 'loc_3_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.cls_4_insert_conv2d = self.__conv(2, name='cls_4_insert_conv2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.loc_4_insert_conv2d = self.__conv(2, name='loc_4_insert_conv2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.cls_3_conv = self.__conv(2, name='cls_3_conv', in_channels=64, out_channels=8, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loc_3_conv = self.__conv(2, name='loc_3_conv', in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.cls_4_insert_conv2d_bn = self.__batch_normalization(2, 'cls_4_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.loc_4_insert_conv2d_bn = self.__batch_normalization(2, 'loc_4_insert_conv2d_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.cls_4_conv = self.__conv(2, name='cls_4_conv', in_channels=64, out_channels=8, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.loc_4_conv = self.__conv(2, name='loc_4_conv', in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        conv2d_0_pad    = F.pad(x, (1, 1, 1, 1))
        conv2d_0        = self.conv2d_0(conv2d_0_pad)
        conv2d_0_bn     = self.conv2d_0_bn(conv2d_0)
        conv2d_0_activation = F.relu(conv2d_0_bn)
        maxpool2d_0     = F.max_pool2d(conv2d_0_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2d_1_pad    = F.pad(maxpool2d_0, (1, 1, 1, 1))
        conv2d_1        = self.conv2d_1(conv2d_1_pad)
        conv2d_1_bn     = self.conv2d_1_bn(conv2d_1)
        conv2d_1_activation = F.relu(conv2d_1_bn)
        maxpool2d_1     = F.max_pool2d(conv2d_1_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2d_2_pad    = F.pad(maxpool2d_1, (1, 1, 1, 1))
        conv2d_2        = self.conv2d_2(conv2d_2_pad)
        conv2d_2_bn     = self.conv2d_2_bn(conv2d_2)
        conv2d_2_activation = F.relu(conv2d_2_bn)
        #maxpool2d_2_pad = F.pad(conv2d_2_activation, (0, 1, 0, 1), value=float('-inf'))
        maxpool2d_2     = F.max_pool2d(conv2d_2_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2d_3_pad    = F.pad(maxpool2d_2, (1, 1, 1, 1))
        conv2d_3        = self.conv2d_3(conv2d_3_pad)
        conv2d_3_bn     = self.conv2d_3_bn(conv2d_3)
        conv2d_3_activation = F.relu(conv2d_3_bn)
        maxpool2d_3_pad = F.pad(conv2d_3_activation, (0, 1, 0, 1), value=float('-inf'))
        maxpool2d_3     = F.max_pool2d(maxpool2d_3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        cls_0_insert_conv2d_pad = F.pad(conv2d_3_activation, (1, 1, 1, 1))
        cls_0_insert_conv2d = self.cls_0_insert_conv2d(cls_0_insert_conv2d_pad)
        loc_0_insert_conv2d_pad = F.pad(conv2d_3_activation, (1, 1, 1, 1))
        loc_0_insert_conv2d = self.loc_0_insert_conv2d(loc_0_insert_conv2d_pad)
        conv2d_4_pad    = F.pad(maxpool2d_3, (1, 1, 1, 1))
        conv2d_4        = self.conv2d_4(conv2d_4_pad)
        cls_0_insert_conv2d_bn = self.cls_0_insert_conv2d_bn(cls_0_insert_conv2d)
        loc_0_insert_conv2d_bn = self.loc_0_insert_conv2d_bn(loc_0_insert_conv2d)
        conv2d_4_bn     = self.conv2d_4_bn(conv2d_4)
        cls_0_insert_conv2d_activation = F.relu(cls_0_insert_conv2d_bn)
        loc_0_insert_conv2d_activation = F.relu(loc_0_insert_conv2d_bn)
        conv2d_4_activation = F.relu(conv2d_4_bn)
        cls_0_conv_pad  = F.pad(cls_0_insert_conv2d_activation, (1, 1, 1, 1))
        cls_0_conv      = self.cls_0_conv(cls_0_conv_pad)
        loc_0_conv_pad  = F.pad(loc_0_insert_conv2d_activation, (1, 1, 1, 1))
        loc_0_conv      = self.loc_0_conv(loc_0_conv_pad)
        maxpool2d_4_pad = F.pad(conv2d_4_activation, (0, 1, 0, 1), value=float('-inf'))
        maxpool2d_4     = F.max_pool2d(maxpool2d_4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        cls_1_insert_conv2d_pad = F.pad(conv2d_4_activation, (1, 1, 1, 1))
        cls_1_insert_conv2d = self.cls_1_insert_conv2d(cls_1_insert_conv2d_pad)
        loc_1_insert_conv2d_pad = F.pad(conv2d_4_activation, (1, 1, 1, 1))
        loc_1_insert_conv2d = self.loc_1_insert_conv2d(loc_1_insert_conv2d_pad)
        cls_0_reshape   = torch.reshape(input = cls_0_conv.permute(0,2,3,1) , shape = (cls_0_conv.size(0),-1,2))
        loc_0_reshape   = torch.reshape(input = loc_0_conv.permute(0,2,3,1) , shape = (loc_0_conv.size(0),-1,4))
        conv2d_5_pad    = F.pad(maxpool2d_4, (1, 1, 1, 1))
        conv2d_5        = self.conv2d_5(conv2d_5_pad)
        cls_1_insert_conv2d_bn = self.cls_1_insert_conv2d_bn(cls_1_insert_conv2d)
        loc_1_insert_conv2d_bn = self.loc_1_insert_conv2d_bn(loc_1_insert_conv2d)
        cls_0_activation = torch.sigmoid(cls_0_reshape)
        conv2d_5_bn     = self.conv2d_5_bn(conv2d_5)
        cls_1_insert_conv2d_activation = F.relu(cls_1_insert_conv2d_bn)
        loc_1_insert_conv2d_activation = F.relu(loc_1_insert_conv2d_bn)
        conv2d_5_activation = F.relu(conv2d_5_bn)
        cls_1_conv_pad  = F.pad(cls_1_insert_conv2d_activation, (1, 1, 1, 1))
        cls_1_conv      = self.cls_1_conv(cls_1_conv_pad)
        loc_1_conv_pad  = F.pad(loc_1_insert_conv2d_activation, (1, 1, 1, 1))
        loc_1_conv      = self.loc_1_conv(loc_1_conv_pad)
        maxpool2d_5_pad = F.pad(conv2d_5_activation, (0, 1, 0, 1), value=float('-inf'))
        maxpool2d_5     = F.max_pool2d(maxpool2d_5_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        cls_2_insert_conv2d_pad = F.pad(conv2d_5_activation, (1, 1, 1, 1))
        cls_2_insert_conv2d = self.cls_2_insert_conv2d(cls_2_insert_conv2d_pad)
        loc_2_insert_conv2d_pad = F.pad(conv2d_5_activation, (1, 1, 1, 1))
        loc_2_insert_conv2d = self.loc_2_insert_conv2d(loc_2_insert_conv2d_pad)
        cls_1_reshape   = torch.reshape(input = cls_1_conv.permute(0,2,3,1) , shape = (cls_1_conv.size(0),-1,2))
        loc_1_reshape   = torch.reshape(input = loc_1_conv.permute(0,2,3,1) , shape = (loc_1_conv.size(0),-1,4))
        conv2d_6_pad    = F.pad(maxpool2d_5, (1, 1, 1, 1))
        conv2d_6        = self.conv2d_6(conv2d_6_pad)
        cls_2_insert_conv2d_bn = self.cls_2_insert_conv2d_bn(cls_2_insert_conv2d)
        loc_2_insert_conv2d_bn = self.loc_2_insert_conv2d_bn(loc_2_insert_conv2d)
        cls_1_activation = torch.sigmoid(cls_1_reshape)
        conv2d_6_bn     = self.conv2d_6_bn(conv2d_6)
        cls_2_insert_conv2d_activation = F.relu(cls_2_insert_conv2d_bn)
        loc_2_insert_conv2d_activation = F.relu(loc_2_insert_conv2d_bn)
        conv2d_6_activation = F.relu(conv2d_6_bn)
        cls_2_conv_pad  = F.pad(cls_2_insert_conv2d_activation, (1, 1, 1, 1))
        cls_2_conv      = self.cls_2_conv(cls_2_conv_pad)
        loc_2_conv_pad  = F.pad(loc_2_insert_conv2d_activation, (1, 1, 1, 1))
        loc_2_conv      = self.loc_2_conv(loc_2_conv_pad)
        conv2d_7        = self.conv2d_7(conv2d_6_activation)
        cls_3_insert_conv2d_pad = F.pad(conv2d_6_activation, (1, 1, 1, 1))
        cls_3_insert_conv2d = self.cls_3_insert_conv2d(cls_3_insert_conv2d_pad)
        loc_3_insert_conv2d_pad = F.pad(conv2d_6_activation, (1, 1, 1, 1))
        loc_3_insert_conv2d = self.loc_3_insert_conv2d(loc_3_insert_conv2d_pad)
        cls_2_reshape   = torch.reshape(input = cls_2_conv.permute(0,2,3,1) , shape = (cls_2_conv.size(0),-1,2))
        loc_2_reshape   = torch.reshape(input = loc_2_conv.permute(0,2,3,1) , shape = (loc_2_conv.size(0),-1,4))
        conv2d_7_bn     = self.conv2d_7_bn(conv2d_7)
        cls_3_insert_conv2d_bn = self.cls_3_insert_conv2d_bn(cls_3_insert_conv2d)
        loc_3_insert_conv2d_bn = self.loc_3_insert_conv2d_bn(loc_3_insert_conv2d)
        cls_2_activation = torch.sigmoid(cls_2_reshape)
        conv2d_7_activation = F.relu(conv2d_7_bn)
        cls_3_insert_conv2d_activation = F.relu(cls_3_insert_conv2d_bn)
        loc_3_insert_conv2d_activation = F.relu(loc_3_insert_conv2d_bn)
        cls_4_insert_conv2d_pad = F.pad(conv2d_7_activation, (1, 1, 1, 1))
        cls_4_insert_conv2d = self.cls_4_insert_conv2d(cls_4_insert_conv2d_pad)
        loc_4_insert_conv2d_pad = F.pad(conv2d_7_activation, (1, 1, 1, 1))
        loc_4_insert_conv2d = self.loc_4_insert_conv2d(loc_4_insert_conv2d_pad)
        cls_3_conv_pad  = F.pad(cls_3_insert_conv2d_activation, (1, 1, 1, 1))
        cls_3_conv      = self.cls_3_conv(cls_3_conv_pad)
        loc_3_conv_pad  = F.pad(loc_3_insert_conv2d_activation, (1, 1, 1, 1))
        loc_3_conv      = self.loc_3_conv(loc_3_conv_pad)
        cls_4_insert_conv2d_bn = self.cls_4_insert_conv2d_bn(cls_4_insert_conv2d)
        loc_4_insert_conv2d_bn = self.loc_4_insert_conv2d_bn(loc_4_insert_conv2d)
        cls_3_reshape   = torch.reshape(input = cls_3_conv.permute(0,2,3,1) , shape = (cls_3_conv.size(0),-1,2))
        loc_3_reshape   = torch.reshape(input = loc_3_conv.permute(0,2,3,1) , shape = (loc_3_conv.size(0),-1,4))
        cls_4_insert_conv2d_activation = F.relu(cls_4_insert_conv2d_bn)
        loc_4_insert_conv2d_activation = F.relu(loc_4_insert_conv2d_bn)
        cls_3_activation = torch.sigmoid(cls_3_reshape)
        cls_4_conv_pad  = F.pad(cls_4_insert_conv2d_activation, (1, 1, 1, 1))
        cls_4_conv      = self.cls_4_conv(cls_4_conv_pad)
        loc_4_conv_pad  = F.pad(loc_4_insert_conv2d_activation, (1, 1, 1, 1))
        loc_4_conv      = self.loc_4_conv(loc_4_conv_pad)
        cls_4_reshape   = torch.reshape(input = cls_4_conv.permute(0,2,3,1) , shape = (cls_4_conv.size(0),-1,2))
        loc_4_reshape   = torch.reshape(input = loc_4_conv.permute(0,2,3,1) , shape = (loc_4_conv.size(0),-1,4))
        cls_4_activation = torch.sigmoid(cls_4_reshape)
        loc_branch_concat = torch.cat((loc_0_reshape, loc_1_reshape, loc_2_reshape, loc_3_reshape, loc_4_reshape), 1)
        cls_branch_concat = torch.cat((cls_0_activation, cls_1_activation, cls_2_activation, cls_3_activation, cls_4_activation), 1)
        return loc_branch_concat, cls_branch_concat


    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()


        return layer

def test(image,
        conf_thresh=0.5,
        iou_thresh=0.4,
        target_shape=(360, 360),
        anchors=None,
        model=None,
        draw_result=True,
        show_result=True,
        ):

    output_info = []
    
    id2class = {0: 'Mask', 1: 'NoMask'}
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape) / 255.0
    image_exp = np.expand_dims(image_resized, axis=0) #batch_size = 1

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    img_tensor = torch.tensor(image_transposed).float().to(torch.device(args.device))

    y_bboxes, y_scores, = model.forward(img_tensor)
    y_bboxes_output = y_bboxes.detach().cpu().numpy()
    y_cls_output = y_scores.detach().cpu().numpy()

    y_bboxes = decode_bbox(anchors, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
    print("process end at :"+str(timer()))
    return output_info

def model_load(model_path):
    model = KitModel()
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device(args.device))
    return model



class Detector():
    def __init__(self, model_dir, img_dir):
        self.model_dir = model_dir
        self.img_dir = img_dir
        self.model = model_load(self.model_dir)
        self.anchors = generate_anchors(args.feature_map_sizes, args.anchor_sizes, args.anchor_ratios)
    
    def checkMask(self):
        tic = timer()
        img = cv2.imread(self.img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ans = test(img, show_result=True, target_shape=(360, 360), anchors=self.anchors, model=self.model)
        toc = timer()
        print(toc - tic)
        print(ans)
        return ans

model = model_load(args.model_path)
anchors = generate_anchors(args.feature_map_sizes, args.anchor_sizes, args.anchor_ratios)


#if __name__ == "main":       
detec = Detector(args.model_path, args.img_path)
#while(1):
#tic = timer()
while(1):
    detec.checkMask()
#toc = timer()
#print(toc - tic)

"""import multiprocessing
pool = multiprocessing.Pool(processes = 2)
tic = timer()
#while(1):
img = cv2.imread(args.img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pool.apply_async(test,(img, True, (360, 360), anchors, model))
pool.close()
pool.join()
#pool.apply_async(test,(img, True, (360, 360), anchors, model))
#test(img, show_result=True, target_shape=(360, 360), anchors=anchors, model=model)
toc = timer()
print(toc - tic)"""



