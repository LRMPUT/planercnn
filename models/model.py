"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.profiler import PassThroughProfiler

from disjoint_set import DisjointSet

import utils
import cv2
from models.modules import *
from utils import *
import nms

# from pytorch_memlab import MemReporter


############################################################
#  Pytorch Utility Functions
############################################################

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]

def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


############################################################
#  FPN Graph
############################################################

class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels, bilinear_upsampling=False):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.bilinear_upsampling = bilinear_upsampling
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)
        
        if self.bilinear_upsampling:
            p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2, mode='bilinear')
            p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2, mode='bilinear')
            p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2, mode='bilinear')
        else:
            p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
            p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
            p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
            pass

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        ## P6 is used for the 5th anchor scale in RPN. Generated by
        ## subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        return [p2_out, p3_out, p4_out, p5_out, p6_out]


############################################################
#  Resnet Graph
############################################################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, architecture, stage5=False, numInputChannels=3):
        super(ResNet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.C1 = nn.Sequential(
            nn.Conv2d(numInputChannels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x


    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion, eps=0.001, momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    ## Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    ## Apply deltas
    center_y = center_y + deltas[:, 0] * height
    center_x = center_x + deltas[:, 1] * width
    height = height * torch.exp(deltas[:, 2])
    width = width * torch.exp(deltas[:, 3])
    ## Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result

def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    ## Currently only supports batchsize 1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    ## Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    ## Box deltas [batch, num_rois, 4]
    deltas = inputs[1]
    
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev
    ## Improve performance by trimming to top anchors by score
    ## and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.data, :]
    anchors = anchors[order.data, :]

    ## Apply deltas to anchors to get refined anchors.
    ## [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    ## Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    ## Filter out small boxes
    ## According to Xinlei Chen's paper, this reduces detection accuracy
    ## for small objects, so we're skipping it.

    ## Non-max suppression
    # keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    keep = torchvision.ops.nms(boxes.index_select(1, torch.LongTensor([1, 0, 3, 2]).cuda()), scores, nms_threshold)

    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    
    
    ## Normalize dimensions to range of 0 to 1.
    norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    ## Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes


############################################################
#  ROIAlign Layer
############################################################

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    ## Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    ## Feature Maps. List of feature maps from different level of the
    ## feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    ## Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    ## Equation 1 in the Feature Pyramid Networks paper. Account for
    ## the fact that our coordinates are normalized here.
    ## e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h*w)/(640.0/torch.sqrt(image_area)))
    # roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)

    ## Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix  = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:,0]
        level_boxes = boxes[ix.data, :]

        ## Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        ## Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        ## Crop and Resize
        ## From Mask R-CNN paper: "We sample four regular locations, so
        ## that we can evaluate either max or average pooling. In fact,
        ## interpolating only a single value at each bin center (without
        ## pooling) is nearly as effective."
        #
        ## Here we use the simplified approach of a single value per bin,
        ## which is how it's done in tf.crop_and_resize()
        ## Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]),requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)  #CropAndResizeFunction needs batch dimension
        # pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled_features = roi_align(feature_maps[i], [level_boxes], (pool_size, pool_size))
        pooled.append(pooled_features)

    ## Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)
    ## Pack box_to_level mapping into one array and add another
    ## column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    ## Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled


def coordinates_roi(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    ## Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    ## Feature Maps. List of feature maps from different level of the
    ## feature pyramid. Each is [batch, height, width, channels]
    cooridnates = inputs[1]

    ## Stop gradient propogation to ROI proposals
    boxes = boxes.detach()

    cooridnates = cooridnates.unsqueeze(0)  ## CropAndResizeFunction needs batch dimension
    # pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(cooridnates, boxes, ind)
    pooled_features = roi_align(cooridnates, [boxes], (pool_size, pool_size))

    return pooled_features


def disp_roi_align(inputs, pool_size):
    """Implements ROI Align with one sampling point

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    ## Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    ## Feature Maps. List of feature maps from different level of the
    ## feature pyramid. Each is [batch, height, width, channels]
    disp = inputs[1]

    ## Stop gradient propogation to ROI proposals
    boxes = boxes.detach()

    # needs batch dimension
    disp = disp.unsqueeze(0)
    # pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(cooridnates, boxes, ind)
    pooled_features = roi_align(disp, [boxes], (pool_size, pool_size), sampling_ratio=1)

    return pooled_features


############################################################
##  Detection Target Layer
############################################################
def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    ## 1. Tile boxes2 and repeate boxes1. This allows us to compare
    ## every boxes1 against every boxes2 without loops.
    ## TF doesn't have an equivalent to np.repeate() so simulate it
    ## using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    ## 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    ## 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    ## 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps

def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, gt_parameters, disp, config, camera):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    ## Currently only supports batchsize 1
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)
    gt_parameters = gt_parameters.squeeze(0)
    # no_crowd_bool =  Variable(torch.ByteTensor(proposals.size()[0]*[True]), requires_grad=False)
    # if config.GPU_COUNT:
    #     no_crowd_bool = no_crowd_bool.cuda()

    ## Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    ## Determine postive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    ## 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5
    #print('positive count', positive_roi_bool.sum())

    ## Subsample ROIs. Aim for 33% positive
    ## Positive ROIs
    if positive_roi_bool.sum() > 0:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.data,:]

        ## Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data,:]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data,:]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]
        roi_gt_parameters = gt_parameters[roi_gt_box_assignment.data]
        
        ## Compute bbox refinement for positive ROIs
        deltas = Variable(utils.box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
        std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), requires_grad=False)
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev

        ## Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data]

        ## Compute mask targets
        # y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
        # y1 /= disp.shape[2]
        # x1 /= disp.shape[3]
        # y2 /= disp.shape[2]
        # x2 /= disp.shape[3]
        # boxes_disp = torch.cat([y1, x1, y2, x2], dim=1)

        boxes = positive_rois
        if config.USE_MINI_MASK:
            ## Transform ROI corrdinates from normalized image space
            ## to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        # box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
        # if config.GPU_COUNT:
        #     box_ids = box_ids.cuda()

        if config.NUM_PARAMETER_CHANNELS > 0:
            # masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks[:, :, :, 0].contiguous().unsqueeze(1), boxes, box_ids).data, requires_grad=False).squeeze(1)
            masks = Variable(roi_align(roi_masks[:, :, :, 0].contiguous().unsqueeze(1),
                                       boxes.chunk(boxes.shape[0], dim=0),
                                       (config.MASK_SHAPE[0], config.MASK_SHAPE[1])).data,
                             requires_grad=False).squeeze(1)
            masks = torch.round(masks)
            # parameters = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks[:, :, :, 1].contiguous().unsqueeze(1), boxes, box_ids).data, requires_grad=False).squeeze(1)
            parameters = Variable(roi_align(roi_masks[:, :, :, 1].contiguous().unsqueeze(1),
                                       boxes.chunk(boxes.shape[0], dim=0),
                                       (config.MASK_SHAPE[0], config.MASK_SHAPE[1])).data,
                             requires_grad=False).squeeze(1)
            masks = torch.stack([masks, parameters], dim=-1)
        else:
            # masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False).squeeze(1)
            masks = Variable(roi_align(roi_masks.unsqueeze(1),
                                       boxes.chunk(boxes.shape[0], dim=0),
                                       (config.MASK_SHAPE[0], config.MASK_SHAPE[1])).data,
                             requires_grad=False).squeeze(1)
            masks = torch.round(masks)

            fx = camera[0]
            fy = camera[1]
            cx = camera[2]
            cy = camera[3]
            w = camera[4]
            h = camera[5]

            # ranges = config.getRangesFull(camera).transpose(1, 2).transpose(0, 1)
            # ranges_rois = roi_align(ranges.view(1, 3, ranges.shape[1], ranges.shape[2]),
            #                         [positive_rois],
            #                         (config.SUPPORT_SHAPE[0], config.SUPPORT_SHAPE[1]),
            #                         sampling_ratio=1)
            # ranges_rois_comp = ranges_rois.view(-1, 3, config.SUPPORT_SHAPE[0] *config.SUPPORT_SHAPE[1])
            ranges_rois = get_support_ranges(camera, positive_rois)
            # top left (0, 0), top right (640, 0)
            # bottom left (0, 480), bottom right (640, 480)
            # shape (1, 3, 2, 2)
            # ranges = torch.tensor([[[[(0.0 - cx)/fx, (w - cx)/fx],
            #                       [(0.0 - cx)/fx, (w - cx)/fx]],
            #                        [[1.0, 1.0],
            #                         [1.0, 1.0]],
            #                        [[-(0.0 - cy)/fy, -(0.0 - cy)/fy],
            #                         [-(h - cy)/fy, -(h - cy)/fy]]]], dtype=torch.float, device=roi_gt_parameters.device)
            # q dot p - 1 = 0
            roi_gt_planes = roi_gt_parameters / roi_gt_parameters.norm(dim=1, keepdim=True).square()
            # roi_gt_planes = roi_gt_parameters
            support = (roi_gt_planes.view(-1, 3, 1) * ranges_rois).sum(dim=1) * (config.BASELINE * fx)
            # if (support < -191).any() or (support > 191).any():
            #     print(support)
            pass

        ## Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        ## binary cross entropy loss.
    else:
        positive_count = 0

    ## 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    # negative_roi_bool = negative_roi_bool & no_crowd_bool
    ## Negative ROIs. Add enough to maintain positive:negative ratio.
    if (negative_roi_bool > 0).sum() > 0 and positive_count>0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.data, :]
    else:
        negative_count = 0

    #print('count', positive_count, negative_count)
    #print(roi_gt_class_ids)
    
    ## Append negative ROIs and pad bbox deltas and masks that
    ## are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        if config.NUM_PARAMETER_CHANNELS > 0:
            zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1], 2), requires_grad=False)
        else:
            zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
            pass
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)

        zeros = torch.zeros(negative_count, support.shape[1], device=support.device, requires_grad=False)
        support = torch.cat([support, zeros], dim=0)
        
        zeros = Variable(torch.zeros(negative_count, config.NUM_PARAMETERS), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_parameters = torch.cat([roi_gt_parameters, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros

        zeros = torch.zeros(negative_count, support.shape[1], device=support.device,
                            requires_grad=False)
        support = zeros

        zeros = Variable(torch.zeros(negative_count, config.NUM_PARAMETERS), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_parameters = torch.cat([roi_gt_parameters, zeros], dim=0)        
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        support = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_parameters = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()
            support = support.cuda()
            roi_gt_parameters = roi_gt_parameters.cuda()
            pass

    return rois, roi_gt_class_ids, deltas, masks, roi_gt_parameters, support


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes = torch.stack([boxes[:, 0].clamp(float(window[0]), float(window[2])), boxes[:, 1].clamp(float(window[1]), float(window[3])), boxes[:, 2].clamp(float(window[0]), float(window[2])), boxes[:, 3].clamp(float(window[1]), float(window[3]))], dim=-1)
    return boxes

def refine_detections(rois, probs, deltas, parameters,
                      window, config, return_indices=False, use_nms=1, one_hot=True):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    ## Class IDs per ROI
    
    if len(probs.shape) == 1:
        class_ids = probs.long()
    else:
        _, class_ids = torch.max(probs, dim=1)
        pass

    ## Class probability of the top class of each ROI
    ## Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()
        
    if len(probs.shape) == 1:
        class_scores = torch.ones(class_ids.shape)
        deltas_specific = deltas
        class_parameters = parameters
        if config.GPU_COUNT:
            class_scores = class_scores.cuda()
    else:
        class_scores = probs[idx, class_ids.data]
        deltas_specific = deltas[idx, class_ids.data]
        class_parameters = parameters[idx, class_ids.data]
    ## Apply bounding box deltas
    ## Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
        
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)
    ## Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois = refined_rois * scale
    ## Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    ## Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)
    
    ## TODO: Filter out boxes with zero area

    ## Filter out background boxes
    keep_bool = class_ids > 0

    ## Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE and False:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)

    keep_bool = keep_bool & (refined_rois[:, 2] > refined_rois[:, 0]) & (refined_rois[:, 3] > refined_rois[:, 1])

    if keep_bool.sum() == 0:
        if return_indices:
            return torch.zeros((0, 10)).cuda(), torch.zeros(0).long().cuda(), torch.zeros((0, 4)).cuda()
        else:
            return torch.zeros((0, 10)).cuda()
        pass
        
    keep = torch.nonzero(keep_bool)[:,0]

    if use_nms == 2:
        ## Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]

        ixs = torch.arange(len(pre_nms_class_ids)).long().cuda()
        ## Sort
        ix_rois = pre_nms_rois
        ix_scores = pre_nms_scores
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.data,:]
        
        # nms_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
        nms_keep = torchvision.ops.nms(ix_rois.index_select(1, torch.LongTensor([1, 0, 3, 2]).cuda()), ix_scores, config.DETECTION_NMS_THRESHOLD)
        nms_keep = keep[ixs[order[nms_keep].data].data]
        keep = intersect1d(keep, nms_keep)        
    elif use_nms == 1:
        ## Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]

        for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
            ## Pick detections of this class
            ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]

            ## Sort
            ix_rois = pre_nms_rois[ixs.data]
            ix_scores = pre_nms_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order.data,:]

            # class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
            class_keep = torchvision.ops.nms(ix_rois.index_select(1, torch.LongTensor([1, 0, 3, 2]).cuda()), ix_scores, config.DETECTION_NMS_THRESHOLD)

            ## Map indicies
            class_keep = keep[ixs[order[class_keep].data].data]

            if i==0:
                nms_keep = class_keep
            else:
                nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
        keep = intersect1d(keep, nms_keep)
    else:
        pass

    ## Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]
    #print('num detectinos', len(keep))

    ### Apply plane anchors
    class_parameters = config.applyAnchorsTensor(class_ids, class_parameters)
    ## Arrange output as [N, (y1, x1, y2, x2, class_id, score, parameters)]
    ## Coordinates are in image domain.
    result = torch.cat((refined_rois[keep.data],
                        class_ids[keep.data].unsqueeze(1).float(),
                        class_scores[keep.data].unsqueeze(1),
                        class_parameters[keep.data]), dim=1)

    if return_indices:
        ori_rois = rois * scale
        ori_rois = clip_to_window(window, ori_rois)
        ori_rois = torch.round(ori_rois)
        ori_rois = ori_rois[keep.data]
        return result, keep.data, ori_rois
    
    return result


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, mrcnn_parameter,
                    image_meta, return_indices=False, use_nms=1, one_hot=True):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    ## Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    if len(mrcnn_class) == 0:
        if return_indices:
            return torch.zeros(0), torch.zeros(0), torch.zeros(0)
        else:
            return torch.zeros(0)
        
    return refine_detections(rois, mrcnn_class, mrcnn_bbox, mrcnn_parameter, window, config,
                             return_indices=return_indices, use_nms=use_nms, one_hot=one_hot)


############################################################
#  Region Proposal Network
############################################################

class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        ## Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        ## Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        ## Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        ## Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        ## Bounding box refinement. [batch, H, W, anchors per location, depth]
        ## where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        ## Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0,2,3,1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  Region Proposal Network anchor
############################################################

class RPNanchor(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth, desc_len):
        super(RPNanchor, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth
        self.desc_len = desc_len

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_desc = nn.Conv2d(512, self.desc_len * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        ## Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        ## Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        ## Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        ## Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        rpn_desc = self.conv_desc(x)

        ## Reshape to [batch, 4, anchors]
        rpn_desc = rpn_desc.permute(0, 2, 3, 1)
        rpn_desc = rpn_desc.contiguous()
        rpn_desc = rpn_desc.view(x.size()[0], -1, self.desc_len)

        return [rpn_class_logits, rpn_probs, rpn_desc]


class DescDist(nn.Module):
    """Builds the model of descriptor distance computation.

    desc_len

    Returns:
        score: [batch, 2]
    """
    def __init__(self, desc_len):
        super(DescDist, self).__init__()
        self.desc_len = desc_len
        self.nfeats = 128

        self.conv1 = nn.Conv1d(2 * self.desc_len, 128, 1)

        self.block1 = self.make_block(self.nfeats)
        self.block2 = self.make_block(self.nfeats)
        self.block3 = self.make_block(self.nfeats)
        self.block4 = self.make_block(self.nfeats)

        self.conv2 = nn.Conv1d(128, 2, 1)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def make_block(nfeats):
        return nn.Sequential(nn.Conv1d(nfeats, nfeats, 1),
                             nn.InstanceNorm1d(nfeats, momentum=0.01),
                             nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        # adding dummy dimension to enable 2D convolution
        x = torch.cat([x1, x2], dim=2).transpose(1, 2)
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        score_logit = self.conv2(x)
        score = self.softmax(score_logit)
        return score_logit, score


############################################################
#  Feature Pyramid Network Heads
############################################################

class Classifier(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes, num_parameters, debug=False):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_parameters = num_parameters
        self.padding = SamePad2d(kernel_size=3, stride=1)
        # self.conv1 = nn.Conv2d(self.depth + 64, self.depth + 64, kernel_size=3, stride=1)
        # self.bn1 = nn.BatchNorm2d(self.depth + 64, eps=0.001, momentum=0.01)
        # self.conv1b = nn.Conv2d(self.depth + 64, 1024, kernel_size=self.pool_size, stride=1)
        # self.bn1b = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv1 = nn.Conv2d(self.depth + 64, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

        self.debug = debug
        if self.debug:
            self.linear_parameters = nn.Linear(3, num_classes * self.num_parameters)
        else:
            self.linear_parameters = nn.Linear(1024, num_classes * self.num_parameters)
            pass

    def forward(self, x, rois, ranges, config, disp, pool_features=True, gt=None, writer=None):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        ranges = coordinates_roi([rois] + [ranges, ], self.pool_size, self.image_shape)

        roi_features = torch.cat([x, ranges], dim=1)
        # roi_features = torch.cat([x, ranges], dim=1)
        # x = self.conv1(self.padding(roi_features))
        # x = self.bn1(x)
        # x = self.conv1b(x)
        # x = self.bn1b(x)
        x = self.conv1(roi_features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, 1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        if self.debug:
            x = gt
            pass
        
        mrcnn_parameters = self.linear_parameters(x)
        
        if self.debug:
            pass
        
        mrcnn_parameters = mrcnn_parameters.view(mrcnn_parameters.size()[0], -1, self.num_parameters)
        if pool_features:
            return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, mrcnn_parameters, roi_features]
        else:
            return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, mrcnn_parameters]


class Mask(nn.Module):
    def __init__(self, config, depth, pool_size, image_shape, num_classes):
        super(Mask, self).__init__()
        self.config = config
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes + config.NUM_PARAMETER_CHANNELS, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois, config, disp, pool_features=True, writer=None):
        if pool_features:
            roi_features = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        else:
            roi_features = x
            pass
        x = self.conv1(self.padding(roi_features))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)

        x_mask = self.deconv(x)
        x_mask = self.relu(x_mask)
        x_mask = self.conv5(x_mask)

        if self.config.NUM_PARAMETER_CHANNELS > 0 and not self.config.OCCLUSION:
            x_mask = torch.cat([self.sigmoid(x_mask[:, :-self.num_parameter_channels]), x_mask[:, -self.num_parameter_channels:]], dim=1)
        else:
            x_mask = self.sigmoid(x_mask)
            pass
        return x_mask, roi_features


class PlaneParams(nn.Module):
    def __init__(self, config, depth, pool_size, image_shape, num_classes):
        super(PlaneParams, self).__init__()
        self.config = config
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_feats = 1024
        self.num_pts = 1
        self.padding = SamePad2d(kernel_size=5, stride=1)
        self.padding2 = SamePad2d(kernel_size=5, stride=2)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.25)

        # features and mask as input
        self.conv_vis = nn.Conv2d(self.depth + num_classes + 64,
                                  self.num_pts * self.num_feats,
                                  kernel_size=self.pool_size,
                                  stride=1)
        self.bn_vis = nn.BatchNorm1d(self.num_pts * self.num_feats, eps=0.001, momentum=0.01)

        # disparity and mask as input
        self.conv_geom = nn.Conv2d(1 + num_classes + 64,
                                   self.num_pts * self.num_feats,
                                   kernel_size=self.pool_size,
                                   stride=1)
        self.bn_geom = nn.BatchNorm1d(self.num_pts * self.num_feats, eps=0.001, momentum=0.01)

        self.lin_sup = nn.Linear(2*self.num_pts*self.num_feats, self.num_pts*self.num_feats)
        self.bn1 = nn.BatchNorm1d(self.num_pts * self.num_feats, eps=0.001, momentum=0.01)
        # self.lin_sup_disp = nn.Linear(self.num_pts * self.num_feats, self.num_pts * self.config.MAXDISP * 2)
        self.lin_sup_class = nn.Linear(self.num_pts * self.num_feats, 3 * num_classes)

        self.values = None

    def forward(self, x, rois, masks, disp, ranges_feat, writer=None, target=None, target_class=None, target_params=None):
        # if self.training and rois.shape[0] < 16:
        #     ## Set batchnorm in eval mode during training when few ROIs
        #     def set_bn_eval(m):
        #         classname = m.__class__.__name__
        #         if classname.find('BatchNorm') != -1:
        #             m.eval()
        #
        #     self.apply(set_bn_eval)
        # elif self.training:
        #     ## Set batchnorm in training when enough ROIs
        #     def set_bn_train(m):
        #         classname = m.__class__.__name__
        #         if classname.find('BatchNorm') != -1:
        #             m.train()
        #
        #     self.apply(set_bn_train)

        roi_feat = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        roi_ranges = coordinates_roi([rois] + [ranges_feat], self.pool_size, self.image_shape)
        roi_disp = disp_roi_align([rois] + [disp], self.pool_size)
        # masks_pos = masks[0, :, 1:2, :, :]
        # roi_disp_mean = torch.sum(roi_disp * masks_pos, dim=(2, 3), keepdim=True) /\
        #                 torch.clamp(torch.sum(masks_pos, dim=(2, 3), keepdim=True), min=1.0)
        # roi_disp_stddev = torch.sum(((roi_disp - roi_disp_mean) * masks_pos).square(), dim=(2, 3), keepdim=True) / \
        #                   torch.clamp(torch.sum(masks_pos, dim=(2, 3), keepdim=True), min=1.0)
        # roi_disp_stddev = torch.clamp(roi_disp_stddev.sqrt(), min=1.0)
        # roi_disp_norm = (roi_disp - roi_disp_mean) / roi_disp_stddev

        x_vis = torch.cat([roi_feat, roi_ranges, masks.squeeze(0)], dim=1)
        x_vis = self.conv_vis(x_vis)
        x_vis = self.relu(x_vis)
        x_vis = x_vis.view(-1, self.num_pts * self.num_feats)
        x_vis = self.bn_vis(x_vis)

        x_geom = torch.cat([roi_disp, roi_ranges, masks.squeeze(0)], dim=1)
        x_geom = self.conv_geom(x_geom)
        x_geom = self.relu(x_geom)
        x_geom = x_geom.view(-1, self.num_pts*self.num_feats)
        x_geom = self.bn_geom(x_geom)

        x = torch.cat([x_vis, x_geom], dim=1)

        x = self.lin_sup(x)
        x = self.relu(x)
        x = self.bn1(x)

        x_class = self.lin_sup_class(x)
        x_class = x_class.view(-1, self.num_classes, 3)

        # if target is not None and target_class is not None:
        #     if (target_class > 0).sum() > 0:
        #         ## Only positive ROIs contribute to the loss. And only
        #         ## the class specific mask of each ROI.
        #         positive_ix = torch.nonzero(target_class > 0)[:, 0]
        #         # positive_class_ids = target_class_ids[positive_ix.data].long()
        #         # indices = torch.stack((positive_ix, positive_class_ids), dim=1)
        #
        #         ## Gather the masks (predicted and true) that contribute to loss
        #         target_support_pos = target[positive_ix, :]
        #         mrcnn_support_pos = pred[positive_ix, :]
        #
        #         # outlier_mask = target_norm.abs() > 20.0
        #         # if outlier_mask.any():
        #         #     torch.set_printoptions(precision=4, sci_mode=False)
        #         #     print('\ntarget_norm = ', target_norm[outlier_mask.any(1)])
        #         #     print('target_support_pos = ', target_support_pos[outlier_mask.any(1)])
        #         #     print('target_params = ', target_params[positive_ix, :][outlier_mask.any(1)])
        #         #     print('rois = ', rois[0, positive_ix, :][outlier_mask.any(1)])
        #         #     print('roi_disp_mean = ', roi_disp_mean.view(-1, 1)[positive_ix, :][outlier_mask.any(1)])
        #         #     print('roi_disp_stddev = ', roi_disp_stddev.view(-1, 1)[positive_ix, :][outlier_mask.any(1)])
        #         #     torch.set_printoptions(precision=1, linewidth=180)
        #         #     print('roi_disp = ', (roi_disp * masks_pos)[positive_ix, :, :, :][outlier_mask.any(1)])
        #         #     print('masks_pos = ', masks_pos[positive_ix, :, :, :][outlier_mask.any(1)])
        #         #     torch.set_printoptions()
        #         # if self.values is None:
        #         #     self.values = target_norm.view(-1)
        #         # else:
        #         #     max_values = 100000
        #         #     self.values = torch.cat([self.values, target_norm.view(-1)])
        #         #     if self.values.shape[0] > max_values:
        #         #         self.values = self.values[-max_values:]
        #         #
        #         # if writer is not None:
        #         #     writer.add_histogram('support_disp', self.values, bins=50)
        #
        #         loss = F.smooth_l1_loss(mrcnn_support_pos, target_support_pos)
        #         if loss > 6 or torch.isnan(loss).any():
        #             print('loss')
        #             # torch.set_printoptions(precision=4, sci_mode=False)
        #             # print('\ntarget_norm = ', target_norm[outlier_mask.any(1)])
        #             # print('target_support_pos = ', target_support_pos[outlier_mask.any(1)])
        #             # print('target_params = ', target_params[positive_ix, :][outlier_mask.any(1)])
        #             # print('rois = ', rois[0, positive_ix, :][outlier_mask.any(1)])
        #             # print('roi_disp_mean = ', roi_disp_mean.view(-1, 1)[positive_ix, :][outlier_mask.any(1)])
        #             # print('roi_disp_stddev = ', roi_disp_stddev.view(-1, 1)[positive_ix, :][outlier_mask.any(1)])
        #             # torch.set_printoptions(precision=1, linewidth=180)
        #             # print('roi_disp = ', (roi_disp * masks_pos)[positive_ix, :, :, :][outlier_mask.any(1)])
        #             # print('masks_pos = ', masks_pos[positive_ix, :, :, :][outlier_mask.any(1)])
        #             # torch.set_printoptions()

        # if torch.isnan(pred).any() or torch.isinf(pred).any():
        #     print(pred)

        return x_class


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x*disp, 1)
        return out


class Depth(nn.Module):
    def __init__(self, num_input_feats, num_output_channels=1):
        super(Depth, self).__init__()
        self.num_output_channels = num_output_channels        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input_feats, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        # self.conv1b = nn.Sequential(
        #         nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
        #         nn.ReLU(inplace=True)
        # )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input_feats, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        # self.conv2b = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
        #     nn.ReLU(inplace=True)
        # )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_input_feats, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        # self.conv3b = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
        #     nn.ReLU(inplace=True)
        # )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_input_feats, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        # self.conv4b = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
        #     nn.ReLU(inplace=True)
        # )
        self.conv5 = nn.Sequential(
            nn.Conv2d(num_input_feats, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        # self.conv5b = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
        #     nn.ReLU(inplace=True)
        # )
        
        self.deconv1 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        self.depth_pred = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)
        self.conv_pred = nn.Sequential(
                nn.Conv2d(num_output_channels, num_output_channels, kernel_size=3, stride=1, padding=1)
        )

        self.crop = False
        return
    
    def forward(self, feature_maps):
        # testing only disparity features
        # for c in range(0, 5):
        #     feature_maps[c] = feature_maps[c][:, 256:448]
        if self.crop:
            padding = 5
            for c in range(2, 5):
                feature_maps[c] = feature_maps[c][:, :, padding * pow(2, c - 2):-padding * pow(2, c - 2)]
                continue
            pass
        x = self.deconv1(self.conv1(feature_maps[0]))
        x = self.deconv2(torch.cat([self.conv2(feature_maps[1]), x], dim=1))
        if self.crop:
            x = x[:, :, 5:35]
        x = self.deconv3(torch.cat([self.conv3(feature_maps[2]), x], dim=1))
        x = self.deconv4(torch.cat([self.conv4(feature_maps[3]), x], dim=1))
        x = self.deconv5(torch.cat([self.conv5(feature_maps[4]), x], dim=1))
        x = self.depth_pred(x)

        # x = disparityregression(192)(feature_maps[4][:, 256:448])[:, None, :, :]
        # x = 100.0 / torch.clamp(x, min=1.0)
        # x[:, :, :20, :] = 0.0
        # x[:, :, -20:, :] = 0.0
        
        if self.crop:
            x = torch.nn.functional.interpolate(x, size=(480, 640), mode='bilinear')
            zeros = torch.zeros((len(x), self.num_output_channels, 80, 640)).cuda()
            x = torch.cat([zeros, x, zeros], dim=2)
        else:
            x = torch.nn.functional.interpolate(x, size=(640, 640), mode='bilinear')
            pass

        x = self.conv_pred(x)

        return x


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class Hourglass(nn.Module):
    def __init__(self, inplanes):
        super(Hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class DepthStereo(nn.Module):
    def __init__(self, maxdisp, im_h, im_w, inplanes):
        super(DepthStereo, self).__init__()

        self.maxdisp = maxdisp
        self.im_h = im_h
        self.im_w = im_w
        self.inplanes = inplanes

        self.layer4 = self._make_layer(BasicBlock, 32, 3, 1, 1, 2)


        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
        #                               nn.ReLU(inplace=True),
        #                               convbn_3d(32, 1, 3, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, feat_left, feat_right):
        feat_left = self.layer4(feat_left)
        feat_right = self.layer4(feat_right)

        # matching
        cost = Variable(torch.FloatTensor(feat_left.size()[0],
                                          feat_left.size()[1] * 2,
                                          self.maxdisp // 4,
                                          feat_left.size()[2],
                                          feat_left.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :feat_left.size()[1], i, :, i:] = feat_left[:, :, :, i:]
                cost[:, feat_left.size()[1]:, i, :, i:] = feat_right[:, :, :, :-i]
            else:
                cost[:, :feat_left.size()[1], i, :, :] = feat_left
                cost[:, feat_left.size()[1]:, i, :, :] = feat_right
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)
        cost = F.upsample(cost, [self.maxdisp, self.im_h, self.im_w], mode='trilinear')
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1)
        pred = DisparityRegression(self.maxdisp)(pred)

        return pred
   
   
############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    ## Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    ## Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    ## Positive and Negative anchors contribute to the loss,
    ## but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    ## Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:,0],indices.data[:,1],:]
    anchor_class = anchor_class[indices.data[:,0],indices.data[:,1]]

    ## Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss

def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    ## Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    ## Positive anchors contribute to the loss, but negative and
    ## neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match==1)
    ## Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:,0],indices.data[:,1]]

    ## Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0,:rpn_bbox.size()[0],:]

    ## Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    ## Loss
    if len(target_class_ids) > 0:
        loss = F.cross_entropy(pred_class_logits,target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids), dim=1)

        ## Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:,0].data,:]
        pred_bbox = pred_bbox[indices[:,0].data,indices[:,1].data,:]

        ## Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(config, target_masks, target_class_ids, target_parameters, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        ## Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:,0].data,:,:]

        if config.GLOBAL_MASK:
            y_pred = pred_masks[indices[:,0],0,:,:]
        else:
            y_pred = pred_masks[indices[:,0].data,indices[:,1].data,:,:]
            pass

        if config.NUM_PARAMETER_CHANNELS == 1:
            if config.OCCLUSION:
                visible_pred = pred_masks[indices[:,0],-1,:,:]
                visible_gt = y_true[:, :, :, -1]
                y_true = y_true[:, :, :, 0]
                loss = F.binary_cross_entropy(y_pred, y_true) + F.binary_cross_entropy(visible_pred, visible_gt)
            else:
                depth_pred = pred_masks[indices[:,0],-1,:,:]
                depth_gt = y_true[:, :, :, -1]
                y_true = y_true[:, :, :, 0]
                loss = F.binary_cross_entropy(y_pred, y_true) + l1LossMask(depth_pred, depth_gt, (depth_gt > 1e-4).float())
                pass
        elif config.NUM_PARAMETER_CHANNELS == 4:
            depth_pred = pred_masks[indices[:,0],-config.NUM_PARAMETER_CHANNELS,:,:]
            depth_gt = y_true[:, :, :, -1]
            y_true = y_true[:, :, :, 0]
            normal_pred = pred_masks[indices[:,0],-(config.NUM_PARAMETER_CHANNELS - 1):,:,:]
            normal_gt = target_parameters[indices[:,0]]
            normal_gt = normal_gt / torch.clamp(torch.norm(normal_gt, dim=-1, keepdim=True), min=1e-4)
            loss = F.binary_cross_entropy(y_pred, y_true) + l1LossMask(depth_pred, depth_gt, (depth_gt > 1e-4).float()) + l2NormLossMask(normal_pred, normal_gt.unsqueeze(-1).unsqueeze(-1), y_true, dim=1)
        else:
            ## Binary cross entropy
            loss = F.binary_cross_entropy(y_pred, y_true)
            pass
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss

def compute_mrcnn_parameter_loss(target_parameters, target_class_ids, pred_parameters):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids), dim=1)

        ## Gather the deltas (predicted and true) that contribute to loss
        target_parameters = target_parameters[indices[:,0].data,:]
        pred_parameters = pred_parameters[indices[:,0].data,indices[:,1].data,:]
        ## Smooth L1 loss
        loss = F.smooth_l1_loss(pred_parameters, target_parameters)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()
    return loss


def compute_mrcnn_support_loss(target_support, mrcnn_support, mrcnn_support_class, target_class_ids):
    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        # positive_class_ids = target_class_ids[positive_ix.data].long()
        # indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        ## Gather the masks (predicted and true) that contribute to loss
        target_support_pos = target_support[positive_ix, :]
        mrcnn_support_pos = mrcnn_support[positive_ix, :]
        target_support_class_pos = torch.where(target_support_pos > 0,
                                               torch.zeros(1, dtype=torch.long, device=target_support.device),
                                               torch.ones(1, dtype=torch.long, device=target_support.device))
        mrcnn_support_class_pos = mrcnn_support_class[positive_ix, :]

        mrcnn_support_pos = torch.where(target_support_pos > 0, mrcnn_support_pos[:, 0, :], mrcnn_support_pos[:, 1, :])

        loss_disp = F.smooth_l1_loss(mrcnn_support_pos, target_support_pos)
        loss_class = F.cross_entropy(mrcnn_support_class_pos, target_support_class_pos)
        # if loss > 1 or torch.isnan(loss).any():
        #     print('\nsupport loss: ', loss.float())
        #     print('num rois: ', target_support_pos.shape[0])
        #     print('mean target disp: ', torch.mean(target_support_pos).float())
        #     print('stddev target disp: ', torch.std(target_support_pos).float())
    else:
        loss_disp = torch.zeros(1, dtype=torch.float, device=target_support.device)
        loss_class = torch.zeros(1, dtype=torch.float, device=target_support.device)
    return loss_disp, loss_class

def compute_losses(config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                   target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters,
                   target_support, mrcnn_support, mrcnn_support_class):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(config, target_mask, target_class_ids, target_parameters, mrcnn_mask)
    mrcnn_parameter_loss = 50*compute_mrcnn_parameter_loss(target_parameters, target_class_ids, mrcnn_parameters)
    mrcnn_support_loss, mrcnn_support_class_loss = compute_mrcnn_support_loss(target_support, mrcnn_support, mrcnn_support_class, target_class_ids)
    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_parameter_loss,
            mrcnn_support_loss, mrcnn_support_class_loss]



############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir='test'):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture.
        """

        ## Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        ## Build the shared convolutional layers.
        ## Bottom-up Layers
        ## Returns a list of the last layers of each stage, 5 in total.
        ## Don't create the thead (stage 5), so we pick the 4th item in the list.
        # resnet = ResNet("resnet101", stage5=True, numInputChannels=config.NUM_INPUT_CHANNELS)
        resnet = ResNet("resnet101", stage5=True, numInputChannels=3)
        # resnet = ResNet("resnet50", stage5=True, numInputChannels=config.NUM_INPUT_CHANNELS)
        C1, C2, C3, C4, C5 = resnet.stages()

        ## Top-down Layers
        ## TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256, bilinear_upsampling=self.config.BILINEAR_UPSAMPLING)

        ## Generate Anchors
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                config.RPN_ANCHOR_RATIOS,
                                                                                config.BACKBONE_SHAPES,
                                                                                config.BACKBONE_STRIDES,
                                                                                config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        ## RPN
        # self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256 + config.MAXDISP)
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        ## Coordinate feature
        self.range_conv = nn.Conv2d(3, 64, kernel_size=1, stride=1)

        # self.disp = nn.Sequential(nn.Conv2d(self.config.MAXDISP, self.config.MAXDISP, kernel_size=3, stride=1),
        #                           nn.BatchNorm2d(self.config.MAXDISP, eps=0.001, momentum=0.01),
        #                           nn.ReLU(inplace=True),
        #                           nn.Conv2d(self.config.MAXDISP, self.config.MAXDISP, kernel_size=3, stride=1),
        #                           nn.BatchNorm2d(self.config.MAXDISP, eps=0.001, momentum=0.01),
        #                           nn.ReLU(inplace=True),
        #                           nn.Conv2d(self.config.MAXDISP, self.config.MAXDISP, kernel_size=3, stride=1),
        #                           nn.BatchNorm2d(self.config.MAXDISP, eps=0.001, momentum=0.01),
        #                           nn.ReLU(inplace=True),
        #                           nn.Conv2d(self.config.MAXDISP, self.config.MAXDISP, kernel_size=3, stride=1),
        #                           nn.BatchNorm2d(self.config.MAXDISP, eps=0.001, momentum=0.01),
        #                           nn.ReLU(inplace=True),
        #                           nn.Conv2d(self.config.MAXDISP, self.config.MAXDISP, kernel_size=3, stride=1),
        #                           nn.BatchNorm2d(self.config.MAXDISP, eps=0.001, momentum=0.01),
        #                           nn.ReLU(inplace=True))

        self.disp_conv = nn.Sequential(nn.Conv2d(self.config.MAXDISP, self.config.MAXDISP, kernel_size=1, stride=1),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm2d(self.config.MAXDISP, eps=0.001, momentum=0.01))

        ## FPN Classifier
        self.debug = False
        # self.classifier = Classifier(256 + config.MAXDISP, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES, config.NUM_PARAMETERS, debug=self.debug)
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES, config.NUM_PARAMETERS, debug=self.debug)

        ## FPN Mask
        # self.mask = Mask(config, 256 + config.MAXDISP, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)
        self.mask = Mask(config, 256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        self.plane_params = PlaneParams(config, 256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        if self.config.PREDICT_DEPTH:
            if self.config.PREDICT_STEREO:
                self.depth = DepthStereo(self.config.MAXDISP,
                                         self.config.IMAGE_SHAPE[0],
                                         self.config.IMAGE_SHAPE[1],
                                         256)
            else:
                if self.config.PREDICT_BOUNDARY:
                    self.depth = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
                    # self.depth = Depth(256 + config.MAXDISP, num_output_channels=3)
                    # self.depth = Depth(256, num_output_channels=3)
                else:
                    self.depth = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
                    # self.depth = Depth(config.MAXDISP, num_output_channels=1)
                    # self.depth = Depth(256, num_output_channels=1)
                    pass
                pass

        ## Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        # self.bn_exceptions = {'classifier', 'mask', 'plane_params'}
        self.bn_exceptions = {}
        for (mname, m) in self.named_children():
            if mname not in self.bn_exceptions:
                m.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        ## Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        ## If we have a model path with date and epochs use them
        if model_path:
            ## Continue from we left of. Get epoch and date from the file name
            ## A sample model path might look like:
            ## /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        ## Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        ## Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        ## Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        ## Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        ## Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            try:
                self.load_state_dict(state_dict, strict=False)
            except:
                print('load only base model')
                try:
                    state_dict = {k: v for k, v in state_dict.items() if 'classifier.linear_class' not in k
                                  and 'classifier.linear_bbox' not in k
                                  and 'mask.conv5' not in k}
                    state = self.state_dict()
                    state.update(state_dict)
                    self.load_state_dict(state)
                except:
                    print('change input dimension')
                    state_dict = {k: v for k, v in state_dict.items() if 'classifier.linear_class' not in k
                                  and 'classifier.linear_bbox' not in k
                                  and 'mask.conv5' not in k
                                  and 'mask.conv1' not in k
                                  # and 'fpn.C1.0' not in k
                                  and 'classifier.conv1' not in k
                                  and 'classifier.bn1' not in k
                                  and 'rpn.conv_shared' not in k
                                  }
                    state = self.state_dict()
                    state.update(state_dict)
                    self.load_state_dict(state)
                    pass
                pass
        else:
            print("Weight file not found ...")
            exit(1)
        ## Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def detect(self, images, camera, mold_image=True, image_metas=None):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        ## Mold inputs to format expected by the neural network
        if mold_image:
            molded_images, image_metas, windows = mold_inputs(self.config, images)
        else:
            molded_images = images
            windows = [(0, 0, images.shape[1], images.shape[2]) for _ in range(len(images))]
            pass

        ## Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

        ## To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        ## Wrap in variable
        #molded_images = Variable(molded_images, volatile=True)

        ## Run object detection
        detections, mrcnn_mask, depth_np = self.predict([molded_images, image_metas, camera], mode='inference')

        if len(detections[0]) == 0:
            return [{'rois': [], 'class_ids': [], 'scores': [], 'masks': [], 'parameters': []}]
        
        ## Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        ## Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks, final_parameters =\
                unmold_detections(self.config, detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "parameters": final_parameters,                
            })
        return results

    def predict(self, input, mode, use_nms=1, use_refinement=False, return_feature_map=False, writer=None):
        molded_images = input[0]
        image_metas = input[1]
        camera = input[6]
        fx = camera[0]

        gt_depth = input[7].unsqueeze(1)
        if gt_depth is not None:
            gt_disp = fx * self.config.BASELINE / torch.clamp(gt_depth, min=1.0e-4)
            gt_disp = torch.clamp(gt_disp, min=0.0, max=self.config.MAXDISP)

        # # TODO Checking disp features
        # gt_depth = input[7].unsqueeze(1)
        # camera = input[6]
        # fx = camera[0]
        # gt_disp = fx * torch.tensor(self.config.BASELINE, dtype=torch.float, device=gt_depth.device, requires_grad=False) / \
        #                 torch.clamp(gt_depth, min=1.0e-4)
        # gt_disp = torch.clamp(gt_disp, min=0.0, max=self.config.MAXDISP)

        # remove mean value
        # molded_images = torch.cat([molded_images, 10*(gt_disp - 15.0)], dim=1)
        # molded_images = torch.cat([molded_images, gt_depth - 5.0], dim=1)

        if mode == 'inference':
            self.eval()
        elif 'training' in mode:
            self.train()
            ## Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            for (mname, m) in self.named_children():
                if mname not in self.bn_exceptions:
                    m.apply(set_bn_eval)

        ## Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)
        ## Note that P6 is used in RPN, but not in the classifier heads.

        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # # TODO Checking disp features
        # if writer is not None:
        #     writer.add_image('disp_feat/image',
        #                      torch.clamp(unmold_image_torch(molded_images, self.config), min=0, max=255).squeeze(0),
        #                      dataformats='CHW')
        #
        # h_max = rpn_feature_maps[0].shape[2]
        # w_max = rpn_feature_maps[0].shape[3]
        # cur_disp = torch.nn.functional.interpolate(gt_disp,
        #                                            size=(h_max, w_max),
        #                                            mode='bilinear').view(-1, h_max, w_max, 1)
        # cur_disp = torch.clamp(cur_disp, min=0.0, max=self.config.MAXDISP - 1)
        # disp_vol = torch.zeros((1, h_max, w_max, self.config.MAXDISP),
        #                        dtype=torch.float, device=cur_disp.device, requires_grad=False)
        #
        # cur_disp_low = cur_disp.floor().to(torch.long)
        # cur_disp_high = cur_disp.ceil().to(torch.long)
        # mask = cur_disp_high < self.config.MAXDISP
        #
        # # urange = torch.arange(w, dtype=torch.long, requires_grad=False).cuda().reshape(1, -1).repeat(h, 1)
        # # vrange = torch.arange(h, dtype=torch.long, requires_grad=False).cuda().reshape(-1, 1).repeat(1, w)
        # # ind = torch.stack([torch.zeros_like(cur_disp_low).cuda(),
        # #                    cur_disp_low,
        # #                    vrange.expand_as(cur_disp_low),
        # #                    urange.expand_as(cur_disp_low)],
        # #                   dim=-1)
        # mask_vol = torch.zeros_like(disp_vol, dtype=torch.bool, device=cur_disp.device, requires_grad=False)
        # mask_vol.scatter_(-1, cur_disp_low.clamp(max=self.config.MAXDISP - 1), mask)
        # disp_vol[mask_vol] = 1.0 - (cur_disp[mask] - cur_disp_low[mask])
        #
        # mask_vol = torch.zeros_like(disp_vol, dtype=torch.bool, device=cur_disp.device, requires_grad=False)
        # mask_vol.scatter_(-1, cur_disp_high.clamp(max=self.config.MAXDISP - 1), mask)
        # disp_vol[mask_vol] = 1.0 - (cur_disp_high[mask] - cur_disp[mask])
        #
        # # move disp dimension (3) to 1
        # disp_vol = disp_vol.permute(0, 3, 1, 2)
        # disp_feat = self.disp_conv(disp_vol)
        #
        # if writer is not None:
        #     pred = DisparityRegression(self.config.MAXDISP)(disp_vol)
        #     min_d = pred.min()
        #     max_d = pred.max()
        #     writer.add_image('disp_feat/disp_est', (pred.squeeze(0) - min_d) / (max_d - min_d), dataformats='HW')
        #     writer.add_image('disp_feat/disp_gt', (cur_disp.squeeze(0).squeeze(-1) - min_d) / (max_d - min_d),
        #                      dataformats='HW')
        #
        # for stage in range(2, 7):
        #     h = rpn_feature_maps[stage-2].shape[2]
        #     w = rpn_feature_maps[stage-2].shape[3]
        #
        #     cur_disp_feat = torch.nn.functional.interpolate(disp_feat,
        #                                                    size=(h, w),
        #                                                    mode='bilinear')
        #
        #     rpn_feature_maps[stage-2] = torch.cat([rpn_feature_maps[stage-2], cur_disp_feat], dim=1)
        #     if stage < 6:
        #         mrcnn_feature_maps[stage-2] = torch.cat([mrcnn_feature_maps[stage-2], cur_disp_feat], dim=1)

        feature_maps = [feature_map for index, feature_map in enumerate(rpn_feature_maps[::-1])]
        if self.config.PREDICT_DEPTH:
            if self.config.PREDICT_STEREO:
                molded_images_r = input[7]
                [p2_out_r, p3_out_r, p4_out_r, p5_out_r, p6_out_r] = self.fpn(molded_images_r)
                if self.training:
                    disp_np = self.depth(p2_out, p2_out_r)
                else:
                    disp_np = self.depth(p2_out, p2_out_r)
                pass
                camera = input[6]
                fx = camera[0]
                depth_np = fx * self.config.BASELINE / torch.clamp(disp_np, min=1.0e-4)

                if writer is not None:
                    min_l = p2_out.min()
                    max_l = p2_out.max()
                    min_r = p2_out_r.min()
                    max_r = p2_out_r.max()
                    writer.add_image('disp/feat_l', (p2_out[0, 0] - min_l)/(max_l - min_l), dataformats='HW')
                    writer.add_image('disp/feat_r', (p2_out_r[0, 0] - min_r)/(max_r - min_r), dataformats='HW')
            else:
                depth_np = self.depth(feature_maps)
                disp_np = fx * self.config.BASELINE / torch.clamp(depth_np, min=1.0e-4)

                # disp_np = disparityregression(self.config.MAXDISP)(disp_vol)[:, None, :, :]

                # disp_np = gt_disp
                # depth_np = 100.0 / torch.clamp(disp_np, min=1.0)
                # depth_np[:, :, :20, :] = 0.0
                # depth_np[:, :, -20:, :] = 0.0
                #
                # depth_np = torch.nn.functional.interpolate(depth_np, size=(640, 640), mode='bilinear')
                # depth_np = self.depth(depth_np)

                if self.config.PREDICT_BOUNDARY:
                    boundary = depth_np[:, 1:]
                    depth_np = depth_np[:, 0]
                else:
                    depth_np = depth_np.squeeze(1)
                    pass
        else:
            depth_np = torch.ones((1, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)).cuda()
            disp_np = torch.ones((1, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)).cuda()
            pass
        
        ranges = self.config.getRanges(camera).transpose(1, 2).transpose(0, 1)
        zeros = torch.zeros(3, (self.config.IMAGE_MAX_DIM - self.config.IMAGE_MIN_DIM) // 2, self.config.IMAGE_MAX_DIM).cuda()
        ranges = torch.cat([zeros, ranges, zeros], dim=1)
        ranges_inter = torch.nn.functional.interpolate(ranges.unsqueeze(0), size=(160, 160), mode='bilinear')
        ranges_feat = self.range_conv(ranges_inter * 10)

        ## Loop through pyramid layers
        layer_outputs = []  ## list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        ## Concatenate layer outputs
        ## Convert from list of lists of level outputs to list of lists
        ## of outputs across levels.
        ## e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        ## Generate proposals
        ## Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        ## and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if 'training' in mode and use_refinement == False \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                 proposal_count=proposal_count,
                                 nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                 anchors=self.anchors,
                                 config=self.config)
        
        if mode == 'inference':
            ## Network Heads
            ## Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_parameters = self.classifier(mrcnn_feature_maps, rpn_rois, ranges_feat)

            ## Detections
            ## output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, mrcnn_parameters, image_metas)

            if len(detections) == 0:
                return [[]], [[]], depth_np
            ## Convert boxes to normalized coordinates
            ## TODO: let DetectionLayer return normalized coordinates to avoid
            ##       unnecessary conversions
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            detection_boxes = detections[:, :4] / scale

            ## Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)

            ## Create masks for detections
            mrcnn_mask, roi_features = self.mask(mrcnn_feature_maps, detection_boxes)

            ## Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)
            return [detections, mrcnn_mask, depth_np]

        elif mode == 'training':

            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]
            gt_parameters = input[5]
            
            ## Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            ## Generate detection targets
            ## Subsamples proposals and generates target outputs for training
            ## Note that proposal class IDs, gt_boxes, and gt_masks are zero
            ## padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask, target_parameters, target_support = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_parameters, disp_np, self.config, camera)

            if len(rois) == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_parameters = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_parameters = mrcnn_parameters.cuda()
            else:
                ## Network Heads
                ## Proposal classifier and BBox regressor heads
                #print([maps.shape for maps in mrcnn_feature_maps], target_parameters.shape)
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_parameters = self.classifier(mrcnn_feature_maps, rois, ranges_feat, target_parameters)
                
                ## Create masks for detections
                mrcnn_mask, _ = self.mask(mrcnn_feature_maps, rois)

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, rois, depth_np]
        
        elif mode in ['training_detection', 'inference_detection']:
            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]
            gt_parameters = input[5]
            
            ## Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()

            gt_boxes = gt_boxes / scale
                        
            ## Generate detection targets
            ## Subsamples proposals and generates target outputs for training
            ## Note that proposal class IDs, gt_boxes, and gt_masks are zero
            ## padded. Equally, returned rois and targets are zero padded.
            
            rois, target_class_ids, target_deltas, target_mask, target_parameters, target_support = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_parameters, disp_np, self.config, camera)

            if len(rois) == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_parameters = Variable(torch.FloatTensor())
                mrcnn_support = Variable(torch.FloatTensor())
                mrcnn_support_class = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_parameters = mrcnn_parameters.cuda()
                    mrcnn_support = mrcnn_support.cuda()
                    mrcnn_support_class = mrcnn_support_class.cuda()
            else:
                ## Network Heads
                ## Proposal classifier and BBox regressor heads
                [mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_parameters, roi_features] = self.classifier(
                    mrcnn_feature_maps, rois.unsqueeze(0), ranges_feat, self.config, disp_np, pool_features=True, writer=writer)
                ## Create masks for detections
                mrcnn_mask, _ = self.mask(mrcnn_feature_maps, rois.unsqueeze(0), self.config, disp_np, writer=writer)
                target_mask_prob = torch.zeros((target_mask.shape[0],
                                               self.config.NUM_CLASSES,
                                               target_mask.shape[1],
                                               target_mask.shape[2]),
                                              dtype=torch.float,
                                              device=target_mask.device)
                target_mask_prob.scatter_(1,
                                          target_mask.unsqueeze(1).long(),
                                          torch.ones((target_mask.shape[0],
                                                      1,
                                                      target_mask.shape[1],
                                                      target_mask.shape[2]),
                                                     dtype=torch.float,
                                                     device=target_mask.device))
                target_mask_prob = torch.nn.functional.interpolate(target_mask_prob,
                                                                   size=(self.config.MASK_POOL_SIZE,
                                                                         self.config.MASK_POOL_SIZE),
                                                                   mode='nearest')
                mrcnn_parameters = self.plane_params(mrcnn_feature_maps,
                                                      rois.unsqueeze(0),
                                                      target_mask_prob.unsqueeze(0),
                                                      disp_np,
                                                      ranges_feat,
                                                      writer=writer,
                                                      target=target_support,
                                                      target_class=target_class_ids,
                                                      target_params=target_parameters)
                mrcnn_support = torch.zeros((target_support.shape[0], 2, target_support.shape[1]), device=target_support.device)
                mrcnn_support_class = torch.zeros((target_support.shape[0], 2, target_support.shape[1]), device=target_support.device)
                pass

            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()

            if use_refinement:
                [mrcnn_class_logits_final, mrcnn_class_final, mrcnn_bbox_final, mrcnn_parameters_final,
                 roi_features] = self.classifier(mrcnn_feature_maps, rpn_rois, ranges_feat, self.config, disp_np,
                                                 pool_features=True)

                ## Add back batch dimension
                ## Create masks for detections
                detections, indices, _ = detection_layer(self.config, rpn_rois, mrcnn_class_final, mrcnn_bbox_final,
                                                         mrcnn_parameters_final, image_metas, return_indices=True,
                                                         use_nms=use_nms)
                if len(detections) > 0:                
                    detection_boxes = detections[:, :4] / scale                
                    detection_boxes = detection_boxes.unsqueeze(0)
                    detection_masks, _ = self.mask(mrcnn_feature_maps, detection_boxes, self.config,
                                                                      disp_np, writer=writer)
                    detection_masks_h = torch.nn.functional.interpolate(detection_masks,
                                                                        size=(self.config.MASK_POOL_SIZE,
                                                                              self.config.MASK_POOL_SIZE),
                                                                        mode='nearest')
                    class_parameters = self.plane_params(mrcnn_feature_maps,
                                                          detection_boxes,
                                                          detection_masks_h.unsqueeze(0),
                                                          disp_np,
                                                          ranges_feat)

                    _, class_ids = torch.max(mrcnn_class_final, dim=1)
                    class_ids = class_ids[indices]
                    idx = torch.arange(class_ids.size()[0], device=class_ids.device).long()
                    class_parameters = class_parameters[idx, class_ids, :]
                    detections[:, 6:9] = self.config.applyAnchorsTensor(class_ids, class_parameters)

                    detection_support = torch.zeros((class_parameters.shape[0], target_support.shape[1]), device=class_parameters.device)

                    roi_features = roi_features[indices]
                    pass
            else:
                mrcnn_class_logits_final, mrcnn_class_final, mrcnn_bbox_final, mrcnn_parameters_final = mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_parameters

                rpn_rois = rois
                detections, indices, _ = detection_layer(self.config, rpn_rois, mrcnn_class_final, mrcnn_bbox_final,
                                                         mrcnn_parameters_final, image_metas, return_indices=True,
                                                         use_nms=use_nms)

                if len(detections) > 0:
                    detection_boxes = detections[:, :4] / scale
                    detection_boxes = detection_boxes.unsqueeze(0)
                    detection_masks, _ = self.mask(mrcnn_feature_maps, detection_boxes, self.config,
                                                                      disp_np, writer=writer)
                    detection_masks_h = torch.nn.functional.interpolate(detection_masks,
                                                                        size=(self.config.MASK_POOL_SIZE,
                                                                              self.config.MASK_POOL_SIZE),
                                                                        mode='nearest')
                    class_parameters = self.plane_params(mrcnn_feature_maps,
                                                          detection_boxes,
                                                          detection_masks_h.unsqueeze(0),
                                                          disp_np,
                                                          ranges_feat)

                    _, class_ids = torch.max(mrcnn_class_final, dim=1)
                    class_ids = class_ids[indices]
                    idx = torch.arange(class_ids.size()[0], device=class_ids.device).long()
                    class_parameters = class_parameters[idx, class_ids, :]
                    detections[:, 6:9] = self.config.applyAnchorsTensor(class_ids, class_parameters)

                    detection_support = torch.zeros((class_parameters.shape[0], target_support.shape[1]),
                                                    device=class_parameters.device)

                    roi_features = roi_features[indices]                    
                    pass
                pass

            valid = False                
            if len(detections) > 0:
                positive_rois = detection_boxes.squeeze(0)                

                gt_class_ids = gt_class_ids.squeeze(0)
                gt_boxes = gt_boxes.squeeze(0)
                gt_masks = gt_masks.squeeze(0)
                gt_parameters = gt_parameters.squeeze(0)

                ## Compute overlaps matrix [proposals, gt_boxes]
                overlaps = bbox_overlaps(positive_rois, gt_boxes)

                ## Determine postive and negative ROIs
                roi_iou_max = torch.max(overlaps, dim=1)[0]

                ## 1. Positive ROIs are those with >= 0.5 IoU with a GT box
                if 'inference' in mode:
                    positive_roi_bool = roi_iou_max > -1
                else:
                    positive_roi_bool = roi_iou_max > 0.2
                    pass
                detections = detections[positive_roi_bool]
                detection_masks = detection_masks[positive_roi_bool]
                detection_support = detection_support[positive_roi_bool]
                roi_features = roi_features[positive_roi_bool]
                if len(detections) > 0:
                    positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

                    positive_rois = positive_rois[positive_indices.data]

                    ## Assign positive ROIs to GT boxes.
                    positive_overlaps = overlaps[positive_indices.data,:]
                    roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
                    roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data,:]
                    roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]
                    roi_gt_parameters = gt_parameters[roi_gt_box_assignment.data]
                    roi_gt_parameters = self.config.applyAnchorsTensor(roi_gt_class_ids.long(), roi_gt_parameters)
                    ## Assign positive ROIs to GT masks
                    roi_gt_masks = gt_masks[roi_gt_box_assignment.data,:,:]

                    valid_mask = positive_overlaps.max(0)[1]
                    valid_mask = (valid_mask[roi_gt_box_assignment] == torch.arange(len(roi_gt_box_assignment)).long().cuda()).long()
                    roi_indices = roi_gt_box_assignment * valid_mask + (-1) * (1 - valid_mask)

                    ## Compute mask targets
                    boxes = positive_rois
                    if self.config.USE_MINI_MASK:
                        ## Transform ROI corrdinates from normalized image space
                        ## to normalized mini-mask space.
                        y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
                        gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
                        gt_h = gt_y2 - gt_y1
                        gt_w = gt_x2 - gt_x1
                        y1 = (y1 - gt_y1) / gt_h
                        x1 = (x1 - gt_x1) / gt_w
                        y2 = (y2 - gt_y1) / gt_h
                        x2 = (x2 - gt_x1) / gt_w
                        boxes = torch.cat([y1, x1, y2, x2], dim=1)
                        pass
                    roi_gt_masks = Variable(roi_align(roi_gt_masks.unsqueeze(1),
                                                      boxes.chunk(boxes.shape[0], dim=0),
                                                      (self.config.FINAL_MASK_SHAPE[0], self.config.FINAL_MASK_SHAPE[1])).data,
                                            requires_grad=False)
                    roi_gt_masks = roi_gt_masks.squeeze(1)

                    roi_gt_masks = torch.round(roi_gt_masks)
                    valid = True
                    pass
                pass
            if not valid:
                detections = torch.FloatTensor()
                detection_masks = torch.FloatTensor()
                detection_support = torch.FloatTensor()
                roi_gt_class_ids = torch.FloatTensor()
                roi_gt_parameters = torch.FloatTensor()
                roi_gt_masks = torch.FloatTensor()
                roi_features = torch.FloatTensor()
                roi_indices = torch.LongTensor()
                if self.config.GPU_COUNT:
                    detections = detections.cuda()
                    detection_masks = detection_masks.cuda()
                    detection_support = detection_masks.cuda()
                    roi_gt_class_ids = roi_gt_class_ids.cuda()
                    roi_gt_parameters = roi_gt_parameters.cuda()
                    roi_gt_masks = roi_gt_masks.cuda()
                    roi_features = roi_features.cuda()
                    roi_indices = roi_indices.cuda()
                    pass
                pass

            # reporter = MemReporter(self)
            # reporter.report(verbose=True)

            info = [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                    target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, target_support, mrcnn_support,
                    mrcnn_support_class, detections, detection_masks, detection_support,
                    roi_gt_class_ids, roi_gt_parameters, roi_gt_masks, rpn_rois, roi_features, roi_indices]
            if return_feature_map:
                feature_map = mrcnn_feature_maps
                info.append(feature_map)
                pass

            if self.config.PREDICT_STEREO:
                info.append(depth_np)
                if self.training:
                    info.append(disp1_np)
            else:
                info.append(depth_np)
                if self.config.PREDICT_BOUNDARY:
                    info.append(boundary)
                    pass

            return info


class AnchorScores(pl.LightningModule):
    def __init__(self, options, config, profiler=PassThroughProfiler()):

        super().__init__()
        self.options = options
        self.config = config
        self.build()
        self.initialize_weights()
        self.profiler = profiler
        self.score_thresh = 0.95

        print("Loading pretrained weights ", self.options.MaskRCNNPath)
        self.load_weights(self.options.MaskRCNNPath)

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            try:
                self.load_state_dict(state_dict, strict=False)
            except:
                print('load only base model')
                try:
                    state_dict = {k: v for k, v in state_dict.items() if 'classifier.linear_class' not in k
                                  and 'classifier.linear_bbox' not in k
                                  and 'mask.conv5' not in k}
                    state = self.state_dict()
                    state.update(state_dict)
                    self.load_state_dict(state)
                except:
                    print('change input dimension')
                    state_dict = {k: v for k, v in state_dict.items() if 'classifier.linear_class' not in k
                                  and 'classifier.linear_bbox' not in k
                                  and 'mask.conv5' not in k
                                  and 'mask.conv1' not in k
                                  # and 'fpn.C1.0' not in k
                                  and 'classifier.conv1' not in k
                                  # and 'rpn.conv_shared' not in k
                                  }
                    state = self.state_dict()
                    state.update(state_dict)
                    self.load_state_dict(state)
                    pass
                pass
        else:
            raise Exception("Weight file not found")

    def build(self):
        """Build Mask R-CNN architecture.
        """

        ## Image size must be dividable by 2 multiple times
        h, w = self.config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        ## Build the shared convolutional layers.
        ## Bottom-up Layers
        ## Returns a list of the last layers of each stage, 5 in total.
        ## Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True, numInputChannels=self.config.NUM_INPUT_CHANNELS)
        # resnet = ResNet("resnet50", stage5=True, numInputChannels=config.NUM_INPUT_CHANNELS)
        C1, C2, C3, C4, C5 = resnet.stages()

        ## Top-down Layers
        ## TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256, bilinear_upsampling=self.config.BILINEAR_UPSAMPLING)

        ## Generate Anchors
        self.register_buffer('anchors',
                             torch.from_numpy(utils.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                                            self.config.RPN_ANCHOR_RATIOS,
                                                                            self.config.BACKBONE_SHAPES,
                                                                            self.config.BACKBONE_STRIDES,
                                                                            self.config.RPN_ANCHOR_STRIDE)).float())
        ## RPN
        self.rpn = RPNanchor(len(self.config.RPN_ANCHOR_RATIOS), self.config.RPN_ANCHOR_STRIDE, 256, 64)

        self.desc_dist = DescDist(64)

        ## Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for name, param in self.named_parameters():
            trainable = bool(re.fullmatch(layer_regex, name))
            if not trainable:
                param.requires_grad = False

    @staticmethod
    def nms_anchors(anchors, planes=None):
        sizes = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
        keep, num_to_keep, _ = nms.nms(anchors,
                                       sizes,
                                       overlap=0.5,
                                       top_k=400)
        keep = keep[:num_to_keep]

        anchors_nms = anchors[keep]
        planes_nms = None
        if planes is not None:
            planes_nms = planes[keep]

        return anchors_nms, planes_nms

    @staticmethod
    def find_overlaps(anchors, merge_thresh=0.4):
        iou = calc_iou_batch(anchors)

        merge_thresh = 0.4
        overlap_idxs = torch.where(torch.max(iou, iou.transpose(0, 1)) > merge_thresh)
        overlap_idxs = (overlap_idxs[0].cpu().numpy(), overlap_idxs[1].cpu().numpy())

        return overlap_idxs

    def forward(self, input):
        molded_images = input[0]

        if self.training:
            ## Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        ## Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)
        ## Note that P6 is used in RPN, but not in the classifier heads.

        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        ## Loop through pyramid layers
        layer_outputs = []  ## list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        ## Concatenate layer outputs
        ## Convert from list of lists of level outputs to list of lists
        ## of outputs across levels.
        ## e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_desc = outputs

        return [rpn_class_logits, rpn_class, rpn_desc]

    def configure_optimizers(self):
        trainable_params = []
        if self.options.trainingMode != '':
            ## Specify which layers to train, default is "all"
            layer_regex = {
                ## all layers but the backbone
                "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                ## From a specific Resnet stage and up
                "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                ## All layers
                "all": ".*",
                "classifier": "(classifier.*)|(mask.*)|(depth.*)",
            }
            assert (self.options.trainingMode in layer_regex.keys())
            layers = layer_regex[self.options.trainingMode]
            self.set_trainable(layers)
            trainable_params = [(name, param) for name, param in self.named_parameters() if bool(re.fullmatch(layers, name))]
        else:
            trainable_params = self.named_parameters()

        trainables_wo_bn = [param for name, param in trainable_params if not 'bn' in name]
        trainables_only_bn = [param for name, param in trainable_params if 'bn' in name]

        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': 0.0001},
            {'params': trainables_only_bn}], lr=self.options.LR, momentum=0.9)

        return optimizer

    def training_step(self, batch, batch_idx):
        input_pair = []
        detection_pair = []
        dicts_pair = []

        camera = batch[30][0]
        for indexOffset in [0, 13]:
            [images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth,
             extrinsics, gt_plane, gt_segmentation, plane_indices] = \
                batch[indexOffset + 0], batch[indexOffset + 1], batch[indexOffset + 2], batch[
                    indexOffset + 3], batch[indexOffset + 4], batch[indexOffset + 5], batch[
                    indexOffset + 6], batch[indexOffset + 7], batch[indexOffset + 8], batch[
                    indexOffset + 9], batch[indexOffset + 10], batch[indexOffset + 11], batch[
                    indexOffset + 12]

            input_pair.append({'image': images,
                               'image_meta': image_metas,
                               'depth': gt_depth,
                               'mask': gt_masks,
                               'bbox': gt_boxes,
                               'extrinsics': extrinsics,
                               'segmentation': gt_segmentation,
                               'class_ids': gt_class_ids,
                               'parameters': gt_parameters,
                               'plane': gt_plane,
                               'rpn_match': rpn_match,
                               'rpn_bbox': rpn_bbox,
                               'camera': camera})

        [rpn_class_logits, rpn_probs, rpn_desc] = self([input_pair[0]['image']])

        rpn_target_class = torch.where(input_pair[0]['rpn_match'] > self.score_thresh,
                                       torch.tensor(1, dtype=torch.long,
                                                    device=self.device),
                                       torch.tensor(0, dtype=torch.long,
                                                    device=self.device))
        rpn_cross_loss = F.cross_entropy(rpn_class_logits.squeeze(0), rpn_target_class.squeeze(0).squeeze(-1))

        desc_dist_cross_loss = torch.tensor([0], dtype=torch.float, device=self.device)
        # probability of being planar
        positive_idxs = torch.nonzero(rpn_target_class[0] == 1)[:, 0]
        if positive_idxs.shape[0] > 0:
            anchors_s = self.anchors[positive_idxs]
            # TOOD Hack to pass plane parameters
            planes_s = input_pair[0]['rpn_bbox'][0, positive_idxs]

            anchors_nms, planes_nms = self.nms_anchors(anchors_s, planes_s)

            overlap_idxs = self.find_overlaps(anchors_nms)

            positive_matches = []
            negative_matches = []
            for i in range(overlap_idxs[0].shape[0]):
                if overlap_idxs[0][i] != overlap_idxs[1][i]:
                    norm_dot = plane_to_plane_dot(planes_nms[overlap_idxs[0][i]],
                                                  planes_nms[overlap_idxs[1][i]])
                    # dist = plane_to_plane_dist(planes_nms[i], planes_nms[j])
                    if norm_dot > np.cos(10.0 * np.pi / 180.0):
                        positive_matches.append((overlap_idxs[0][i], overlap_idxs[1][i]))
                    else:
                        negative_matches.append((overlap_idxs[0][i], overlap_idxs[1][i]))

            neg_ratio = 2
            sel_positive_matches = []
            sel_negative_matches = []
            # sample neg_ratio / (neg_ratio + 1) negative samples and 1 / (neg_ratio + 1) positive samples
            if neg_ratio * len(positive_matches) > len(negative_matches):
                sel_positive_matches = random.sample(positive_matches, k=int(len(negative_matches) / neg_ratio))
                sel_negative_matches = negative_matches
            else:
                sel_positive_matches = positive_matches
                sel_negative_matches = random.sample(negative_matches, k=int(len(positive_matches) * neg_ratio))

            if len(sel_positive_matches) > 0 and len(sel_negative_matches) > 0:
                i_idxs_pos = [i for (i, j) in sel_positive_matches]
                j_idxs_pos = [j for (i, j) in sel_positive_matches]
                i_idxs_neg = [i for (i, j) in sel_negative_matches]
                j_idxs_neg = [j for (i, j) in sel_negative_matches]
                descs_i = torch.cat([rpn_desc[:, i_idxs_pos, :], rpn_desc[:, i_idxs_neg, :]], dim=1)
                descs_j = torch.cat([rpn_desc[:, j_idxs_pos, :], rpn_desc[:, j_idxs_neg, :]], dim=1)
                target_class = torch.cat([torch.ones((1, len(i_idxs_pos)), dtype=torch.long, device=self.device),
                                          torch.zeros((1, len(i_idxs_neg)), dtype=torch.long, device=self.device)], dim=1)

                class_logits, class_prob = self.desc_dist(descs_i, descs_j)

                desc_dist_cross_loss = F.cross_entropy(class_logits, target_class)

        total_loss = rpn_cross_loss + desc_dist_cross_loss

        return {'loss': total_loss, 'log': {'rpn_cross_loss': rpn_cross_loss,
                                            'desc_dist_cross_loss': desc_dist_cross_loss}}

    def validation_step(self, batch, batch_idx):
        input_pair = []
        detection_pair = []
        dicts_pair = []

        camera = batch[30][0]
        for indexOffset in [0, 13]:
            [images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth,
             extrinsics, gt_plane, gt_segmentation, plane_indices] = \
                batch[indexOffset + 0], batch[indexOffset + 1], batch[indexOffset + 2], batch[
                    indexOffset + 3], batch[indexOffset + 4], batch[indexOffset + 5], batch[
                    indexOffset + 6], batch[indexOffset + 7], batch[indexOffset + 8], batch[
                    indexOffset + 9], batch[indexOffset + 10], batch[indexOffset + 11], batch[
                    indexOffset + 12]

            input_pair.append({'image': images,
                               'image_meta': image_metas,
                               'depth': gt_depth,
                               'mask': gt_masks,
                               'bbox': gt_boxes,
                               'extrinsics': extrinsics,
                               'segmentation': gt_segmentation,
                               'class_ids': gt_class_ids,
                               'parameters': gt_parameters,
                               'plane': gt_plane,
                               'rpn_match': rpn_match,
                               'rpn_bbox': rpn_bbox,
                               'camera': camera})

        [rpn_class_logits, rpn_probs, rpn_desc] = self([input_pair[0]['image']])

        rpn_target_class = torch.where(input_pair[0]['rpn_match'] > self.score_thresh,
                                       torch.tensor(1, dtype=torch.long,
                                                    device=self.device),
                                       torch.tensor(0, dtype=torch.long,
                                                    device=self.device))
        rpn_cross_loss = F.cross_entropy(rpn_class_logits.squeeze(0), rpn_target_class.squeeze(0).squeeze(-1))

        positive_idxs = torch.nonzero(rpn_probs[0, :, 1] > 0.5)[:, 0]
        if positive_idxs.shape[0] > 0:
            anchors_s = self.anchors[positive_idxs]
            # TOOD Hack to pass plane parameters
            planes_s = input_pair[0]['rpn_bbox'][0, positive_idxs]

            anchors_nms, planes_nms = self.nms_anchors(anchors_s, planes_s)

            overlap_idxs = self.find_overlaps(anchors_nms)

            potential_matches = []
            for i in range(overlap_idxs[0].shape[0]):
                if overlap_idxs[0][i] != overlap_idxs[1][i]:
                    potential_matches.append((overlap_idxs[0][i], overlap_idxs[1][i]))

            djs = DisjointSet()
            for i in range(anchors_nms.shape[0]):
                djs.find(i)
            if len(potential_matches) > 0:
                i_idxs = [i for (i, j) in potential_matches]
                j_idxs = [j for (i, j) in potential_matches]
                descs_i = rpn_desc[:, i_idxs, :]
                descs_j = rpn_desc[:, j_idxs, :]

                class_logits, class_prob = self.desc_dist(descs_i, descs_j)
                for idx in range(class_prob.shape[2]):
                    if class_prob[0, 1, idx] > 0.5:
                        djs.union(i_idxs[idx], j_idxs[idx])

        # drawing
        box_image = input_pair[0]['image'].cpu().numpy()
        box_image = np.ascontiguousarray(np.transpose(box_image, axes=[0, 2, 3, 1]).squeeze(0))
        box_image = unmold_image(box_image, self.config)
        box_image_target = box_image.copy()
        h = box_image.shape[0]
        w = box_image.shape[1]

        positive_idxs_target = input_pair[0]['rpn_match'][:, :, 0] > self.score_thresh
        anchors_target = self.anchors[positive_idxs_target.squeeze(0)]
        planes_target = input_pair[0]['rpn_bbox'][positive_idxs_target]
        anchors_target_nms, planes_target_nms = self.nms_anchors(anchors_target, planes_target)
        overlap_idxs_target = self.find_overlaps(anchors_target_nms)

        djs_target = DisjointSet()
        for i in range(anchors_target_nms.shape[0]):
            djs_target.find(i)
        for i in range(overlap_idxs_target[0].shape[0]):
            if overlap_idxs_target[0][i] != overlap_idxs_target[1][i]:
                norm_dot = plane_to_plane_dot(planes_target_nms[overlap_idxs_target[0][i]],
                                              planes_target_nms[overlap_idxs_target[1][i]])
                # dist = plane_to_plane_dist(planes_target_nms[overlap_idxs_target[0][i]],
                #                               planes_target_nms[overlap_idxs_target[1][i]])t

                if norm_dot > np.cos(10.0 * np.pi / 180.0):
                    djs_target.union(overlap_idxs_target[0][i], overlap_idxs_target[1][i])

        color_map = ColorPalette(max(anchors_nms.shape[0], anchors_target_nms.shape[0])).getColorMap()
        for idx, anchor in enumerate(anchors_target_nms):
            # y, x
            pt1 = [anchor[0], anchor[1] + 80]
            pt2 = [anchor[2], anchor[3] + 80]

            plane_set = djs_target.find(idx)

            cv2.rectangle(box_image_target, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (int(color_map[plane_set, 0]),
                                                                                 int(color_map[plane_set, 1]),
                                                                                 int(color_map[plane_set, 2])), 2)

        if positive_idxs.shape[0] > 0:
            for idx, anchor in enumerate(anchors_nms):
                # y, x
                pt1 = [anchor[0], anchor[1] + 80]
                pt2 = [anchor[2], anchor[3] + 80]

                plane_set = djs.find(idx)

                cv2.rectangle(box_image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (int(color_map[plane_set, 0]),
                                                                              int(color_map[plane_set, 1]),
                                                                              int(color_map[plane_set, 2])), 2)

        self.logger.experiment.add_image('val/boxes', box_image, dataformats='HWC', global_step=self.global_step)
        self.logger.experiment.add_image('val/boxes_target', box_image_target, dataformats='HWC', global_step=self.global_step)

        return {'loss': rpn_cross_loss, 'log': {'val_loss': rpn_cross_loss}}


############################################################
#  Data Formatting
############################################################

def mold_inputs(config, images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matricies [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matricies:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        ## Resize image to fit the model expected size
        ## TODO: move resizing to mold_image()
        molded_image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        molded_image = mold_image(molded_image, config)
        ## Build image_meta
        image_meta = compose_image_meta(
            0, image.shape, window,
            np.zeros([config.NUM_CLASSES], dtype=np.int32))
        ## Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    ## Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows

def unmold_detections(config, detections, mrcnn_mask, image_shape, window, debug=False):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)]
    mrcnn_mask: [N, height, width, num_classes]
    image_shape: [height, width, depth] Original size of the image before resizing
    window: [y1, x1, y2, x2] Box in the image where the real image is
            excluding the padding.

    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    """
    ## How many detections do we have?
    ## Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    ## Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    parameters = detections[:N, 6:]
    if config.GLOBAL_MASK:
        masks = mrcnn_mask[np.arange(N), :, :, 0]
    else:
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]
        pass

    
    ## Compute scale and shift to translate coordinates to image domain.
    h_scale = image_shape[0] / (window[2] - window[0])
    w_scale = image_shape[1] / (window[3] - window[1])
    scale = min(h_scale, w_scale)
    shift = window[:2]  ## y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

    ## Translate bounding boxes to image domain
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
    
    if debug:
        print(masks.shape, boxes.shape)
        for maskIndex, mask in enumerate(masks):
            print(maskIndex, boxes[maskIndex].astype(np.int32))
            cv2.imwrite('test/local_mask_' + str(maskIndex) + '.png', (mask * 255).astype(np.uint8))
            continue
    
    ## Filter out detections with zero area. Often only happens in early
    ## stages of training when the network weights are still a bit random.
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        parameters = np.delete(parameters, exclude_ix, axis=0)

        N = class_ids.shape[0]

    ## Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        ## Convert neural network mask to full size mask
        full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty((0,) + masks.shape[1:3])

    if debug:
        print(full_masks.shape)
        for maskIndex in range(full_masks.shape[2]):
            cv2.imwrite('test/full_mask_' + str(maskIndex) + '.png', (full_masks[:, :, maskIndex] * 255).astype(np.uint8))
            continue
        pass
    return boxes, class_ids, scores, full_masks, parameters
    
def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            ## size=1
        list(image_shape) +     ## size=3
        list(window) +          ## size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  ## size=num_classes
    )
    return meta


## Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   ## (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
