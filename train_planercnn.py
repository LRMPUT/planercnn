"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import cv2
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm
import numpy as np
import sys
import shutil

from models.model import *
from models.refinement_net import *
from models.modules import *
from datasets.plane_stereo_dataset import *
from datasets.scenenet_rgbd_stereo_dataset import *

from utils import *
from visualize_utils import *
from evaluate_utils import *
from options import parse_args
from config import PlaneConfig

# from pytorch_memlab import MemReporter


def train(options):
    cv2.setNumThreads(0)
    # torch.manual_seed(13)
    # np.random.seed(13)

    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    config = PlaneConfig(options)

    summary_dir = 'runs/train'
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)
    writer = SummaryWriter(summary_dir)

    # dataset = ScenenetRgbdDataset(options, config, split='train', random=False, writer=writer)
    dataset = ScenenetRgbdDataset(options, config, split='train', random=False)
    # dataset_test = ScenenetRgbdDataset(options, config, split='test', random=False, writer=writer)
    # dataset = PlaneDataset(options, config, split='train', random=False)
    # dataset_test = PlaneDataset(options, config, split='test', random=False)

    print('the number of images', len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

    model = MaskRCNN(config)
    # refine_model = RefineModel(options)
    model.cuda()
    model.train()
    # refine_model.cuda()
    # refine_model.train()

    # reporter = MemReporter(model)

    if options.restore == 1:
        ## Resume training
        print('restore')
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
        # refine_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_refine.pth'))
    elif options.restore == 2:
        ## Train upon Mask R-CNN weights
        model_path = options.MaskRCNNPath
        print("Loading pretrained weights ", model_path)
        model.load_weights(model_path)
        pass
    
    if options.trainingMode != '':
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
        assert(options.trainingMode in layer_regex.keys())
        layers = layer_regex[options.trainingMode]
        model.set_trainable(layers)
        pass

    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]

    # model_names = [name for name, param in model.named_parameters()]
    # for name, param in refine_model.named_parameters():
    #     assert(name not in model_names)
    #     continue
    optimizer = optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn},
        # {'params': refine_model.parameters()}
    ], lr=options.LR, momentum=0.9)

    # if 'refine_only' in options.suffix:
    #     optimizer = optim.Adam(refine_model.parameters(), lr = options.LR)
    #     pass
    
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim.pth'))        
        pass

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    for epoch in range(options.numEpochs):
        epoch_losses = []
        data_iterator = tqdm(dataloader, total=len(dataset))

        optimizer.zero_grad()

        for sampleIndex, sample in enumerate(data_iterator):
        # for sampleIndex, sample in enumerate(dataloader):
            losses = []

            input_pair = []
            detection_pair = []
            dicts_pair = []

            camera = sample[30][0].cuda()                
            for indexOffset in [0, 13]:
                images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, gt_plane, gt_segmentation, plane_indices = \
                sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[
                    indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[
                    indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[
                    indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda(), sample[
                    indexOffset + 12].cuda()

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

            if config.PREDICT_STEREO:
                [rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                 target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, target_support, mrcnn_support,
                 detections, detection_masks, detection_support, detection_gt_class_ids, detection_gt_parameters, detection_gt_masks,
                 rpn_rois, roi_features, roi_indices, feature_map, depth_np_pred, disp1_np_pred] = model.predict(
                        [input_pair[0]['image'], input_pair[0]['image_meta'], input_pair[0]['class_ids'],
                         input_pair[0]['bbox'], input_pair[0]['mask'], input_pair[0]['parameters'],
                         input_pair[0]['camera'], input_pair[0]['depth'],
                         input_pair[1]['image']],
                        mode='training_detection', use_nms=2, use_refinement=False,
                        return_feature_map=True, writer=writer)
            else:
                [rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                 target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, target_support, mrcnn_support,
                 detections, detection_masks, detection_support, detection_gt_class_ids, detection_gt_parameters, detection_gt_masks,
                 rpn_rois, roi_features, roi_indices, feature_map, depth_np_pred] = model.predict(
                        [input_pair[0]['image'], input_pair[0]['image_meta'], input_pair[0]['class_ids'],
                         input_pair[0]['bbox'], input_pair[0]['mask'], input_pair[0]['parameters'],
                         input_pair[0]['camera'], input_pair[0]['depth']],
                        mode='training_detection', use_nms=2, use_refinement=False,
                        return_feature_map=True)

            [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss,
             mrcnn_parameter_loss, mrcnn_support_loss] = compute_losses(
                    config, input_pair[0]['rpn_match'], input_pair[0]['rpn_bbox'], rpn_class_logits, rpn_pred_bbox,
                    target_class_ids, mrcnn_class_logits,
                    target_deltas, mrcnn_bbox,
                    target_mask, mrcnn_mask,
                    target_parameters, mrcnn_parameters,
                    target_support, mrcnn_support)

            losses += [rpn_class_loss + rpn_bbox_loss + \
                        mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss]
            # losses += [rpn_class_loss + rpn_bbox_loss + \
            #            mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_support_loss]
            # losses += [rpn_class_loss + rpn_bbox_loss + \
            #            mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_support_loss + mrcnn_support_class_loss]
            # losses += [rpn_class_loss + rpn_bbox_loss +
            #            mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_parameter_loss]
            # losses += [rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss]
            if writer is not None and sampleIndex % 100 == 0:
                writer.add_scalar('maskrcnn_loss', losses[-1], global_step=epoch * len(dataset) + sampleIndex)
                writer.add_scalar('rpn_class_loss', rpn_class_loss, global_step=epoch * len(dataset) + sampleIndex)
                writer.add_scalar('rpn_bbox_loss', rpn_bbox_loss, global_step=epoch * len(dataset) + sampleIndex)
                writer.add_scalar('mrcnn_class_loss', mrcnn_class_loss, global_step=epoch * len(dataset) + sampleIndex)
                writer.add_scalar('mrcnn_bbox_loss', mrcnn_bbox_loss, global_step=epoch * len(dataset) + sampleIndex)
                writer.add_scalar('mrcnn_mask_loss', mrcnn_mask_loss, global_step=epoch * len(dataset) + sampleIndex)
                writer.add_scalar('mrcnn_parameter_loss', mrcnn_parameter_loss, global_step=epoch * len(dataset) + sampleIndex)
                writer.add_scalar('mrcnn_support_loss', mrcnn_support_loss, global_step=epoch * len(dataset) + sampleIndex)
                # writer.add_scalar('mrcnn_support_class_loss', mrcnn_support_class_loss, global_step=epoch * len(dataset) + sampleIndex)

            gt_depth = input_pair[0]['depth']
            if config.PREDICT_NORMAL_NP:
                normal_np_pred = depth_np_pred[0, 1:]
                depth_np_pred = depth_np_pred[:, 0]
                gt_normal = gt_depth[0, 1:]
                gt_depth = gt_depth[:, 0]
                depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                normal_np_loss = l2LossMask(normal_np_pred[:, 80:560], gt_normal[:, 80:560], (torch.norm(gt_normal[:, 80:560], dim=0) > 1e-4).float())
                losses.append(depth_np_loss)
                losses.append(normal_np_loss)
            else:
                if config.PREDICT_STEREO:
                    fx = input_pair[0]['camera'][0]
                    gt_disp = fx * config.BASELINE / torch.clamp(gt_depth, min=1.0e-4)
                    mask = gt_disp < config.MAXDISP
                    # disp_np_loss = 0.5 * F.smooth_l1_loss(disp1_np_pred[mask], gt_disp[mask], size_average=True) +\
                    #                0.7 * F.smooth_l1_loss(disp2_np_pred[mask], gt_disp[mask], size_average=True) +\
                    #                F.smooth_l1_loss(disp3_np_pred[mask], gt_disp[mask], size_average=True)
                    disp_np_loss = F.smooth_l1_loss(disp1_np_pred[mask], gt_disp[mask], reduction='mean')
                    losses.append(disp_np_loss * options.dispWeight)

                    normal_np_pred = None

                    if writer is not None and sampleIndex % 100 == 0:
                        writer.add_scalar('disp/disp_np_loss', losses[-1], global_step=epoch * len(dataset) + sampleIndex)

                    #     disp_scale = 192.0
                    #     writer.add_image('disp/image_left', unmold_image_torch(input_pair[0]['image'].squeeze(0), config), dataformats='CHW')
                    #     writer.add_image('disp/image_right', unmold_image_torch(input_pair[1]['image'].squeeze(0), config), dataformats='CHW')
                    #     # up to 15 m
                    #     writer.add_image('disp/gt_depth', gt_depth.squeeze(0) / 15.0, dataformats='HW')
                    #     writer.add_image('disp/gt_disp', gt_disp.squeeze(0) / disp_scale, dataformats='HW')
                    #     writer.add_image('disp/disp1', disp1_np_pred.squeeze(0) / disp_scale, dataformats='HW')
                    #     # writer.add_image('disp/disp2', disp2_np_pred.squeeze(0) / disp_scale, dataformats='HW')
                    #     # writer.add_image('disp/disp3', disp3_np_pred.squeeze(0) / disp_scale, dataformats='HW')
                    #     writer.add_image('disp/mask', mask.squeeze(0), dataformats='HW')
                    #     writer.add_image('disp/error',
                    #                      torch.clamp(F.smooth_l1_loss(disp1_np_pred.squeeze(0),
                    #                                                   gt_disp.squeeze(0),
                    #                                                   reduction='none'), max=10.0) / 10.0,
                    #                      dataformats='HW')
                else:
                    depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                    losses.append(depth_np_loss)
                    # if writer is not None and sampleIndex % 100 == 0:
                    #     writer.add_scalar('depth_np_loss', losses[-1], global_step=epoch * len(dataset) + sampleIndex)
                    normal_np_pred = None
                    pass

            if len(detections) > 0:
                # fx = camera[0]
                # fy = camera[1]
                # cx = camera[2]
                # cy = camera[3]
                # w = camera[4]
                # h = camera[5]
                #
                # rois = detections[:, :4].clone()
                # # y
                # rois[:, [0, 2]] /= depth_np_pred.shape[1]
                # # x
                # rois[:, [1, 3]] /= depth_np_pred.shape[2]
                # ranges_rois = get_support_ranges(camera, rois)
                # roi_gt_planes = detection_gt_parameters / detection_gt_parameters.norm(dim=1, keepdim=True).square()
                # support_gt = (roi_gt_planes.view(-1, 3, 1) * ranges_rois).sum(dim=1) * (config.BASELINE * fx)

                detections, detection_masks = unmoldDetections(config, camera, detections,
                                                                 detection_masks,
                                                                 detection_support,
                                                                 # support_gt,
                                                                 depth_np_pred,
                                                                 normal_np_pred, debug=False)
                if 'refine_only' in options.suffix:
                    detections, detection_masks = detections.detach(), detection_masks.detach()
                    pass
                XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(config, camera, detections, detection_masks, depth_np_pred, return_individual=True)
                detection_mask = detection_mask.unsqueeze(0)
            else:
                XYZ_pred = torch.zeros((3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                detection_mask = torch.zeros((1, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                plane_XYZ = torch.zeros((1, 3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                pass

            # input_pair.append({
            #                       'image': images,
            #                       'depth': gt_depth,
            #                       'mask': gt_masks,
            #                       'bbox': gt_boxes,
            #                       'extrinsics': extrinsics,
            #                       'segmentation': gt_segmentation,
            #                       'parameters': detection_gt_parameters,
            #                       'plane': gt_plane,
            #                       'camera': camera})
            detection_pair.append({
                                      'XYZ': XYZ_pred,
                                      'depth': XYZ_pred[1:2],
                                      'mask': detection_mask,
                                      'detection': detections,
                                      'masks': detection_masks,
                                      'feature_map': feature_map[0],
                                      'plane_XYZ': plane_XYZ,
                                      'depth_np': depth_np_pred})

            if 'depth' in options.suffix:
                ## Apply supervision on reconstructed depthmap (not used currently)
                if len(detections) > 0:
                    background_mask = torch.clamp(1 - detection_masks.sum(0, keepdim=True), min=0)
                    all_masks = torch.cat([background_mask, detection_masks], dim=0)

                    all_masks = all_masks / all_masks.sum(0, keepdim=True)
                    all_depths = torch.cat([depth_np_pred, plane_XYZ[:, 1]], dim=0)

                    depth_loss = l1LossMask(
                        torch.sum(torch.abs(all_depths[:, 80:560] - gt_depth[:, 80:560]) * all_masks[:, 80:560],
                                  dim=0), torch.zeros(config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM).cuda(),
                        (gt_depth[0, 80:560] > 1e-4).float())
                else:
                    depth_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                    pass
                losses.append(depth_loss)
                if writer is not None and sampleIndex % 100 == 0:
                    writer.add_scalar('depth_loss', losses[-1], global_step=epoch * len(dataset) + sampleIndex)

            # if (len(detection_pair[0]['detection']) > 0 and len(detection_pair[0]['detection']) < 30) and 'refine' in options.suffix:
            #     ## Use refinement network
            #     pose = sample[26][0].cuda()
            #     pose = torch.cat([pose[0:3], pose[3:6] * pose[6]], dim=0)
            #     pose_gt = torch.cat([pose[0:1], -pose[2:3], pose[1:2], pose[3:4], -pose[5:6], pose[4:5]], dim=0).unsqueeze(0)
            #     camera = camera.unsqueeze(0)
            #     c = 0
            #     detection_dict, input_dict = detection_pair[c], input_pair[c]
            #     detections = detection_dict['detection']
            #     detection_masks = detection_dict['masks']
            #     image = (input_dict['image'] + config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
            #     image_2 = (input_pair[1 - c]['image'] + config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
            #     depth_gt = input_dict['depth'].unsqueeze(1)
            #
            #     masks_inp = torch.cat([detection_masks.unsqueeze(1), detection_dict['plane_XYZ']], dim=1)
            #
            #     segmentation = input_dict['segmentation']
            #     plane_depth = detection_dict['depth']
            #     depth_np = detection_dict['depth_np']
            #     if 'large' not in options.suffix:
            #         ## Use 256x192 instead of 640x480
            #         detection_masks = torch.nn.functional.interpolate(detection_masks[:, 80:560].unsqueeze(1), size=(192, 256), mode='nearest').squeeze(1)
            #         image = torch.nn.functional.interpolate(image[:, :, 80:560], size=(192, 256), mode='bilinear')
            #         image_2 = torch.nn.functional.interpolate(image_2[:, :, 80:560], size=(192, 256), mode='bilinear')
            #         masks_inp = torch.nn.functional.interpolate(masks_inp[:, :, 80:560], size=(192, 256), mode='bilinear')
            #         depth_gt = torch.nn.functional.interpolate(depth_gt[:, :, 80:560], size=(192, 256), mode='nearest')
            #         segmentation = torch.nn.functional.interpolate(segmentation[:, 80:560].unsqueeze(1).float(), size=(192, 256), mode='nearest').squeeze().long()
            #         plane_depth = torch.nn.functional.interpolate(plane_depth[:, 80:560].unsqueeze(1).float(), size=(192, 256), mode='bilinear').squeeze(1)
            #         depth_np = torch.nn.functional.interpolate(depth_np[:, 80:560].unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1)
            #     else:
            #         detection_masks = detection_masks[:, 80:560]
            #         image = image[:, :, 80:560]
            #         image_2 = image_2[:, :, 80:560]
            #         masks_inp = masks_inp[:, :, 80:560]
            #         depth_gt = depth_gt[:, :, 80:560]
            #         segmentation = segmentation[:, 80:560]
            #         plane_depth = plane_depth[:, 80:560]
            #         depth_np = depth_np[:, 80:560]
            #         pass
            #
            #     depth_inv = invertDepth(depth_gt)
            #     depth_inv_small = depth_inv[:, :, ::4, ::4].contiguous()
            #
            #     ## Generate supervision target for the refinement network
            #     segmentation_one_hot = (segmentation == torch.arange(segmentation.max() + 1).cuda().view((-1, 1, 1, 1))).long()
            #     intersection = (torch.round(detection_masks).long() * segmentation_one_hot).sum(-1).sum(-1)
            #     max_intersection, segments_gt = intersection.max(0)
            #     mapping = intersection.max(1)[1]
            #     detection_areas = detection_masks.sum(-1).sum(-1)
            #     valid_mask = (mapping[segments_gt] == torch.arange(len(segments_gt)).cuda()).float()
            #
            #     masks_gt_large = (segmentation == segments_gt.view((-1, 1, 1))).float()
            #     masks_gt_small = masks_gt_large[:, ::4, ::4]
            #     planes_gt = input_dict['plane'][0][segments_gt]
            #
            #     ## Run the refinement network
            #     results = refine_model(image, image_2, camera, masks_inp, detection_dict['detection'][:, 6:9], plane_depth, depth_np)
            #
            #     plane_depth_loss = torch.zeros(1).cuda()
            #     depth_loss = torch.zeros(1).cuda()
            #     plane_loss = torch.zeros(1).cuda()
            #     mask_loss = torch.zeros(1).cuda()
            #     flow_loss = torch.zeros(1).cuda()
            #     flow_confidence_loss = torch.zeros(1).cuda()
            #     pose_loss = torch.zeros(1).cuda()
            #     for resultIndex, result in enumerate(results[1:]):
            #         if 'mask' in result:
            #             masks_pred = result['mask'][:, 0]
            #             if masks_pred.shape[-1] == masks_gt_large.shape[-1]:
            #                 masks_gt = masks_gt_large
            #             else:
            #                 masks_gt = masks_gt_small
            #                 pass
            #
            #             all_masks_gt = torch.cat([1 - masks_gt.max(dim=0, keepdim=True)[0], masks_gt], dim=0)
            #             segmentation = all_masks_gt.max(0)[1].view(-1)
            #             masks_logits = masks_pred.squeeze(1).transpose(0, 1).transpose(1, 2).contiguous().view((segmentation.shape[0], -1))
            #             detection_areas = all_masks_gt.sum(-1).sum(-1)
            #             detection_weight = detection_areas / detection_areas.sum()
            #             detection_weight = -torch.log(torch.clamp(detection_weight, min=1e-4, max=1 - 1e-4))
            #             if 'weight' in options.suffix:
            #                 mask_loss += torch.nn.functional.cross_entropy(masks_logits, segmentation, weight=detection_weight)
            #             else:
            #                 mask_loss += torch.nn.functional.cross_entropy(masks_logits, segmentation, weight=torch.cat([torch.ones(1).cuda(), valid_mask], dim=0))
            #                 pass
            #             masks_pred = (masks_pred.max(0, keepdim=True)[1] == torch.arange(len(masks_pred)).cuda().long().view((-1, 1, 1))).float()[1:]
            #             pass
            #         continue
            #     losses += [mask_loss + depth_loss + plane_depth_loss + plane_loss]
            #     if writer is not None and sampleIndex % 100 == 0:
            #         writer.add_scalar('detection_loss', losses[-1], global_step=epoch * len(dataset) + sampleIndex)
            #
            #     masks = results[-1]['mask'].squeeze(1)
            #     all_masks = torch.softmax(masks, dim=0)
            #     masks_small = all_masks[1:]
            #     all_masks = torch.nn.functional.interpolate(all_masks.unsqueeze(1), size=(480, 640), mode='bilinear').squeeze(1)
            #     all_masks = (all_masks.max(0, keepdim=True)[1] == torch.arange(len(all_masks)).cuda().long().view((-1, 1, 1))).float()
            #     masks = all_masks[1:]
            #     detection_masks = torch.zeros(detection_dict['masks'].shape).cuda()
            #     detection_masks[:, 80:560] = masks
            #     detection_dict['masks'] = detection_masks
            #     results[-1]['mask'] = masks_small
            #
            #     camera = camera.squeeze(0)
            #
            #     if 'refine_after' in options.suffix:
            #         ## Build the warping loss upon refined results
            #         XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(config, camera, detections, detection_masks, detection_dict['depth_np'], return_individual=True)
            #         detection_dict['XYZ'] = XYZ_pred
            #         pass
            # else:
            #     losses += [torch.zeros(1).cuda()]
            #     pass

            ## The warping loss
            # for c in range(1, 2):
            #     if 'warping' not in options.suffix:
            #         break
            #
            #     detection_dict = detection_pair[1 - c]
            #     neighbor_info = torch.cat([detection_dict['XYZ'], detection_dict['mask'], input_pair[1 - c]['image'][0]], dim=0).unsqueeze(0)
            #     warped_info, valid_mask = warpModuleDepth(config, camera, input_pair[c]['depth'][0], neighbor_info, input_pair[c]['extrinsics'][0], input_pair[1 - c]['extrinsics'][0], width=config.IMAGE_MAX_DIM, height=config.IMAGE_MIN_DIM)
            #
            #     XYZ = warped_info[:3].view((3, -1))
            #     XYZ = torch.cat([XYZ, torch.ones((1, int(XYZ.shape[1]))).cuda()], dim=0)
            #     transformed_XYZ = torch.matmul(input_pair[c]['extrinsics'][0], torch.matmul(input_pair[1 - c]['extrinsics'][0].inverse(), XYZ))
            #     transformed_XYZ = transformed_XYZ[:3].view(detection_dict['XYZ'].shape)
            #     warped_depth = transformed_XYZ[1:2]
            #     warped_images = warped_info[4:7].unsqueeze(0)
            #     warped_mask = warped_info[3]
            #
            #     with torch.no_grad():
            #         valid_mask = valid_mask * (input_pair[c]['depth'] > 1e-4).float()
            #         pass
            #
            #     warped_depth_loss = l1LossMask(warped_depth, input_pair[c]['depth'], valid_mask)
            #     losses += [warped_depth_loss]
            #
            #     if 'warping1' in options.suffix or 'warping3' in options.suffix:
            #         warped_mask_loss = l1LossMask(warped_mask, (input_pair[c]['segmentation'] >= 0).float(), valid_mask)
            #         losses += [warped_mask_loss]
            #         pass
            #
            #     if 'warping2' in options.suffix or 'warping3' in options.suffix:
            #         warped_image_loss = l1NormLossMask(warped_images, input_pair[c]['image'], dim=1, valid_mask=valid_mask)
            #         losses += [warped_image_loss]
            #         pass
            #
            #     input_pair[c]['warped_depth'] = (warped_depth * valid_mask + (1 - valid_mask) * 10).squeeze()
            #     continue
            loss = sum(losses)

            if torch.isnan(loss).any():
                print(rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_support_loss)
                exit(-1)
            # try:
            #     losses = [l.data.item() for l in losses]
            # except ValueError as e:
            #     print(e)
            #     print(rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_support_loss)

            losses = [l.data.item() for l in losses]

            epoch_losses.append(losses)
            status = str(epoch + 1) + ' loss: '
            for l in losses:
                status += '%0.5f '% l
                continue


            # sys.stdout.write('\r ' + str(sampleIndex) + ' ' + status)
            # sys.stdout.flush()
            
            data_iterator.set_description(status)

            loss.backward()

            # reporter.report()

            if (sampleIndex + 1) % options.batchSize == 0:
                optimizer.step()
                optimizer.zero_grad()
                pass

            if sampleIndex % 500 < options.batchSize or options.visualizeMode == 'debug':
                ## Visualize intermediate results
                visualizeBatchPair(options, config, input_pair, detection_pair, indexOffset=sampleIndex % 500, writer=writer)
                # if (len(detection_pair[0]['detection']) > 0 and len(detection_pair[0]['detection']) < 30) and 'refine' in options.suffix:
                #     visualizeBatchRefinement(options, config, input_pair[0], [{'mask': masks_gt, 'plane': planes_gt}, ] + results, indexOffset=sampleIndex % 500, concise=True)
                #     pass
                if options.visualizeMode == 'debug' and sampleIndex % 500 >= options.batchSize - 1:
                    exit(1)
                    pass
                pass

            if (sampleIndex + 1) % options.numTrainingImages == 0:
                ## Save models
                print('loss', np.array(epoch_losses).mean(0))
                torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint.pth')
                # torch.save(refine_model.state_dict(), options.checkpoint_dir + '/checkpoint_refine.pth')
                torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim.pth')
                pass
            continue
        # scheduler.step()
        continue
    return


if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'planercnn'

    args.keyname += '_' + args.anchorType
    if args.dataset != '':
        args.keyname += '_' + args.dataset
        pass
    if args.trainingMode != 'all':
        args.keyname += '_' + args.trainingMode
        pass
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = args.dataFolder + '/test/' + args.keyname

    if False:
        writeHTML(args.test_dir, ['image_0', 'segmentation_0', 'depth_0', 'depth_0_detection', 'depth_0_detection_ori'], labels=['input', 'segmentation', 'gt', 'before', 'after'], numImages=20, image_width=160, convertToImage=True)
        exit(1)
        
    os.system('rm ' + args.test_dir + '/*.png')
    print('keyname=%s task=%s started'%(args.keyname, args.task))

    train(args)

