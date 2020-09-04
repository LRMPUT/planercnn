"""
Copyright Jan Wietrzykowski 2020
"""

import numpy as np
import cv2
import sys
import os
import torch
import torchvision
import nms
from joblib import Parallel, delayed
from disjoint_set import DisjointSet

from options import parse_args
from config import PlaneConfig
import utils

ROOT_FOLDER = "/mnt/data/datasets/JW/scenenet_rgbd/scenes/"


def plane_to_plane_dist(plane1, plane2):
    plane1_d = 1.0 / plane1.norm()
    p1 = plane1 * plane1_d * plane1_d
    # d = (p1.dot(plane2) - 1.0) / plane2.norm()
    d = (p1.dot(plane2) - 1.0)

    return d


def plane_to_plane_dot(plane1, plane2):
    return plane1.dot(plane2) / (plane1.norm() * plane2.norm())


def calc_iou(box1, box2):
    y1_i = max(box1[0], box2[0])
    y2_i = min(box1[2], box2[2])
    x1_i = max(box1[1], box2[1])
    x2_i = min(box1[3], box2[3])

    area_i = torch.max(y2_i - y1_i, torch.tensor(0.0, dtype=torch.float)) *\
             torch.max(x2_i - x1_i, torch.tensor(0.0, dtype=torch.float))
    area_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # iou = area_i / (area_1 + area_2 - area_i)
    iou = area_i / area_2

    if iou > 1:
        print('iou = ', iou)

    return iou


def remove_nms(anchors, scores, planes, iou_thresh=0.3, score_thresh=0.9):
    score_mask = scores > score_thresh
    score_idxs = torch.where(score_mask)[0]
    anchors_s = anchors[score_mask]
    scores_s = scores[score_mask]
    planes_s = planes[score_mask]

    sizes_s = (anchors_s[:, 2] - anchors_s[:, 0]) * (anchors_s[:, 3] - anchors_s[:, 1])
    # sort according to size and then score
    # sort_idxs = torch.argsort(-(sizes_s * 2 + scores_s))
    sort_idxs = torch.argsort(-(sizes_s * scores_s))

    # max_idx = 0
    # for i in range(anchors.shape[0]):
    #     if scores[sort_idxs[i]] < 0.9:
    #         max_idx = i
    #         break

    print('pytorch nms on %d anchors' % score_idxs.shape[0])
    # keep_torch = torchvision.ops.nms(anchors_s, (sizes_s * scores_s), iou_thresh)
    keep_torch, num_to_keep, _ = nms.nms(anchors_s.cuda(), (sizes_s * scores_s).cuda(), overlap=iou_thresh, top_k=400)
    keep_torch = keep_torch[:num_to_keep].cpu()
    print('keep_torch.shape = ', keep_torch.shape)
    keep_idxs_all = score_idxs[keep_torch]

    # print('nms on %d anchors' % score_idxs.shape[0])
    # keep_mask = torch.full([anchors_s.shape[0]], fill_value=True, dtype=torch.bool)
    # for i in range(anchors_s.shape[0]):
    #     if i % 1000 == 0:
    #         print('i = ', i)
    #     if keep_mask[i]:
    #         for j in range(i + 1, anchors_s.shape[0]):
    #             if keep_mask[j]:
    #                 iou = calc_iou(anchors_s[sort_idxs[i]], anchors_s[sort_idxs[j]])
    #                 if iou > iou_thresh:
    #                     # norm_dot = plane_to_plane_dot(planes_s[sort_idxs[i]], planes_s[sort_idxs[j]])
    #                     # if norm_dot > torch.tensor(10.0 * np.pi / 180.0).cos():
    #                     #     dist = plane_to_plane_dist(planes_s[sort_idxs[i]], planes_s[sort_idxs[j]])
    #                     #     if dist.abs() < 0.1:
    #                     keep_mask[j] = False

    # nms_anchors = anchors[keep_mask]
    # nms_scores = scores[keep_mask]
    # keep_idxs = sort_idxs[keep_mask]
    # keep_idxs_all = score_idxs[keep_idxs]

    # keep_idxs_all = torch.zeros_like(keep_idxs)
    # for keep_idx in keep_idxs:
    #     keep_idxs_all = score_idxs[keep_idx]

    return keep_idxs_all


def comp_score(idx, cur_points, scores, masks, planes):
    if idx % 1000 == 0:
        print('idx = ', idx)

    cur_points = cur_points.reshape((3, -1)).transpose(0, 1).clone()
    valid_mask = torch.norm(cur_points, dim=-1) > 1e-3
    valid_points = cur_points[valid_mask]
    plane = None
    if valid_points.shape[0] > 0:
        mask_valid, plane = utils.fit_plane_ransac(valid_points.reshape(-1, 3))
        num_inliers = mask_valid.sum()
    else:
        num_inliers = torch.zeros(1)
    # print('num_inliners = ', num_inliers, ', num points = ', cur_points.shape[0])
    cur_score = num_inliers.float() / cur_points.shape[0]
    # print('cur_score = ', cur_score)
    scores[idx] = cur_score
    if plane is not None:
        planes[idx] = plane
        mask = torch.zeros(masks[idx].shape, dtype=torch.bool)
        mask[valid_mask.reshape(masks[idx].shape)] = mask_valid
        masks[idx][mask] = 255
        # print(masks[idx])


def comp_score_np(idx, cur_points, scores, masks, planes):
    if idx % 1000 == 0:
        print('idx = ', idx)

    cur_points = cur_points.reshape((3, -1)).transpose(0, 1).clone()
    valid_mask = torch.norm(cur_points, dim=-1) > 1e-3
    valid_points = cur_points[valid_mask]
    plane = None
    if valid_points.shape[0] > 0:
        mask_valid, plane = utils.fit_plane_ransac(valid_points.reshape(-1, 3))
        num_inliers = mask_valid.sum()
    else:
        num_inliers = torch.zeros(1)
    # print('num_inliners = ', num_inliers, ', num points = ', cur_points.shape[0])
    cur_score = num_inliers.float() / cur_points.shape[0]
    # print('cur_score = ', cur_score)
    scores[idx] = cur_score
    if plane is not None:
        planes[idx] = plane
        mask = torch.zeros(masks[idx].shape, dtype=torch.bool)
        mask[valid_mask.reshape(masks[idx].shape)] = mask_valid
        masks[idx][mask] = 255
        # print(masks[idx])


def test_planes(scene_id, config):
    images_list = sorted(os.listdir(os.path.join(ROOT_FOLDER, scene_id, 'frames', 'color_left')))

    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)
    anchors = torch.from_numpy(anchors).float()

    K = np.array([[554.256, 0.0, 320.0],
                  [0.0, 579.411, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float)
    for im_file in images_list[0:1]:
        print(im_file)

        im = cv2.imread(os.path.join(ROOT_FOLDER, scene_id, 'frames', 'color_left', im_file))
        depth = cv2.imread(os.path.join(ROOT_FOLDER, scene_id, 'frames', 'depth_left', im_file.replace('.jpg', '.png')),
                           cv2.IMREAD_ANYDEPTH) / 1000.0

        points = utils.calc_points_depth(depth, K)
        points = np.expand_dims(np.transpose(points, axes=(2, 0, 1)), axis=0)
        points = torch.from_numpy(points).float()
        anchor_points = torchvision.ops.roi_align(points,
                                                  [anchors],
                                                  (config.MASK_SHAPE[0], config.MASK_SHAPE[1]))
        scores = torch.zeros(anchor_points.shape[0], dtype=torch.float)
        planes = torch.zeros([anchor_points.shape[0], 3], dtype=torch.float)
        masks = torch.zeros([anchor_points.shape[0], config.MASK_SHAPE[0], config.MASK_SHAPE[1]], dtype=torch.uint8)

        # Parallel(n_jobs=4, require='sharedmem')(
        #         delayed(comp_score)(idx, anchor_points[idx], scores, masks, planes) for idx in range(anchor_points.shape[0]))

        # for idx in range(anchor_points.shape[0]):
        #     comp_score(idx, anchor_points[idx], scores, masks, planes)
        #
        # with open('test/scores_' + scene_id + ' ' + im_file.replace('.jpg', '') + '.pt', 'wb') as f:
        #     torch.save(scores, f)
        # with open('test/planes_' + scene_id + ' ' + im_file.replace('.jpg', '') + '.pt', 'wb') as f:
        #     torch.save(planes, f)
        # with open('test/masks_' + scene_id + ' ' + im_file.replace('.jpg', '') + '.pt', 'wb') as f:
        #     torch.save(masks, f)

        with open('test/scores_' + scene_id + ' ' + im_file.replace('.jpg', '') + '.pt', 'rb') as f:
            scores = torch.load(f)
        with open('test/planes_' + scene_id + ' ' + im_file.replace('.jpg', '') + '.pt', 'rb') as f:
            planes = torch.load(f)
        with open('test/masks_' + scene_id + ' ' + im_file.replace('.jpg', '') + '.pt', 'rb') as f:
            masks = torch.load(f)

        # keep = torchvision.ops.nms(anchors, scores, 0.3)
        keep = remove_nms(anchors, scores, planes, 0.5, 0.9)

        print('anchors kept: ', keep.shape[0])

        keep_ds = DisjointSet()
        merge_thresh = 0.4
        for i in range(keep.shape[0]):
            for j in range(i + 1, keep.shape[0]):
                si = keep_ds.find(i)
                sj = keep_ds.find(j)
                if si != sj:
                    iou1 = calc_iou(anchors[keep[i]], anchors[keep[j]])
                    iou2 = calc_iou(anchors[keep[j]], anchors[keep[i]])
                    norm_dot = planes[keep[i]].dot(planes[keep[j]]) / (planes[keep[i]].norm() * planes[keep[j]].norm())
                    if (iou1 > merge_thresh or iou2 > merge_thresh) and norm_dot > torch.tensor(10.0 * np.pi / 180.0).cos():
                    # if (iou1 > merge_thresh or iou2 > merge_thresh):
                        keep_ds.union(i, j)

        segm = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)
        color_map = utils.ColorPalette(keep.shape[0]).getColorMap()
        for idx in range(keep.shape[0]):
            pt1 = np.round(anchors[keep[idx], 0:2].numpy()).astype(np.int)
            pt2 = np.round(anchors[keep[idx], 2:4].numpy()).astype(np.int)
            id = keep_ds.find(idx)
            # print('id = ', id)

            mask = cv2.resize(masks[keep[idx]].numpy(), (pt2[0] - pt1[0], pt2[1] - pt1[1]), interpolation=cv2.INTER_NEAREST) / 255
            # segm_color = [(idx + 1) * 100 % 256, (idx + 1) * 100 / 256 % 256, (idx + 1) * 100 / (256 * 256)]
            segm_color = [(id + 1) * 100 % 256, (id + 1) * 100 / 256 % 256, (id + 1) * 100 / (256 * 256)]
            x1 = max(pt1[0], 0)
            x2 = min(pt2[0], segm.shape[1])
            y1 = max(pt1[1], 0)
            y2 = min(pt2[1], segm.shape[0])
            # print(pt1, pt2)
            segm[y1:y2, x1:x2, :] = np.tile(np.expand_dims(mask[y1 - pt1[1]: mask.shape[0] - (pt2[1] - y2), x1 - pt1[0]: mask.shape[1] - (pt2[0] - x2)], axis=-1), [1, 1, 3]) * segm_color
            # print(tuple(color_map[id]))
            cv2.rectangle(im, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (int(color_map[id,0]),int(color_map[id,1]),int(color_map[id,2])), 2)
            # if idx > 0:
            #     iou = calc_iou(anchors[keep[idx - 1]], anchors[keep[idx]])
            #     # print('keep[idx - 1] = ', keep[idx - 1], ', keep[idx] = ', keep[idx])
            #     print('iou = ', iou, ', score1 = ', scores[keep[idx - 1]], ', score2 = ', scores[keep[idx]])
        cv2.imshow('planes', im)
        cv2.imshow('segm', segm)
        cv2.waitKey()


def main():
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
    args.test_dir = 'test/' + args.keyname

    scene_ids = os.listdir(ROOT_FOLDER)
    scene_ids = sorted(scene_ids)
    print(scene_ids)

    np.random.seed(13)

    config = PlaneConfig(args)

    for index, scene_id in enumerate(scene_ids):
        print(index, scene_id)
        test_planes(scene_id, config)


if __name__=='__main__':
    main()
