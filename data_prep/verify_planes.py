"""
Copyright Jan Wietrzykowski 2020
"""

import numpy as np
import cv2
import sys
import os

ROOT_FOLDER = "/mnt/data/datasets/JW/scenenet_rgbd/scenes/"


def draw_planes(scene_id):
    draw_nonplanar = False

    images_list = sorted(os.listdir(os.path.join(ROOT_FOLDER, scene_id, 'annotation', 'segmentation')))
    planes_file = os.path.join(ROOT_FOLDER, scene_id, 'annotation', 'planes.npy')

    planes = np.load(planes_file)
    for seg_file in images_list:
        seg = cv2.imread(os.path.join(ROOT_FOLDER, scene_id, 'annotation', 'segmentation', seg_file)).astype(np.int32)
        plane_idxs = (seg[:, :, 2] * 256 * 256 + seg[:, :, 1] * 256 + seg[:, :, 0]) // 100 - 1
        for cur_plane_idx, cur_plane in enumerate(planes):
            if np.linalg.norm(cur_plane) < 1e-4:
                if draw_nonplanar:
                    seg[plane_idxs == cur_plane_idx] = 255 - seg[plane_idxs == cur_plane_idx]
                else:
                    seg[plane_idxs == cur_plane_idx] = 0

        cv2.imshow('planes', seg.astype(np.uint8))
        cv2.waitKey()


def main():
    scene_ids = os.listdir(ROOT_FOLDER)
    scene_ids = sorted(scene_ids)
    print(scene_ids)

    np.random.seed(13)

    for index, scene_id in enumerate(scene_ids):
        print(index, scene_id)
        draw_planes(scene_id)


if __name__=='__main__':
    main()
