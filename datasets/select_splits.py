#!/usr/bin/env python

import sys
import argparse
import subprocess
import os
import shutil
import numpy as np
import random


def select_split(inputdir, split, frame_gap, num_samples):
    # to parse invalid indices we have to know relation between indices and scene ids
    # doing it the same way as in PlaneRCNN
    planenet_scene_ids_val = np.load(inputdir + '/scene_ids_val.npy')
    planenet_scene_ids_val = {scene_id.decode('utf-8'): True for scene_id in planenet_scene_ids_val}
    cnt = 0
    scene_index_to_id = {}
    scene_id_to_index = {}
    split_scenes_list = []
    with open(inputdir + '/ScanNet/Tasks/Benchmark/scannetv1_' + split + '.txt') as f:
        for line in f:
            scene_id = line.strip()
            if split == 'test':
                if scene_id not in planenet_scene_ids_val:
                    continue
            scenePath = inputdir + '/scans/' + scene_id
            if not os.path.exists(scenePath + '/' + scene_id + '.txt') or not os.path.exists(
                    scenePath + '/annotation/planes.npy'):
                continue
            scene_index_to_id[cnt] = scene_id
            scene_id_to_index[scene_id] = cnt
            split_scenes_list.append(scene_id)
            cnt += 1

    invalid_scenes_dict = {}
    with open(inputdir + '/invalid_indices_' + split + '_old.txt', 'r') as f:
        for line in f:
            tokens = line.split(' ')
            if len(tokens) == 3:
                assert (int(tokens[2]) < 10000)
                scene_id = scene_index_to_id[int(tokens[1])]
                if scene_id not in invalid_scenes_dict:
                    invalid_scenes_dict[scene_id] = {int(tokens[2])}
                else:
                    invalid_scenes_dict[scene_id].add(int(tokens[2]))

    scenes_len = {}
    view_list = []
    for scene_id in split_scenes_list:
        invalid_views = {}
        if scene_id in invalid_scenes_dict:
            invalid_views = invalid_scenes_dict[scene_id]
        with open(os.path.join(inputdir, 'scans', scene_id, 'num_frames')) as f:
            num_images = int(f.readline())
        # num_images = len(os.listdir(os.path.join(inputdir, 'scans', scene_id, 'annotation', 'segmentation')))
        cur_view_list = [(scene_id, idx) for idx in range(num_images) if idx not in invalid_views]
        if len(cur_view_list) + len(invalid_views) != num_images:
            # print('%d + %d != %d' % (len(cur_view_list), len(invalid_views),  num_images))
            raise Exception('%d + %d != %d' % (len(cur_view_list), len(invalid_views),  num_images))

        scenes_len[scene_id] = num_images
        view_list.extend(cur_view_list)

    view_list_filt = []
    view_set = set(view_list)
    for (scene_id, frame_idx) in view_list:
        if frame_idx + frame_gap < scenes_len[scene_id]:
            idx2 = frame_idx + frame_gap
        else:
            idx2 = frame_idx - frame_gap
        if (scene_id, idx2) in view_set:
            view_list_filt.append((scene_id, frame_idx))

    view_list_samp = random.sample(view_list_filt, num_samples)
    view_set_samp = set(view_list_samp)
    view_set_unpack = view_set_samp
    for (scene_id, frame_idx) in view_list_samp:
        if frame_idx + frame_gap < scenes_len[scene_id]:
            idx2 = frame_idx + frame_gap
        else:
            idx2 = frame_idx - frame_gap
        view_set_unpack.add((scene_id, idx2))

    new_invalid_indices = []
    for scene_id in split_scenes_list:
        scene_idx = scene_id_to_index[scene_id]
        valid_view_idxs = [view_idx for (scene_id_it, view_idx) in view_list_samp if scene_id_it == scene_id]
        invalid_view_idxs = [view_idx for view_idx in range(scenes_len[scene_id]) if view_idx not in valid_view_idxs]
        new_invalid_indices.extend([(scene_idx, view_idx) for view_idx in invalid_view_idxs])

    all_cnt = 0
    for scene_id in split_scenes_list:
        all_cnt += scenes_len[scene_id]
    print("all_cnt = %d" % all_cnt)

    return view_list_samp, list(view_set_unpack), new_invalid_indices


def save_split(inputdir, split, unpack, inv_ind):
    with open(inputdir + '/ScanNet/Tasks/Benchmark/unpack_' + split + '.txt', 'w') as f:
        for view in unpack:
            f.write(view[0] + ' ' + str(view[1]) + '\n')

    with open(inputdir + '/invalid_indices_' + split + '.txt', 'w') as f:
        for idx in inv_ind:
            f.write('%d %d %d\n' % (0, idx[0], idx[1]))


def main():
    parser = argparse.ArgumentParser(description='Export images from rosbag.')
    parser.add_argument('inputdir',
                        help='input scene_id')
    parser.add_argument('-v', '--verbose', action="store_true", default=False,
                        help='verbose output')

    args = parser.parse_args()

    random.seed(7)

    if args.verbose:
        print("Reading folder: " + args.inputdir)

    [train_split, train_unpack, train_inv_ind] = select_split(args.inputdir, 'train', 50, 100000)
    print('number of training images = %d, to unpack = %d, inv ind = %d' % (len(train_split), len(train_unpack), len(train_inv_ind)))

    [test_split, test_unpack, test_inv_ind] = select_split(args.inputdir, 'test', 50, 10000)
    print('number of testing images = %d, to unpack = %d, inv ind = %d' % (len(test_split), len(test_unpack), len(test_inv_ind)))

    save_split(args.inputdir, 'train', train_unpack, train_inv_ind)
    save_split(args.inputdir, 'test', test_unpack, test_inv_ind)


if __name__ == "__main__":
    main()
