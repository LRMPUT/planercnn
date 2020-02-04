#!/usr/bin/env python

import sys
import argparse
import subprocess
import os
import shutil
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Export images from rosbag.')
    parser.add_argument('inputdir',
                        help='input scene_id')
    parser.add_argument('scriptpath',
                        help='path to unpacking script')
    parser.add_argument('-v', '--verbose', action="store_true", default=False,
                        help='verbose output')

    args = parser.parse_args()

    if args.verbose:
        print("Reading scene_id file: " + args.inputdir)

    scenes_list = sorted(os.listdir(os.path.join(args.inputdir, 'scans')))

    # to parse invalid indices we have to know relation between indices and scene ids
    # doing it the same way as in PlaneRCNN
    cnt = 0
    scene_train_index_to_id = {}
    with open(args.inputdir + '/ScanNet/Tasks/Benchmark/scannetv1_train.txt') as f:
        for line in f:
            scene_id = line.strip()
            scenePath = args.inputdir + '/scans/' + scene_id
            if not os.path.exists(scenePath + '/' + scene_id + '.txt') or not os.path.exists(
                    scenePath + '/annotation/planes.npy'):
                continue
            scene_train_index_to_id[cnt] = scene_id
            cnt += 1

    planenet_scene_ids_val = np.load(args.inputdir + '/scene_ids_val.npy')
    planenet_scene_ids_val = {scene_id.decode('utf-8'): True for scene_id in planenet_scene_ids_val}
    cnt = 0
    scene_test_index_to_id = {}
    with open(args.inputdir + '/ScanNet/Tasks/Benchmark/scannetv1_test.txt') as f:
        for line in f:
            scene_id = line.strip()
            if scene_id not in planenet_scene_ids_val:
                continue
            scenePath = args.inputdir + '/scans/' + scene_id
            if not os.path.exists(scenePath + '/' + scene_id + '.txt') or not os.path.exists(
                    scenePath + '/annotation/planes.npy'):
                continue
            scene_test_index_to_id[cnt] = scene_id
            cnt += 1

    split_scenes_list = []
    with open(args.inputdir + '/ScanNet/Tasks/Benchmark/scannetv1_train.txt') as f:
        for line in f:
            scene_id = line.strip()
            split_scenes_list.append(scene_id)
    with open(args.inputdir + '/ScanNet/Tasks/Benchmark/scannetv1_test.txt') as f:
        for line in f:
            scene_id = line.strip()
            split_scenes_list.append(scene_id)

    scenes_list = [scene_id for scene_id in scenes_list if scene_id in split_scenes_list]

    invalid_scenes_dict = {}
    with open(args.inputdir + '/invalid_indices_train.txt', 'r') as f:
        for line in f:
            tokens = line.split(' ')
            if len(tokens) == 3:
                assert(int(tokens[2]) < 10000)
                scene_id = scene_train_index_to_id[int(tokens[1])]
                if scene_id not in invalid_scenes_dict:
                    invalid_scenes_dict[scene_id] = {int(tokens[2])}
                else:
                    invalid_scenes_dict[scene_id].add(int(tokens[2]))

    with open(args.inputdir + '/invalid_indices_test.txt', 'r') as f:
        for line in f:
            tokens = line.split(' ')
            if len(tokens) == 3:
                assert(int(tokens[2]) < 10000)
                scene_id = scene_test_index_to_id[int(tokens[1])]
                if scene_id not in invalid_scenes_dict:
                    invalid_scenes_dict[scene_id] = {int(tokens[2])}
                else:
                    invalid_scenes_dict[scene_id].add(int(tokens[2]))

    view_list = []
    for scene_id in scenes_list:
        if not os.path.exists(os.path.join(args.inputdir, 'scans', scene_id, 'frames')):
            print('unpacking ' + 'scans/' + scene_id)

            # python extractor
            # print(' '.join(['executing:', 'python',
            #                          args.scriptpath,
            #                          '--filename=' + scene_id + '.sens',
            #                          '--output_path=frames',
            #                          '--export_depth_images',
            #                          '--export_color_images',
            #                          '--export_poses',
            #                          '--export_intrinsics']))
            # print('in scene_id: ' + os.path.join(args.inputdir, 'scans', scene_id))
            # proc = subprocess.Popen(['python',
            #                          args.scriptpath,
            #                          '--filename=' + scene_id + '.sens',
            #                          '--output_path=frames',
            #                          '--export_depth_images',
            #                          '--export_color_images',
            #                          '--export_poses',
            #                          '--export_intrinsics'],
            #                         cwd=os.path.join(args.inputdir, 'scans', scene_id))

            # C++ extractor
            # print(' '.join(['executing:',
            #                 args.scriptpath,
            #                 scene_id + '.sens',
            #                 'frames']))
            # print('in scene_id: ' + os.path.join(args.inputdir, 'scans', scene_id))
            # proc = subprocess.Popen([args.scriptpath,
            #                          scene_id + '.sens',
            #                          'frames'],
            #                         cwd=os.path.join(args.inputdir, 'scans', scene_id))
            # stdout, stderr = proc.communicate()
            # if proc.returncode != 0:
            #     print('Extracting failed')
            #     break

            # if scene_id in invalid_scenes_dict:
            #     print('removing %d invalid views' % len(invalid_scenes_dict[scene_id]))
            #     for frame_idx in invalid_scenes_dict[scene_id]:
            #         color_image_path = os.path.join(args.inputdir, 'scans', scene_id, 'frames',
            #                                         'frame-%06d.color.jpg' % frame_idx)
            #         depth_image_path = os.path.join(args.inputdir, 'scans', scene_id, 'frames',
            #                                         'frame-%06d.depth.pgm' % frame_idx)
            #         pose_path = os.path.join(args.inputdir, 'scans', scene_id, 'frames',
            #                                  'frame-%06d.pose.txt' % frame_idx)
            #         # print('deleting ' + color_image_path)
            #         os.remove(color_image_path)
            #         # print('deleting ' + depth_image_path)
            #         os.remove(depth_image_path)
            #         # print('deleting ' + pose_path)
            #         os.remove(pose_path)

            invalid_views = {}
            if scene_id in invalid_scenes_dict:
                invalid_views = invalid_scenes_dict[scene_id]
            num_images = len(os.listdir(os.path.join(args.inputdir, 'scans', scene_id, 'annotation', 'segmentation')))
            cur_view_list = [(scene_id, idx) for idx in range(num_images) if idx not in invalid_views]
            if len(cur_view_list) + len(invalid_views) != num_images:
                print('%d + %d != %d' % (len(cur_view_list), len(invalid_views),  num_images))
            view_list.extend(cur_view_list)
        else:
            print('skipping ' + 'scans/' + scene_id)
            # print('removing ' + os.path.join(args.inputdir, 'scans', scene_id, 'frames'))
            # shutil.rmtree(os.path.join(args.inputdir, 'scans', scene_id, 'frames'))

    print('number of images = %d' % len(view_list))


if __name__ == "__main__":
    main()
