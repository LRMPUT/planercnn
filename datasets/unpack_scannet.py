#!/usr/bin/env python

import sys
import argparse
import subprocess
import os
import shutil
import numpy as np


def read_unpack_file(inputdir, split, scene_set, scene_id_to_valid_idxs):
    with open(os.path.join(inputdir, 'ScanNet/Tasks/Benchmark/unpack_' + split + '.txt')) as f:
        for line in f:
            tokens = line.split(' ')
            scene_id = tokens[0]
            view_idx = int(tokens[1])
            scene_set.add(scene_id)
            if scene_id in scene_id_to_valid_idxs:
                scene_id_to_valid_idxs[scene_id].add(view_idx)
            else:
                scene_id_to_valid_idxs[scene_id] = {view_idx}


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

    # scene_set = set()
    # scene_id_to_valid_idxs = {}
    # read_unpack_file(args.inputdir, 'train', scene_set, scene_id_to_valid_idxs)
    # read_unpack_file(args.inputdir, 'test', scene_set, scene_id_to_valid_idxs)

    # for scene_id, valid_idxs in scene_id_to_valid_idxs.items():
    for scene_id in scenes_list:
        if not os.path.exists(os.path.join(args.inputdir, 'scans', scene_id, 'frames')):
            print('unpacking ' + 'scans/' + scene_id)
            # pass

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
            print(' '.join(['executing:', 'python',
                                     args.scriptpath,
                                     '--filename=scans/' + scene_id + '.sens',
                                     '--output_path=.',
                                     '--export_num_frames']))
            print('in scene_id: ' + os.path.join(args.inputdir, 'scans', scene_id))
            proc = subprocess.Popen(['python',
                                     args.scriptpath,
                                     '--filename=scans/' + scene_id + '.sens',
                                     '--output_path=.',
                                     '--export_num_frames'],
                                    cwd=os.path.join(args.inputdir, 'scans', scene_id))

            # # C++ extractor
            # print(' '.join(['executing:',
            #                 args.scriptpath,
            #                 scene_id + '.sens',
            #                 'frames']))
            # print('in scene_id: ' + os.path.join(args.inputdir, 'scans', scene_id))
            # proc = subprocess.Popen([args.scriptpath,
            #                          scene_id + '.sens',
            #                          'frames'],
            #                         cwd=os.path.join(args.inputdir, 'scans', scene_id))
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print('Extracting failed')
                break
            #
            # print('leaving %d valid views' % len(valid_idxs))
            # num_images = len(os.listdir(os.path.join(args.inputdir, 'scans', scene_id, 'annotation', 'segmentation')))
            # for frame_idx in range(num_images):
            #     if frame_idx not in valid_idxs:
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
        else:
            print('skipping ' + 'scans/' + scene_id)
            # print('removing ' + os.path.join(args.inputdir, 'scans', scene_id, 'frames'))
            # shutil.rmtree(os.path.join(args.inputdir, 'scans', scene_id, 'frames'))


if __name__ == "__main__":
    main()
