#!/usr/bin/env python

import sys
import argparse
import subprocess
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description='Export images from rosbag.')
    parser.add_argument('inputdir',
                        help='input directory')
    parser.add_argument('scriptpath',
                        help='path to unpacking script')
    parser.add_argument('-v', '--verbose', action="store_true", default=False,
                        help='verbose output')

    args = parser.parse_args()

    if args.verbose:
        print("Reading directory file: " + args.inputdir)

    scenes_list = sorted(os.listdir(args.inputdir))

    for directory in scenes_list:
        if not os.path.exists(os.path.join(args.inputdir, directory, 'frames')):
            print('unpacking ' + directory)
            print(' '.join(['executing:', 'python',
                                     args.scriptpath,
                                     '--filename=' + directory + '.sens',
                                     '--output_path=frames',
                                     '--export_depth_images',
                                     '--export_color_images',
                                     '--export_poses',
                                     '--export_intrinsics']))
            print('in directory: ' + os.path.join(args.inputdir, directory))
            proc = subprocess.Popen(['python',
                                     args.scriptpath,
                                     '--filename=' + directory + '.sens',
                                     '--output_path=frames',
                                     '--export_depth_images',
                                     '--export_color_images',
                                     '--export_poses',
                                     '--export_intrinsics'],
                                    cwd=os.path.join(args.inputdir, directory))
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print('Extracting failed')
                break
        else:
            print('skipping ' + directory)


if __name__ == "__main__":
    main()
