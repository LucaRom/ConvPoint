#!/usr/bin/python()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import warnings
import laspy
from examples.airborne_lidar.airborne_lidar_utils import write_features


def read_las_format(raw_path, normalize=True):
    """Extract data from a .las file.
    If normalize is set to True, will normalize XYZ and intensity between 0 and 1."""

    in_file = laspy.file.File(raw_path, mode='r')
    n_points = len(in_file)
    x = np.reshape(in_file.x, (n_points, 1))
    y = np.reshape(in_file.y, (n_points, 1))
    z = np.reshape(in_file.z, (n_points, 1))
    intensity = np.reshape(in_file.intensity, (n_points, 1))
    nb_return = np.reshape(in_file.num_returns, (n_points, 1))
    labels = np.reshape(in_file.classification, (n_points, 1))
    labels = format_classes(labels)

    if normalize:
        # Converting data to relative xyz reference system.
        norm_x = x - np.min(x)
        norm_y = y - np.min(y)
        norm_z = z - np.min(z)
        # Intensity is normalized based on min max values.
        norm_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        xyzni = np.hstack((norm_x, norm_y, norm_z, nb_return, norm_intensity)).astype(np.float16)
    else:
        xyzni = np.hstack((x, y, z, nb_return, intensity)).astype(np.float16)

    return xyzni, labels, n_points


def format_classes(labels):
    """Format labels array to match the classes of interest. Specific to airborne_lidar dataset.
    # Dict containing the mapping of input (from the .las file) and the output classes (for the training step).
    # 6: Building
    # 9: water
    # 2: ground.
    """
    coi = {'6': 1, '9': 2, '2': 3}
    labels2 = np.full(shape=labels.shape, fill_value=0, dtype=int)
    for key, value in coi.items():
        labels2[labels == int(key)] = value

    return labels2


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', default='/export/sata01/wspace/lidar/POINTCLOUD/data/', help='Path to data folder')
    parser.add_argument("--dest", '-d', default='/export/sata01/wspace/lidar/convpoint_tests/prepared', help='Path to destination folder')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    base_dir = args.folder

    dataset_dict = {'trn': [], 'val': [], 'tst': []}

    for dataset in dataset_dict.keys():
        for f in os.listdir(os.path.join(base_dir, dataset)):
            if f.endswith('.las'):
                dataset_dict[dataset].append(f)

        if len(dataset_dict[dataset]) == 0:
            warnings.warn(f"{os.path.join(base_dir, dataset)} is empty")

    print(f"Las files per dataset:\n Trn: {len(dataset_dict['trn'])} \n Val: {len(dataset_dict['val'])} \n Tst: {len(dataset_dict['tst'])}")

    for dst in dataset_dict:
        for elem in dataset_dict[dst]:
            # make store directories
            path_prepare_label = os.path.join(args.dest, dst)
            if not os.path.exists(path_prepare_label):
                os.makedirs(path_prepare_label)

            xyzni, label, nb_pts = read_las_format(os.path.join(base_dir, dst, elem))

            write_features(f"{path_prepare_label}/{elem.split('.')[0]}_prepared.hdfs", xyzni=xyzni, labels=label)
            print(f"File {dst}/{elem} prepared. {nb_pts:,} points written.")


if __name__ == "__main__":
    main()
