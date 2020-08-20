#!/usr/bin/python()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import warnings
import laspy
from pathlib import Path
from airborne_lidar_utils import write_features
import csv


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', default='/export/sata01/wspace/lidar/POINTCLOUD/data/', help='Path to data folder')
    parser.add_argument("--dest", '-d', default='/export/sata01/wspace/lidar/convpoint_tests/prepared', help='Path to destination folder')
    parser.add_argument("--csv", default='/wspace/disk01/lidar/ERD_2018_aoi1.csv', help='Path to the csv file describing relationship between '
                                                                                        'lasfiles and the dataset in which they will be use.')
    args = parser.parse_args()
    return args


def read_csv(csv_file_name):
    """Open csv file and parse it, returning a list of dict.
        - las_name
        - dataset
    """

    dataset_dict = {'trn': [], 'val': [], 'tst': []}
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            dataset_dict[row[1]].append(row[0])

    return dataset_dict


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

    if normalize:
        # Converting data to relative xyz reference system.
        mask = (labels >= 0)
        x = x[mask].reshape((-1, 1))
        y = y[mask].reshape((-1, 1))
        z = z[mask].reshape((-1, 1))
        intensity = intensity[mask].reshape((-1, 1))
        nb_return = nb_return[mask].reshape((-1, 1))
        labels = labels[mask].reshape((-1, 1))
        norm_x = x - np.min(x)
        norm_y = y - np.min(y)
        norm_z = z - np.min(z)

        # Intensity is normalized based on min max values.
        norm_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        xyzni = np.hstack((norm_x, norm_y, norm_z, nb_return, norm_intensity)).astype(np.float16)
    else:
        xyzni = np.hstack((x, y, z, nb_return, intensity)).astype(np.float16)

    return xyzni, labels, n_points


def main():
    args = parse_args()
    dataset_dict = read_csv(args.csv)
    base_dir = Path(args.folder)

    # List .las files in each dataset.
    print(f"Las files per dataset:\n Trn: {len(dataset_dict['trn'])} \n Val: {len(dataset_dict['val'])} \n Tst: {len(dataset_dict['tst'])}")

    # Write new hdfs of XYZ + number of return + intensity, with labels.
    for dst, values in dataset_dict.items():
        for elem in values:
            # make store directories
            path_prepare_label = Path(args.dest, dst)
            path_prepare_label.mkdir(exist_ok=True)

            xyzni, label, nb_pts = read_las_format(base_dir / f"{elem}.las")

            write_features(Path(f"{path_prepare_label / elem.split('.')[0]}_prepared.hdfs"), xyzni=xyzni, labels=label)
            print(f"File {dst}/{elem} prepared. {nb_pts:,} points written.")


if __name__ == "__main__":
    main()
