"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky

this functions outputs varied ply file to visualize the different steps
"""
import argparse
import sys

import h5py
import os
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser(description='Custom visualization script')
    parser.add_argument('--ROOT_PATH', default='/wspace/disk01/lidar/POINTCLOUD', help='folder containing labeled data')
    parser.add_argument('--res_file', default='', help='folder containing the results')
    parser.add_argument('--model_folder', help='Folder containing the model ')
    parser.add_argument('--output_type', default='gre', help='which cloud to output: g = ground truth, r = prediction result , e = error')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Read model.
    # Read tst dataset.
    # Make predictions.
    # Convert to ply files:
    #   ground truth file
    #   prediction results
    #   error file.



if __name__ == '__main__':
    main()

root = args.ROOT_PATH + '/'
rgb_out = 'i' in args.output_type
gt_out = 'g' in args.output_type
fea_out = 'f' in args.output_type
par_out = 'p' in args.output_type
res_out = 'r' in args.output_type
err_out = 'e' in args.output_type
spg_out = 's' in args.output_type
folder = os.path.split(args.file_path)[0] + '/'
file_name = os.path.split(args.file_path)[1]

if args.dataset == 'airborne_lidar':
    n_labels = 4

# ---load the values------------------------------------------------------------
fea_file = root + "features/" + folder + file_name + '.h5'
if not os.path.isfile(fea_file) or args.supervized_partition:
    fea_file = root + "features_supervision/" + folder + file_name + '.h5'
spg_file = root + "superpoint_graphs/" + folder + file_name + '.h5'
ply_folder = root + "clouds/" + folder
ply_file = ply_folder + file_name
res_file = args.res_file + '.h5'

if not os.path.isdir(root + "clouds/"):
    os.mkdir(root + "clouds/")
if not os.path.isdir(ply_folder):
    os.mkdir(ply_folder)
if not os.path.isfile(fea_file):
    raise ValueError("%s does not exist and is needed" % fea_file)

geof, xyz, rgb, graph_nn, labels, intensity, nb_return = read_features(fea_file)

if (par_out or res_out) and (not os.path.isfile(spg_file)):
    raise ValueError("%s does not exist and is needed to output the partition or result ply" % spg_file)
else:
    graph_spg, components, in_component = read_spg(spg_file)
if res_out or err_out:
    if not os.path.isfile(res_file):
        raise ValueError("%s does not exist and is needed to output the result ply" % res_file)
    try:
        pred_red = np.array(h5py.File(res_file, 'r').get(folder + file_name))
        if len(pred_red) != len(components):
            raise ValueError("It looks like the spg is not adapted to the result file")
        pred_full = reduced_labels2full(pred_red, components, len(xyz))
    except OSError:
        raise ValueError("%s does not exist in %s" % (folder + file_name, res_file))
# ---write the output clouds----------------------------------------------------

if gt_out:
    print("writing the GT file...")
    prediction2ply(ply_file + "_GT.ply", xyz, labels, n_labels, args.dataset)

if res_out and not bool(args.upsample):
    print("writing the prediction file...")
    prediction2ply(ply_file + "_pred.ply", xyz, pred_full + 1, n_labels, args.dataset)

if err_out:
    print("writing the error file...")
    error2ply(ply_file + "_err.ply", xyz, rgb, labels, pred_full + 1)


if res_out and bool(args.upsample):
    if args.dataset == 's3dis':
        data_file = root + 'data/' + folder + file_name + '/' + file_name + ".txt"
        xyz_up, rgb_up = read_s3dis_format(data_file, False)
    elif args.dataset == 'sema3d':  # really not recommended unless you are very confident in your hardware
        data_file = data_folder + file_name + ".txt"
        xyz_up, rgb_up = read_semantic3d_format(data_file, 0, '', 0, args.ver_batch)
    elif args.dataset == 'custom_dataset':
        data_file = data_folder + file_name + ".ply"
        xyz_up, rgb_up = read_ply(data_file)
    del rgb_up
    pred_up = interpolate_labels(xyz_up, xyz, pred_full, args.ver_batch)
    print("writing the upsampled prediction file...")
    prediction2ply(ply_file + "_pred_up.ply", xyz_up, pred_up + 1, n_labels, args.dataset)
