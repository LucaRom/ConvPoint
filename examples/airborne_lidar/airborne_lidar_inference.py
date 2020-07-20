# add the parent folder to the python path to access convpoint library
import sys
import warnings
sys.path.append('/space/partner/nrcan/geobase/work/transfer/work/deep_learning/lidar/CMM_2018/convpoint_tests/ConvPoint')

import argparse
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.utils.data
from pathlib import Path
from airborne_lidar_seg import get_model, nearest_correspondance, count_parameters, class_mode
import laspy
from airborne_lidar_utils import write_features
from airborne_lidar_datasets import PartDatasetTest

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", default='/wspace/disk01/lidar/convpoint_tests/results/SegBig_8168_drop0_2020-03-12-07-47-46', type=str)
    parser.add_argument("--rootdir", default='/wspace/disk01/lidar/POINTCLOUD/data/', type=str,
                        help="Folder conntaining tst subfolder with las files.")
    parser.add_argument("--test_step", default=5, type=float)

    args = parser.parse_args()
    config_dict = read_config_from_yaml(Path(args.modeldir))
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key not in ['rootdir', 'test_step']:
            arg_dict[key] = value

    return args


def read_config_from_yaml(folder):
    with open(folder / 'config.yaml', 'r') as in_file:
        yaml_dict = yaml.load(in_file, Loader=yaml.FullLoader)
    return yaml_dict


def read_las_format(in_file):
    """Extract data from a .las file.
    Will normalize XYZ and intensity between 0 and 1.
    """

    n_points = len(in_file)
    x = np.reshape(in_file.x, (n_points, 1))
    y = np.reshape(in_file.y, (n_points, 1))
    z = np.reshape(in_file.z, (n_points, 1))
    intensity = np.reshape(in_file.intensity, (n_points, 1))
    nb_return = np.reshape(in_file.num_returns, (n_points, 1))

    # Converting data to relative xyz reference system.
    min_x = np.min(x)
    min_y = np.min(y)
    min_z = np.min(z)
    norm_x = x - min_x
    norm_y = y - min_y
    norm_z = z - min_z
    # Intensity is normalized based on min max values.
    norm_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    xyzni = np.hstack((norm_x, norm_y, norm_z, nb_return, norm_intensity)).astype(np.float16)

    return xyzni


def write_las_to_h5(filename):
    with laspy.file.File(filename) as in_file:
        xyzni = read_las_format(in_file)

        filename = f"{filename.parent / filename.name.split('.')[0]}_prepared.hdfs"
        write_features(filename, xyzni=xyzni)
        return filename


def write_to_las(filename, xyz, pred, header, info_class):
    """Write xyz and ASPRS predictions to las file format. """
    # TODO: Write CRS info with file.
    with laspy.file.File(filename, mode='w', header=header) as out_file:
        out_file.x = xyz[:, 0]
        out_file.y = xyz[:, 1]
        out_file.z = xyz[:, 2]
        pred = pred_to_asprs(pred, info_class)
        out_file.classification = pred


def pred_to_asprs(pred, info_class):
    """Converts predicted values (0->n) to the corresponding ASPRS class."""
    labels2 = np.full(shape=pred.shape, fill_value=0, dtype=int)
    for key, value in info_class.items():
        labels2[pred == value['mode']] = int(key)
    return labels2


def test(args, filename, model_folder, info_class):
    nb_class = info_class['nb_class']
    # create the network
    print("Creating network...")
    if torch.cuda.is_available():
        state = torch.load(model_folder)
    else:
        state = torch.load(model_folder, map_location=torch.device('cpu'))
    arg_dict = args.__dict__
    config_dict = state['args'].__dict__
    for key, value in config_dict.items():
        if key not in ['rootdir', 'num_workers', 'batchsize']:
            arg_dict[key] = value
    net, features = get_model(nb_class, args)
    net.load_state_dict(state['state_dict'])
    if torch.cuda.is_available():
        net.cuda()
    else:
        net.cpu()
    net.eval()
    print(f"Number of parameters in the model: {count_parameters(net):,}")
    print(f"Processing {filename}")
    las_filename = Path(args.rootdir) / f"{filename}.las"
    h5_filename = write_las_to_h5(Path(args.rootdir) / f"{filename}.las")
    out_folder = model_folder.parent / 'tst'
    out_folder.mkdir(exist_ok=True)

    ds_tst = PartDatasetTest(h5_filename, block_size=args.blocksize, npoints=args.npoints, test_step=args.test_step, features=features)
    tst_loader = torch.utils.data.DataLoader(ds_tst, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    xyz = ds_tst.xyzni[:, :3]
    scores = np.zeros((xyz.shape[0], nb_class))

    total_time = 0
    iter_nb = 0
    with torch.no_grad():
        t = tqdm(tst_loader, ncols=150)
        for pts, features, indices in t:
            t1 = time.time()
            if torch.cuda.is_available():
                features = features.cuda()
                pts = pts.cuda()
            outputs = net(features, pts)
            t2 = time.time()

            outputs_np = outputs.cpu().numpy().reshape((-1, nb_class))
            scores[indices.cpu().numpy().ravel()] += outputs_np

            iter_nb += 1
            total_time += (t2 - t1)
            t.set_postfix(time=f"{total_time / (iter_nb * args.batchsize):05e}")

    mask = np.logical_not(scores.sum(1) == 0)
    scores = scores[mask]
    pts_src = xyz[mask]

    # create the scores for all points
    scores = nearest_correspondance(pts_src, xyz, scores, K=1)

    # compute softmax
    scores = scores - scores.max(axis=1)[:, None]
    scores = np.exp(scores) / np.exp(scores).sum(1)[:, None]
    scores = np.nan_to_num(scores)
    scores = scores.argmax(1)

    # Save predictions
    with laspy.file.File(las_filename) as in_file:
        header = in_file.header
        xyz = np.vstack((in_file.x, in_file.y, in_file.z)).transpose()
        write_to_las(out_folder / f"{filename}_predictions.las", xyz=xyz, pred=scores, header=header, info_class=info_class['class_info'])


def main():
    args = parse_args()

    # create the file lists (trn / val / tst)
    print("Create file list...")
    base_dir = Path(args.rootdir)
    dataset_dict = []

    for file in base_dir.glob('*.las'):
        dataset_dict.append(file.stem)

    if len(dataset_dict) == 0:
        warnings.warn(f"{base_dir} is empty")

    print(f"Las files in tst dataset: {len(dataset_dict)}")

    info_class = class_mode(args.mode)
    model_folder = Path(args.modeldir)
    for filename in dataset_dict:
        test(args, filename, model_folder, info_class)


if __name__ == '__main__':
    main()
