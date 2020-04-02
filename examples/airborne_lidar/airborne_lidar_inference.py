# add the parent folder to the python path to access convpoint library
import sys
import warnings
sys.path.append('/wspace/disk01/lidar/convpoint_test/ConvPoint/convpoint')

import argparse
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.utils.data
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors
import h5py
from pathlib import Path
from examples.airborne_lidar.airborne_lidar_viz import prediction2ply


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", default='/wspace/disk01/lidar/convpoint_tests/results', type=str)
    parser.add_argument("--rootdir", default='/wspace/disk01/lidar/convpoint_tests/prepared', type=str)
    parser.add_argument("--batchsize", "-b", default=10, type=int)
    parser.add_argument("--npoints", default=8168, type=int, help="Number of points to be sampled in the block.")
    parser.add_argument("--blocksize", default=50, type=int, help="Size of the infinite vertical column, to be processed.")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--features", default="xyzni", type=str, help="Features to process. xyzni means xyz + number of returns + intensity. "
                                                                      "Currently, only xyz and xyzni are supported for this dataset.")
    parser.add_argument("--test_step", default=15, type=float)
    parser.add_argument("--model", default="SegBig", type=str, help="SegBig is the only available model at this time, for this dataset.")
    parser.add_argument("--mode", default=2, type=int, help="Class mode. Currently 2 choices available. "
                                                            "1: building, water, ground."
                                                            "2: building, water, ground, low vegetation and medium + high vegetation")
    args = parser.parse_args()
    return args


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE + str + bcolors.ENDC


def wgreen(str):
    return bcolors.OKGREEN + str + bcolors.ENDC


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src.astype(np.float32), pts_dest.astype(np.float32), K, omp=True)
    print(indices.shape)
    if K == 1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest


def class_mode(mode):
    """
    # Dict containing the mapping of input (from the .las file) and the output classes (for the training step).
    """
    asprs_class_def = {'2': {'name': 'Ground', 'color': [233, 233, 229], 'mode': 0},  # light grey
                       '3': {'name': 'Low vegetation', 'color': [77, 174, 84], 'mode': 0},  # bright green
                       '4': {'name': 'Medium vegetation', 'color': [81, 163, 148], 'mode': 0},  # bluegreen
                       '5': {'name': 'High Vegetation', 'color': [108, 135, 75], 'mode': 0},  # dark green
                       '6': {'name': 'Building', 'color': [223, 52, 52], 'mode': 0},  # red
                       '9': {'name': 'Water', 'color': [95, 156, 196], 'mode': 0}  # blue
                       }
    coi = {}
    unique_class = []
    if mode == 1:
        asprs_class_to_use = {'6': 1, '9': 2, '2': 3}

    elif mode == 2:
        asprs_class_to_use = {'6': 1, '9': 2, '2': 3, '3': 4, '4': 5, '5': 5}

    else:
        raise ValueError(f"Class mode provided ({mode}) is not defined.")

    for key, value in asprs_class_def.items():
        if key in asprs_class_to_use.keys():
            coi[key] = value
            coi[key]['mode'] = asprs_class_to_use[key]
            if asprs_class_to_use[key] not in unique_class:
                unique_class.append(asprs_class_to_use[key])

    nb_class = len(unique_class) + 1
    return {'class_info': coi, 'nb_class': nb_class}


# Part dataset only for testing
class PartDatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzni[:, 0] < pt[0] + bs / 2, self.xyzni[:, 0] > pt[0] - bs / 2)
        mask_y = np.logical_and(self.xyzni[:, 1] < pt[1] + bs / 2, self.xyzni[:, 1] > pt[1] - bs / 2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__(self, filename, folder, block_size=8, npoints=8192, test_step=5, features=False):

        self.filename = filename
        self.folder = Path(folder)
        self.bs = block_size
        self.npoints = npoints
        self.features = features
        self.step = test_step

        # load the points
        data_file = h5py.File(self.folder / f"{filename}.hdfs", 'r')
        self.xyzni = data_file["xyzni"][:]

        discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float) * self.step

    def __getitem__(self, index):

        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzni[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # indices in the original point cloud
        indices = np.where(mask)[0][choice]

        # separate between features and points
        if self.features is False:
            fts = np.ones((pts.shape[0], 1))
        else:
            fts = pts[:, 3:]
            fts = fts.astype(np.float32)

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        indices = torch.from_numpy(indices).long()

        return pts, fts, indices

    def __len__(self):
        return len(self.pts)


def get_model(nb_classes, args):
    # Select the model
    if args.model == "SegBig":
        from networks.network_seg import SegBig as Net
    else:
        raise NotImplemented(f"The model {args.model} does not exist. Only SegBig is available at this time.")

    # Number of features as input
    if args.features == "xyzni":
        input_channels = 2
        features = True
    elif args.features == "xyz":
        input_channels = 1
        features = False
    else:
        raise NotImplemented(f"Features {args.features} are not supported. Only xyzni or xyz, at this time.")

    return Net(input_channels, output_channels=nb_classes, args=args), features


def test(args, flist_test, model_folder, info_class):
    nb_class = info_class['nb_class']
    # create the network
    print("Creating network...")
    net, features = get_model(nb_class, args)
    net.load_state_dict(torch.load(model_folder / "state_dict.pth"))
    net.cuda()
    net.eval()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    for filename in flist_test:
        print(filename)
        ds_tst = PartDatasetTest(filename, args.rootdir, block_size=args.blocksize,
                                 npoints=args.npoints, test_step=args.test_step, features=features)
        tst_loader = torch.utils.data.DataLoader(ds_tst, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

        xyz = ds_tst.xyzni[:, :3]
        scores = np.zeros((xyz.shape[0], nb_class))

        total_time = 0
        iter_nb = 0
        with torch.no_grad():
            t = tqdm(tst_loader, ncols=150)
            for pts, features, indices in t:
                t1 = time.time()
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
        out_folder = model_folder / 'tst'
        out_folder.mkdir(exist_ok=True)
        prediction2ply(model_folder / f"{filename}_predictions.ply", xyz=xyz, prediction=scores, info_class=info_class['class_info'])


def main():
    args = parse_args()

    # create the file lists (trn / val / tst)
    print("Create file list...")
    base_dir = Path(args.rootdir)
    dataset_dict = {'tst': []}

    for dataset in dataset_dict.keys():
        for file in (base_dir / dataset).glob('*.hdfs'):
            dataset_dict[dataset].append(f"{dataset}/{file.stem}")

        if len(dataset_dict[dataset]) == 0:
            warnings.warn(f"{base_dir / dataset} is empty")

    print(f"Las files in tst dataset: {len(dataset_dict['tst'])}")

    info_class = class_mode(args.mode)
    model_folder = args.modeldir
    test(args, dataset_dict['tst'], model_folder, info_class)


if __name__ == '__main__':
    main()
