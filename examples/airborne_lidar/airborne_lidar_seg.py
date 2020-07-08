# Airborne lidar example with ConvPoint

# add the parent folder to the python path to access convpoint library
import sys
import warnings
sys.path.append('/space/partner/nrcan/geobase/work/transfer/work/deep_learning/lidar/CMM_2018/convpoint_tests/ConvPoint')

import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
import time
import torch
import torch.utils.data
import torch.nn.functional as F
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors
import utils.metrics as metrics
from examples.airborne_lidar.airborne_lidar_utils import InformationLogger, print_metric, write_config
import h5py
from pathlib import Path
from examples.airborne_lidar.airborne_lidar_viz import prediction2ply, error2ply
from mlflow import log_params, set_tracking_uri, set_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=True)
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--savedir",
                        default='/space/partner/nrcan/geobase/work/transfer/work/deep_learning/lidar/CMM_2018/convpoint_tests/results', type=str)
    parser.add_argument("--rootdir",
                        default='/space/partner/nrcan/geobase/work/transfer/work/deep_learning/lidar/CMM_2018/convpoint_tests/prepared', type=str)
    parser.add_argument("--mlruns_dir", default='/space/partner/nrcan/geobase/work/transfer/work/deep_learning/mlflow/mlruns')
    parser.add_argument("--batchsize", "-b", default=10, type=int)
    parser.add_argument("--npoints", default=8168, type=int, help="Number of points to be sampled in the block.")
    parser.add_argument("--blocksize", default=50, type=int, help="Size of the infinite vertical column, to be processed.")
    parser.add_argument("--iter", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--features", default="xyzni", type=str, help="Features to process. xyzni means xyz + number of returns + intensity. "
                                                                      "Currently, only xyz and xyzni are supported for this dataset.")
    parser.add_argument("--test_step", default=50, type=float)
    parser.add_argument("--test_labels", default=True, type=bool, help="Labels available for test dataset")
    parser.add_argument("--val_iter", default=1, type=int, help="Number of iterations at validation.")
    parser.add_argument("--nepochs", default=1, type=int)
    parser.add_argument("--model", default="SegBig", type=str, help="SegBig is the only available model at this time, for this dataset.")
    parser.add_argument("--drop", default=0, type=float)

    parser.add_argument("--lr", default=1e-3, help="Learning rate")
    parser.add_argument("--mode", default=2, type=int, help="Class mode. See class_mode function for more available options.")
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


def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1], ])
    return np.dot(batch_data, rotation_matrix)


def class_mode(mode):
    """
    Dict containing the mapping of input (from the .las file) and the output classes (for the training step).
    ASPRS Codes used for this classification
        ASPRS 1 = Unclassified
        ASPRS 2 = Ground
        ASPRS 3 = Low Vegetation
        ASPRS 4 = Medium Vegetation
        ASPRS 5 = High Vegetation
        ASPRS 6 = Buildings
        ASPRS 7 = Low Noise
        ASPRS 8 = Model Key-Point
        ASPRS 9 = Water
        ASPRS 17 = Bridge
        ASPRS 18 = High Noise
Entit
    """
    asprs_class_def = {'2': {'name': 'Ground', 'color': [233, 233, 229], 'mode': 0},  # light grey
                       '3': {'name': 'Low vegetation', 'color': [77, 174, 84], 'mode': 0},  # bright green
                       '4': {'name': 'Medium vegetation', 'color': [81, 163, 148], 'mode': 0},  # bluegreen
                       '5': {'name': 'High Vegetation', 'color': [108, 135, 75], 'mode': 0},  # dark green
                       '6': {'name': 'Building', 'color': [223, 52, 52], 'mode': 0},  # red
                       '9': {'name': 'Water', 'color': [95, 156, 196], 'mode': 0}  # blue
                       }
    dales_class_def = {'1': {'name': 'Ground', 'color': [233, 233, 229], 'mode': 0},  # light grey
                       '2': {'name': 'vegetation', 'color': [77, 174, 84], 'mode': 0},  # bright green
                       '3': {'name': 'cars', 'color': [255, 163, 148], 'mode': 0},  # bluegreen
                       '4': {'name': 'trucks', 'color': [255, 135, 75], 'mode': 0},  # dark green
                       '5': {'name': 'power lines', 'color': [255, 135, 75], 'mode': 0},  # dark green
                       '6': {'name': 'fences', 'color': [255, 135, 75], 'mode': 0},  # dark green
                       '7': {'name': 'poles', 'color': [255, 135, 75], 'mode': 0},  # dark green
                       '8': {'name': 'Building', 'color': [223, 52, 52], 'mode': 0}  # red
                       }
    coi = {}
    unique_class = []
    if mode == 1:
        asprs_class_to_use = {'6': 1, '9': 2, '2': 3}
    elif mode == 2:
        asprs_class_to_use = {'6': 1, '9': 2, '2': 3, '3': 4, '4': 5, '5': 5}  # considering medium and high vegetation as the same class
    elif mode == 3:
        asprs_class_to_use = {'6': 1, '9': 2, '2': 3, '3': 4, '4': 5, '5': 6}  # considering medium and high vegetation as different classes
    elif mode == 4:
        asprs_class_def = dales_class_def
        # ground(1), vegetation(2), cars(3), trucks(4), power lines(5), fences(6), poles(7) and buildings(8)
        asprs_class_to_use = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}
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


# Part dataset only for training / validation
class PartDatasetTrainVal():

    def __init__(self, filelist, folder, training, block_size, npoints, iteration_number, features, class_info):

        self.filelist = filelist
        self.folder = Path(folder)
        self.training = training
        self.bs = block_size
        self.npoints = npoints
        self.iterations = iteration_number
        self.features = features
        self.class_info = class_info
        self.tolerance_percent = 20
        self.xyzni = None
        self.labels = None

    def __getitem__(self, index):

        # Load data
        index = random.randint(0, len(self.filelist) - 1)
        dataset = self.filelist[index]
        data_file = h5py.File(self.folder / f"{dataset}.hdfs", 'r')

        # Get the features
        self.xyzni = data_file["xyzni"][:]
        self.labels = self.format_classes(data_file["labels"][:])
        # self.labels = self.format_classes(self.labels)

        # pick a random point
        pt_id = random.randint(0, self.xyzni.shape[0] - 1)
        pt = self.xyzni[pt_id, :3]

        # Create the mask and select all points in the column.
        pts, lbs = self.adapt_mask(pt)

        # Random selection of npoints in the masked points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        lbs = lbs[choice]

        # Separate features from xyz
        if self.features is False:
            features = np.ones((pts.shape[0], 1))
        else:
            features = pts[:, 3:]
            features = features.astype(np.float32)
        pts = pts[:, :3]

        # Data augmentation (rotation)
        if self.training:
            pts = rotate_point_cloud_z(pts)

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        return self.iterations

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzni[:, 0] < pt[0] + bs / 2, self.xyzni[:, 0] > pt[0] - bs / 2)
        mask_y = np.logical_and(self.xyzni[:, 1] < pt[1] + bs / 2, self.xyzni[:, 1] > pt[1] - bs / 2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def adapt_mask(self, pt):
        # First computation of mask and selection of points.
        mask = self.compute_mask(pt, self.bs)
        pts = self.xyzni[mask]

        # Check if total number of points in the first mask is within tolerance.
        npoints_dif = self.tolerance_percent / 100 * self.npoints
        bs_factor = pts.shape[0] / self.npoints

        if (bs_factor > (self.npoints + npoints_dif) * self.npoints) or (bs_factor < (self.npoints - npoints_dif) * self.npoints):
            # Recompute mask with new block size if outside the tolerance.
            bs = bs_factor * self.bs
            print(f"New block size of {bs:.2f} m.")
            mask = self.compute_mask(pt, bs)
            pts = self.xyzni[mask]

        lbs = self.labels[mask]
        return pts, lbs

    def format_classes(self, labels):
        """Format labels array to match the classes of interest.
        Labels with keys not defined in the coi dict will be set to 0.
        """
        labels2 = np.full(shape=labels.shape, fill_value=0, dtype=int)
        for key, value in self.class_info.items():
            labels2[labels == int(key)] = value['mode']

        return labels2


# Part dataset only for testing
class PartDatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzni[:, 0] < pt[0] + bs / 2, self.xyzni[:, 0] > pt[0] - bs / 2)
        mask_y = np.logical_and(self.xyzni[:, 1] < pt[1] + bs / 2, self.xyzni[:, 1] > pt[1] - bs / 2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__(self, filename, folder, block_size=8, npoints=8192, test_step=5, features=False, labels=True):

        self.filename = filename
        self.folder = Path(folder)
        self.bs = block_size
        self.npoints = npoints
        self.features = features
        self.step = test_step
        self.islabels= labels
        self.h5file = self.folder / f"{self.filename}.hdfs"
        # load the points
        with h5py.File(self.h5file, 'r') as data_file:
            self.xyzni = data_file["xyzni"][:]
            if self.islabels:
                self.labels = data_file["labels"][:]
            else:
                self.labels = None

            discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
            self.pts = np.unique(discretized, axis=0)
            self.pts = self.pts.astype(np.float) * self.step

    def __getitem__(self, index):
        if self.pts is None:
            with h5py.File(self.h5file, 'r') as data_file:
                self.xyzni = data_file["xyzni"][:]
                if self.islabels:
                    self.labels = data_file["labels"][:]
                else:
                    self.labels = None

                discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
                self.pts = np.unique(discretized, axis=0)
                self.pts = self.pts.astype(np.float) * self.step
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
        if self.pts is None:
            with h5py.File(self.h5file, 'r') as data_file:
                self.xyzni = data_file["xyzni"][:]
                if self.islabels:
                    self.labels = data_file["labels"][:]
                else:
                    self.labels = None

                discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
                self.pts = np.unique(discretized, axis=0)
                self.pts = self.pts.astype(np.float) * self.step
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


def train(args, dataset_dict, info_class):

    nb_class = info_class['nb_class']
    print("Creating network...")
    net, features = get_model(nb_class, args)
    net.cuda()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    print("Creating dataloader and optimizer...", end="")
    ds_trn = PartDatasetTrainVal(filelist=dataset_dict['trn'], folder=args.rootdir, training=True, block_size=args.blocksize,
                                 npoints=args.npoints, iteration_number=args.batchsize * args.iter, features=features,
                                 class_info=info_class['class_info'])
    train_loader = torch.utils.data.DataLoader(ds_trn, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    ds_val = PartDatasetTrainVal(filelist=dataset_dict['val'], folder=args.rootdir, training=False, block_size=args.blocksize,
                                 npoints=args.npoints, iteration_number=args.batchsize * args.val_iter, features=features,
                                 class_info=info_class['class_info'])
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    print("done")

    # create the root folder
    print("Creating results folder...", end="")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = Path(f"{args.savedir}/{args.model}_{args.npoints}_drop{args.drop}_{time_string}")
    root_folder.mkdir(exist_ok=True)
    args_dict = vars(args)
    args_dict['data'] = dataset_dict
    write_config(root_folder, args_dict)
    print("done at", root_folder)

    # create the log file
    trn_logs = InformationLogger('trn')
    val_logs = InformationLogger('val')

    # iterate over epochs
    for epoch in range(args.nepochs):

        #######
        # training
        net.train()

        train_loss = 0
        cm = np.zeros((nb_class, nb_class))
        t = tqdm(train_loader, ncols=150, desc="Epoch {}".format(epoch))
        for pts, features, seg in t:
            features = features.cuda()
            pts = pts.cuda()
            seg = seg.cuda()

            optimizer.zero_grad()
            outputs = net(features, pts)
            loss = F.cross_entropy(outputs.view(-1, nb_class), seg.view(-1))
            loss.backward()
            optimizer.step()

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(nb_class)))
            cm += cm_

            oa = f"{metrics.stats_overall_accuracy(cm):.4f}"
            acc = metrics.stats_accuracy_per_class(cm)
            iou = metrics.stats_iou_per_class(cm)

            train_loss += loss.detach().cpu().item()

            t.set_postfix(OA=wblue(oa), AA=wblue(f"{acc[0]:.4f}"), IOU=wblue(f"{iou[0]:.4f}"), LOSS=wblue(f"{train_loss / cm.sum():.4e}"))
        fscore = metrics.stats_f1score_per_class(cm)
        trn_metrics_values = {'loss': f"{train_loss / cm.sum():.4e}", 'acc': acc[0], 'iou': iou[0], 'fscore': fscore[0]}
        trn_class_score = {'acc': acc[1], 'iou': iou[1], 'fscore': fscore[1]}
        trn_logs.add_values(trn_metrics_values, epoch)
        trn_logs.add_values(trn_class_score, epoch, classwise=True)
        print_metric('Training', 'F1-Score', fscore)

        ######
        # validation
        net.eval()
        cm_val = np.zeros((nb_class, nb_class))
        val_loss = 0
        t = tqdm(val_loader, ncols=150, desc="  Validation epoch {}".format(epoch))
        with torch.no_grad():
            for pts, features, seg in t:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()

                outputs = net(features, pts)
                loss = F.cross_entropy(outputs.view(-1, nb_class), seg.view(-1))

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
                target_np = seg.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(nb_class)))
                cm_val += cm_

                oa_val = f"{metrics.stats_overall_accuracy(cm_val):.4f}"
                acc_val = metrics.stats_accuracy_per_class(cm_val)
                iou_val = metrics.stats_iou_per_class(cm_val)

                val_loss += loss.detach().cpu().item()

                t.set_postfix(OA=wgreen(oa_val), AA=wgreen(f"{acc_val[0]:.4f}"), IOU=wgreen(f"{iou_val[0]:.4f}"),
                              LOSS=wgreen(f"{val_loss / cm_val.sum():.4e}"))

        fscore_val = metrics.stats_f1score_per_class(cm_val)

        # save the model
        torch.save(net.state_dict(), root_folder / "state_dict.pth")

        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args':args
        }
        torch.save(state, root_folder / "state_dict.pth")

        # write the logs
        val_metrics_values = {'loss': f"{val_loss / cm_val.sum():.4e}", 'acc': acc_val[0], 'iou': iou_val[0], 'fscore': fscore_val[0]}
        val_class_score = {'acc': acc_val[1], 'iou': iou_val[1], 'fscore': fscore_val[1]}

        val_logs.add_values(val_metrics_values, epoch)
        val_logs.add_values(val_class_score, epoch, classwise=True)
        print_metric('Validation', 'F1-Score', fscore_val)

    return root_folder


def test(args, flist_test, model_folder, info_class):
    nb_class = info_class['nb_class']
    # create the network
    print("Creating network...")
    net, features = get_model(nb_class, args)
    state = torch.load(model_folder / "state_dict.pth")
    net.load_state_dict(state['state_dict'])
    net.cuda()
    net.eval()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    for filename in flist_test:
        print(filename)
        ds_tst = PartDatasetTest(filename, args.rootdir, block_size=args.blocksize,
                                 npoints=args.npoints, test_step=args.test_step, features=features, labels=args.test_labels)
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

        # Compute confusion matrix
        if args.test_labels:
            tst_logs = InformationLogger('tst')
            lbl = ds_tst.labels[:, :]

            cm = confusion_matrix(lbl.ravel(), scores.ravel(), labels=list(range(nb_class)))

            cl_acc = metrics.stats_accuracy_per_class(cm)
            cl_iou = metrics.stats_iou_per_class(cm)
            cl_fscore = metrics.stats_f1score_per_class(cm)

            print(f"Stats for test dataset:")
            print_metric('Test', 'Accuracy', cl_acc)
            print_metric('Test', 'iou', cl_iou)
            print_metric('Test', 'F1-Score', cl_fscore)
            tst_avg_score = {'loss': -1, 'acc': cl_acc[0], 'iou': cl_iou[0], 'fscore': [0]}
            tst_class_score = {'acc': cl_acc[1], 'iou': cl_iou[1], 'fscore': cl_fscore[1]}
            tst_logs.add_values(tst_avg_score, -1)
            tst_logs.add_values(tst_class_score, -1, classwise=True)

            # write error file.
            # error2ply(model_folder / f"{filename}_error.ply", xyz=xyz, labels=lbl, prediction=scores, info_class=info_class['class_info'])

        if args.savepts:
            # Save predictions
            out_folder = model_folder / 'tst'
            out_folder.mkdir(exist_ok=True)
            prediction2ply(model_folder / f"{filename}_predictions.ply", xyz=xyz, prediction=scores, info_class=info_class['class_info'])


def main():
    args = parse_args()

    # mlflow settings
    set_tracking_uri(args.mlruns_dir)
    set_experiment('ConvPoint')
    log_params(vars(args))

    # create the file lists (trn / val / tst)
    print("Create file list...")
    base_dir = Path(args.rootdir)
    dataset_dict = {'trn': [], 'val': [], 'tst': []}

    for dataset in dataset_dict.keys():
        for file in (base_dir / dataset).glob('*.hdfs'):
            dataset_dict[dataset].append(f"{dataset}/{file.stem}")

        if len(dataset_dict[dataset]) == 0:
            warnings.warn(f"{base_dir / dataset} is empty")

    print(f"Las files per dataset:\n Trn: {len(dataset_dict['trn'])} \n Val: {len(dataset_dict['val'])} \n Tst: {len(dataset_dict['tst'])}")

    info_class = class_mode(args.mode)
    # Train + Validate model
    model_folder = train(args, dataset_dict, info_class)
    # Test model
    if args.test:
        test(args, dataset_dict['tst'], model_folder, info_class)


if __name__ == '__main__':
    main()
