# Airborne lidar example with ConvPoint

# add the parent folder to the python path to access convpoint library
import sys
import warnings
sys.path.append('/wspace/disk01/lidar/convpoint_test/ConvPoint/convpoint')

import argparse
import os
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
from examples.airborne_lidar.airborne_lidar_utils import get_airborne_lidar_info, InformationLogger, print_metric, write_config
import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=True)
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--savedir", default='/wspace/disk01/lidar/convpoint_tests/results', type=str)
    parser.add_argument("--rootdir", default='/wspace/disk01/lidar/convpoint_tests/prepared', type=str)
    parser.add_argument("--batchsize", "-b", default=10, type=int)
    parser.add_argument("--npoints", default=8168, type=int, help="Number of points to be sampled in the block.")
    parser.add_argument("--blocksize", default=50, type=int, help="Size of the infinite vertical column, to be processed.")
    parser.add_argument("--iter", default=100, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--features", default="xyzni", type=str, help="Features to process. xyzni means xyz + number of returns + intensity. "
                                                                      "Currently, only xyz and xyzni are supported for this dataset.")
    parser.add_argument("--test_step", default=15, type=float)
    parser.add_argument("--test_labels", default=True, type=bool, help="Labels available for test dataset")
    parser.add_argument("--val_iter", default=100, type=int, help="Number of iterations at validation.")
    parser.add_argument("--nepochs", default=1, type=int)
    parser.add_argument("--model", default="SegBig", type=str, help="SegBig is the only available model at this time, for this dataset.")
    parser.add_argument("--drop", default=0, type=float)

    parser.add_argument("--lr", default=1e-3, help="Learning rate")
    parser.add_argument("--mode", default=1, type=int, help="Class mode. Currently 2 choices available. "
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


def mode_selection(class_mode):
    """
    # Dict containing the mapping of input (from the .las file) and the output classes (for the training step).
    # 6: Building
    # 9: water
    # 2: ground
    # 3: low vegetation
    # 4: medium vegetation
    # 5: high vegetation
    """
    if class_mode == 1:
        coi = {'6': 1, '9': 2, '2': 3}
    elif class_mode == 2:
        coi = {'6': 1, '9': 2, '2': 3, '3': 4, '4': 5, '5': 5}
    else:
        raise ValueError(f"Class formatting provided ({class_mode}) is not defined.")

    nb_class = np.unique([x for x in coi.values()])
    return coi, len(nb_class) + 1


# Part dataset only for training / validation
class PartDatasetTrainVal():

    def __init__(self, filelist, folder, training, block_size, npoints, iteration_number, features, coi):

        self.filelist = filelist
        self.folder = folder
        self.training = training
        self.bs = block_size
        self.npoints = npoints
        self.iterations = iteration_number
        self.features = features
        self.coi = coi

    def __getitem__(self, index):

        # Load data
        index = random.randint(0, len(self.filelist) - 1)
        dataset = self.filelist[index]
        data_file = h5py.File(os.path.join(self.folder, dataset), 'r')

        # Get the features
        xyzni = data_file["xyzni"][:]
        labels = data_file["labels"][:]
        labels = self.format_classes(labels)

        # pick a random point
        pt_id = random.randint(0, xyzni.shape[0] - 1)
        pt = xyzni[pt_id, :3]

        # Create the mask
        mask_x = np.logical_and(xyzni[:, 0] < pt[0] + self.bs / 2, xyzni[:, 0] > pt[0] - self.bs / 2)
        mask_y = np.logical_and(xyzni[:, 1] < pt[1] + self.bs / 2, xyzni[:, 1] > pt[1] - self.bs / 2)
        mask = np.logical_and(mask_x, mask_y)
        pts = xyzni[mask]
        lbs = labels[mask]
        # print(pts.shape)
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

    def format_classes(self, labels):
        """Format labels array to match the classes of interest.
        Labels with keys not defined in the coi dict will be set to 0.
        """
        labels2 = np.full(shape=labels.shape, fill_value=0, dtype=int)
        for key, value in self.coi.items():
            labels2[labels == int(key)] = value

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
        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.features = features
        self.step = test_step

        # load the points
        data_file = h5py.File(os.path.join(self.folder, filename), 'r')
        self.xyzni = data_file["xyzni"][:]
        if labels:
            self.labels = data_file["labels"][:]
        else:
            self.labels = None

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


def train(args, dataset_dict):
    coi, nb_classes = mode_selection(args.mode)

    print("Creating network...")
    net, features = get_model(nb_classes, args)
    net.cuda()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    print("Creating dataloader and optimizer...", end="")
    ds_trn = PartDatasetTrainVal(filelist=dataset_dict['trn'], folder=args.rootdir, training=True, block_size=args.blocksize,
                                 npoints=args.npoints, iteration_number=args.batchsize * args.iter, features=features, coi=coi)
    train_loader = torch.utils.data.DataLoader(ds_trn, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    ds_val = PartDatasetTrainVal(filelist=dataset_dict['val'], folder=args.rootdir, training=False, block_size=args.blocksize,
                                 npoints=args.npoints, iteration_number=args.batchsize * args.val_iter, features=features, coi=coi)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    print("done")

    # create the root folder
    print("Creating results folder...", end="")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, f"{args.model}_{args.npoints}_drop{args.drop}_{time_string}")
    os.makedirs(root_folder, exist_ok=True)
    args_dict = vars(args)
    args_dict['data'] = dataset_dict
    write_config(root_folder, args_dict)
    print("done at", root_folder)

    # create the log file
    trn_logs = InformationLogger(root_folder, 'trn')
    val_logs = InformationLogger(root_folder, 'val')

    # iterate over epochs
    for epoch in range(args.nepochs):

        #######
        # training
        net.train()

        train_loss = 0
        cm = np.zeros((nb_classes, nb_classes))
        t = tqdm(train_loader, ncols=150, desc="Epoch {}".format(epoch))
        for pts, features, seg in t:
            features = features.cuda()
            pts = pts.cuda()
            seg = seg.cuda()

            optimizer.zero_grad()
            outputs = net(features, pts)
            loss = F.cross_entropy(outputs.view(-1, nb_classes), seg.view(-1))
            loss.backward()
            optimizer.step()

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(nb_classes)))
            cm += cm_

            oa = f"{metrics.stats_overall_accuracy(cm):.4f}"
            acc = metrics.stats_accuracy_per_class(cm)
            iou = metrics.stats_iou_per_class(cm)

            train_loss += loss.detach().cpu().item()

            t.set_postfix(OA=wblue(oa), AA=wblue(f"{acc[0]:.4f}"), IOU=wblue(f"{iou[0]:.4f}"), LOSS=wblue(f"{train_loss / cm.sum():.4e}"))
        fscore = metrics.stats_f1score_per_class(cm)
        trn_metrics_values = {'loss': f"{train_loss / cm.sum():.4e}", 'acc': acc[0], 'iou': iou[0], 'fscore': fscore[0]}
        trn_class_score = {'acc': acc[1], 'iou': iou[1], 'fscore': fscore[1]}
        trn_logs.add_metric_values(trn_metrics_values, epoch)
        trn_logs.add_class_scores(trn_class_score, epoch)
        print_metric('Training', 'F1-Score', fscore)

        ######
        # validation
        net.eval()
        cm_val = np.zeros((nb_classes, nb_classes))
        val_loss = 0
        t = tqdm(val_loader, ncols=150, desc="  Validation epoch {}".format(epoch))
        with torch.no_grad():
            for pts, features, seg in t:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()

                outputs = net(features, pts)
                loss = F.cross_entropy(outputs.view(-1, nb_classes), seg.view(-1))

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
                target_np = seg.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(nb_classes)))
                cm_val += cm_

                oa_val = f"{metrics.stats_overall_accuracy(cm_val):.4f}"
                acc_val = metrics.stats_accuracy_per_class(cm_val)
                iou_val = metrics.stats_iou_per_class(cm_val)

                val_loss += loss.detach().cpu().item()

                t.set_postfix(OA=wgreen(oa_val), AA=wgreen(f"{acc_val[0]:.4f}"), IOU=wgreen(f"{iou_val[0]:.4f}"),
                              LOSS=wgreen(f"{val_loss / cm_val.sum():.4e}"))

        fscore_val = metrics.stats_f1score_per_class(cm_val)

        # save the model
        torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))

        # write the logs
        val_metrics_values = {'loss': f"{val_loss / cm_val.sum():.4e}", 'acc': acc_val[0], 'iou': iou_val[0], 'fscore': fscore_val[0]}
        val_class_score = {'acc': acc_val[1], 'iou': iou_val[1], 'fscore': fscore_val[1]}

        val_logs.add_metric_values(val_metrics_values, epoch)
        val_logs.add_class_scores(val_class_score, epoch)
        print_metric('Validation', 'F1-Score', fscore_val)

    return root_folder


def test(args, flist_test, model_folder):
    _, nb_classes = mode_selection(args.mode)

    # create the network
    print("Creating network...")
    net, features = get_model(nb_classes, args)

    net.load_state_dict(torch.load(os.path.join(model_folder, "state_dict.pth")))
    net.cuda()
    net.eval()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    for filename in flist_test:
        print(filename)
        ds_tst = PartDatasetTest(filename, args.rootdir, block_size=args.blocksize,
                                 npoints=args.npoints, test_step=args.test_step, features=features, labels=args.test_labels)
        tst_loader = torch.utils.data.DataLoader(ds_tst, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

        xyz = ds_tst.xyzni[:, :3]
        scores = np.zeros((xyz.shape[0], nb_classes))

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

                outputs_np = outputs.cpu().numpy().reshape((-1, nb_classes))
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
            tst_logs = InformationLogger(model_folder, 'tst')
            lbl = ds_tst.labels[:, :]

            cm = confusion_matrix(lbl.ravel(), scores.ravel(), labels=list(range(nb_classes)))

            cl_acc = metrics.stats_accuracy_per_class(cm)
            cl_iou = metrics.stats_iou_per_class(cm)
            cl_fscore = metrics.stats_f1score_per_class(cm)

            print(f"Stats for test dataset:")
            print_metric('Test', 'Accuracy', cl_acc)
            print_metric('Test', 'iou', cl_iou)
            print_metric('Test', 'F1-Score', cl_fscore)
            tst_avg_score = {'loss': -1, 'acc': cl_acc[0], 'iou': cl_iou[0], 'fscore': [0]}
            tst_class_score = {'acc': cl_acc[1], 'iou': cl_iou[1], 'fscore': cl_fscore[1]}
            tst_logs.add_metric_values(tst_avg_score, -1)
            tst_logs.add_class_scores(tst_class_score, -1)

        os.makedirs(os.path.join(model_folder, filename), exist_ok=True)

        # saving labels
        save_fname = os.path.join(model_folder, filename, "pred.txt")
        np.savetxt(save_fname, scores, fmt='%d')

        if args.savepts:
            save_fname = os.path.join(model_folder, filename, "pts.txt")
            xyzni = np.concatenate([xyz, np.expand_dims(scores, 1)], axis=1)
            np.savetxt(save_fname, xyzni, fmt=['%.4f', '%.4f', '%.4f', '%d'])


def main():

    args = parse_args()

    # create the file lists (trn / val / tst)
    print("Create file list...")
    base_dir = args.rootdir

    dataset_dict = {'trn': [], 'val': [], 'tst': []}

    for dataset in dataset_dict.keys():
        for f in os.listdir(os.path.join(base_dir, dataset)):
            dataset_dict[dataset].append(f"{dataset}/{f}")

        if len(dataset_dict[dataset]) == 0:
            warnings.warn(f"{os.path.join(base_dir, dataset)} is empty")

    print(f"Las files per dataset:\n Trn: {len(dataset_dict['trn'])} \n Val: {len(dataset_dict['val'])} \n Tst: {len(dataset_dict['tst'])}")

    model_folder = train(args, dataset_dict)

    if args.test:
        test(args, dataset_dict['tst'], model_folder)


if __name__ == '__main__':
    main()
