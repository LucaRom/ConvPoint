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
from examples.airborne_lidar.airborne_lidar_utils import get_airborne_lidar_info
import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--savedir", default='/wspace/disk01/lidar/convpoint_tests/results', type=str)
    parser.add_argument("--rootdir", default='/wspace/disk01/lidar/convpoint_tests/prepared', type=str)
    parser.add_argument("--batchsize", "-b", default=16, type=int)
    parser.add_argument("--npoints", default=8168, type=int, help="Number of points to be sampled in the block.")
    parser.add_argument("--blocksize", default=25, type=int, help="Size of the infinite vertical column, to be processed.")
    parser.add_argument("--iter", default=1000, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--features", default="xyzni", type=str, help="Features to process. xyzni means xyz + number of returns + intensity. "
                                                                      "Currently, only xyz and xyzni are supported for this dataset.")
    parser.add_argument("--test_step", default=5, type=float)
    parser.add_argument("--nepochs", default=50, type=int)
    parser.add_argument("--model", default="SegBig", type=str, help="SegBig is the only available model at this time, for this dataset.")
    parser.add_argument("--drop", default=0, type=float)

    parser.add_argument("--lr", default=1e-3, help="Learning rate")
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


# Part dataset only for training / validation
class PartDatasetTrainVal():

    def __init__(self, filelist, folder, training, block_size, npoints, iteration_number, features):

        self.filelist = filelist
        self.folder = folder
        self.training = training
        self.bs = block_size
        self.npoints = npoints
        self.iterations = iteration_number
        self.features = features

    def __getitem__(self, index):

        # Load data
        index = random.randint(0, len(self.filelist) - 1)
        dataset = self.filelist[index]
        data_file = h5py.File(os.path.join(self.folder, dataset), 'r')

        # Get the features
        xyzni = data_file["xyzni"][:]
        labels = data_file["labels"][:]

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


# Part dataset only for testing
class PartDatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzni[:, 0] < pt[0] + bs / 2, self.xyzni[:, 0] > pt[0] - bs / 2)
        mask_y = np.logical_and(self.xyzni[:, 1] < pt[1] + bs / 2, self.xyzni[:, 1] > pt[1] - bs / 2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__(self, filename, folder, block_size=8, npoints=8192, test_step=5, features=False, labels=False):

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

        if self.labels is not None:
            lbs = self.labels[choice]
        else:
            # labels will contain indices in the original point cloud
            lbs = np.where(mask)[0][choice]

        # separate between features and points
        if self.features is False:
            fts = np.ones((pts.shape[0], 1))
        else:
            fts = pts[:, 3:]
            fts = fts.astype(np.float32)

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

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


def train(args, flist_trn, flist_val):
    obj_classes = get_airborne_lidar_info()
    nb_classes = len(obj_classes)

    print("Creating network...")
    net, features = get_model(nb_classes, args)

    net.cuda()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    print("Creating dataloader and optimizer...", end="")
    ds_trn = PartDatasetTrainVal(filelist=flist_trn, folder=args.rootdir, training=True, block_size=args.blocksize,
                                 npoints=args.npoints, iteration_number=args.batchsize * args.iter, features=features)
    train_loader = torch.utils.data.DataLoader(ds_trn, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    ds_val = PartDatasetTrainVal(filelist=flist_val, folder=args.rootdir, training=False, block_size=args.blocksize,
                                 npoints=args.npoints, iteration_number=round(args.batchsize * (args.iter / 10)), features=features)
    test_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    print("done")

    # create the root folder
    print("Creating results folder...", end="")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, f"{args.model}_{args.npoints}_drop{args.drop}_{time_string}")
    os.makedirs(root_folder, exist_ok=True)
    print("done at", root_folder)

    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")
    per_class_log = open(os.path.join(root_folder, "per_class_fscore_log.txt"), "w")

    # Write logs headers
    logs.write(f"epoch oa aa iou oa_val aa_val iou_val\n")
    per_class_log.write(f"Overall Other Building Water Ground\n")

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
            aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.4f}"
            iou = f"{metrics.stats_iou_per_class(cm)[0]:.4f}"

            train_loss += loss.detach().cpu().item()

            t.set_postfix(OA=wblue(oa), AA=wblue(aa), IOU=wblue(iou), LOSS=wblue(f"{train_loss / cm.sum():.4e}"))
        fscore = metrics.stats_f1score_per_class(cm)
        print(f"\nTraining F1-scores:\n  Overall: {fscore[0]:.3f}\n  Other: {fscore[1][0]:.3f}\n  "
              f"Building: {fscore[1][1]:.3f}\n  Water: {fscore[1][2]:.3f}\n  Ground: {fscore[1][3]:.3f}")
        ######
        # validation
        net.eval()
        cm_test = np.zeros((nb_classes, nb_classes))
        test_loss = 0
        t = tqdm(test_loader, ncols=150, desc="  Test epoch {}".format(epoch))
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
                cm_test += cm_

                oa_val = f"{metrics.stats_overall_accuracy(cm_test):.4f}"
                aa_val = f"{metrics.stats_accuracy_per_class(cm_test)[0]:.4f}"
                iou_val = f"{metrics.stats_iou_per_class(cm_test)[0]:.4f}"

                test_loss += loss.detach().cpu().item()

                t.set_postfix(OA=wgreen(oa_val), AA=wgreen(aa_val), IOU=wgreen(iou_val),
                              LOSS=wgreen(f"{test_loss / cm_test.sum():.4e}"))

        fscore = metrics.stats_f1score_per_class(cm_test)
        print(f"\nValidation F1-scores:\n  Overall: {fscore[0]:.3f}\n  Other: {fscore[1][0]:.3f}\n  "
              f"Building: {fscore[1][1]:.3f}\n  Water: {fscore[1][2]:.3f}\n  Ground: {fscore[1][3]:.3f}")

        # save the model
        torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))

        # write the logs
        logs.write(f"{epoch} {oa} {aa} {iou} {oa_val} {aa_val} {iou_val}\n")
        per_class_log.write(f"{fscore[0]:.3f} {fscore[1][0]:.3f} {fscore[1][1]:.3f} {fscore[1][2]:.3f} {fscore[1][3]:.3f}")
        logs.flush()

    logs.close()
    return root_folder


def test(args, flist_test, model_folder):
    obj_classes = get_airborne_lidar_info()
    nb_classes = len(obj_classes)

    # create the network
    print("Creating network...")
    net, features = get_model(nb_classes, args)

    net.load_state_dict(torch.load(os.path.join(model_folder, "state_dict.pth")))
    net.cuda()
    net.eval()
    print(f"Number of parameters in the model: {count_parameters(net)}")

    for filename in flist_test:
        print(filename)
        ds_tst = PartDatasetTest(filename, args.rootdir, block_size=args.blocksize,
                                 npoints=args.npoints, test_step=args.test_step, features=features, labels=True)
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

        os.makedirs(os.path.join(args.savedir, filename), exist_ok=True)

        # saving labels
        save_fname = os.path.join(args.savedir, filename, "pred.txt")
        scores = scores.argmax(1)
        np.savetxt(save_fname, scores, fmt='%d')

        if args.savepts:
            save_fname = os.path.join(args.savedir, filename, "pts.txt")
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

    model_folder = train(args, dataset_dict['trn'], dataset_dict['val'])

    if args.test:
        test(args, dataset_dict['tst'], model_folder)


if __name__ == '__main__':
    main()
