import numpy as np
import random
import torch
import torch.utils.data
import h5py
from pathlib import Path
from math import sqrt


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


def compute_mask(xyzni, pt, bs):
    # build the mask
    mask_x = np.logical_and(xyzni[:, 0] < pt[0] + bs / 2, xyzni[:, 0] > pt[0] - bs / 2)
    mask_y = np.logical_and(xyzni[:, 1] < pt[1] + bs / 2, xyzni[:, 1] > pt[1] - bs / 2)
    mask = np.logical_and(mask_x, mask_y)
    return mask, mask_x, mask_y


def compute_large_mask(xyzni, pt, bs, mask_x, mask_y):
    # build the mask
    mask_x = np.logical_and(xyzni[:, 0] < pt[0] + bs / 2, xyzni[:, 0] > pt[0] - bs / 2, np.logical_not(mask_x))
    mask_y = np.logical_and(xyzni[:, 1] < pt[1] + bs / 2, xyzni[:, 1] > pt[1] - bs / 2, np.logical_not(mask_y))
    mask = np.logical_and(mask_x, mask_y)
    return mask


# Part dataset only for training / validation
class PartDatasetTrainVal():

    def __init__(self, filelist, folder, training, block_size, npoints, iteration_number, features, class_info, tolerance_range, local_info):

        self.filelist = filelist
        self.folder = Path(folder)
        self.training = training
        self.bs = block_size
        self.npoints = npoints
        self.iterations = iteration_number
        self.features = features
        self.class_info = class_info
        self.tolerance_range = tolerance_range
        self.xyzni = None
        self.labels = None
        self.local_info = local_info

    def __getitem__(self, index):

        # Load data
        index = random.randint(0, len(self.filelist) - 1)
        dataset = self.filelist[index]
        data_file = h5py.File(self.folder / f"{dataset}.hdfs", 'r')

        # Get the features
        self.xyzni = data_file["xyzni"][:]
        self.labels = self.format_classes(data_file["labels"][:])

        # pick a random point
        pt_id = random.randint(0, self.xyzni.shape[0] - 1)
        pt = self.xyzni[pt_id, :3]

        # # Create the mask and select all points in the column.
        # pts, lbs, local_info = self.adapt_mask(pt)
        #
        # # Random selection of npoints in the masked points
        # choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        # pts = pts[choice]
        # lbs = lbs[choice]

        pts, lbs, local_info = self.multi_scale(pt)

        # Separate features from xyz
        if self.features is False:
            features = np.ones((pts.shape[0], 1))
        else:
            features = pts[:, 3:]
        if self.local_info:
            dens = np.full(shape=(features.shape[0], 1), fill_value=local_info['density'])
            bs = np.full(shape=(features.shape[0], 1), fill_value=local_info['bs'])
            features = np.hstack((features, dens, bs))

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

    def multi_scale(self, pt):
        # First computation of mask and selection of points.
        mask, mx, my = compute_mask(self.xyzni, pt, self.bs)
        pts = self.xyzni[mask]
        lbs = self.labels[mask]
        local_density = max(int(pts.shape[0] / self.bs ** 2), 1)

        # Random selection of npoints in the masked points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        lbs = lbs[choice]

        # Second computation of mask and selection of points. (second scale)
        mask = compute_large_mask(self.xyzni, pt, self.bs*2, mx, my)
        pts_2 = self.xyzni[mask]
        lbs_2 = self.labels[mask]

        # Random selection of npoints in the masked points
        choice = np.random.choice(pts_2.shape[0], self.npoints, replace=True)

        pts = np.concatenate((pts_2[choice], pts))
        lbs = np.concatenate((lbs_2[choice], lbs))
        return pts, lbs, {'density': local_density, 'bs': self.bs}

    # def adapt_mask(self, pt):
    #     # First computation of mask and selection of points.
    #     mask = compute_mask(self.xyzni, pt, self.bs)
    #     pts = self.xyzni[mask]
    #
    #     # Check if total number of points in the first mask is within tolerance.
    #     local_pt_num = pts.shape[0]
    #     local_density = max(int(local_pt_num / self.bs ** 2), 1)
    #     pts_num_ratio = self.npoints / local_pt_num
    #
    #     # Recompute mask with new block size if outside the tolerance.
    #     if (local_pt_num > (1 + int(self.tolerance_range[1]) / 100) * self.npoints) or (local_pt_num < (1 - int(self.tolerance_range[0]) / 100) * self.npoints):
    #         bs = sqrt(pts_num_ratio) * self.bs
    #         mask = compute_mask(self.xyzni, pt, bs)
    #         pts = self.xyzni[mask]
    #     else:
    #         bs = self.bs
    #
    #     lbs = self.labels[mask]
    #     return pts, lbs, {'density': local_density, 'bs': bs}

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

    def __init__(self, filename, folder, block_size, npoints, step, features, tolerance, local_features):

        self.filename = filename
        self.folder = Path(folder)
        self.bs = block_size
        self.npoints = npoints
        self.features = features
        self.step = step
        self.tolerance_range = tolerance
        self.local_info = local_features
        self.h5file = self.folder / f"{self.filename}.hdfs"
        # load the points
        with h5py.File(self.h5file, 'r') as data_file:
            self.xyzni = data_file["xyzni"][:]
            self.labels = data_file['labels'][:]

            discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
            self.pts = np.unique(discretized, axis=0)
            self.pts = self.pts.astype(np.float) * self.step

    def __getitem__(self, index):
        if self.pts is None:
            with h5py.File(self.h5file, 'r') as data_file:
                self.xyzni = data_file["xyzni"][:]
                self.labels = data_file['labels'][:]

                discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
                self.pts = np.unique(discretized, axis=0)
                self.pts = self.pts.astype(np.float) * self.step

        # # get all points within block
        # pts, local_info, mask = self.adapt_mask(self.pts[index])
        #
        # # choose right number of points
        # choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        # pts = pts[choice]

        # # indices in the original point cloud
        # indices = np.where(mask)[0][choice]

        pts, local_info, indices = self.multi_scale(self.pts[index])

        # Separate features from xyz
        if self.features is False:
            fts = np.ones((pts.shape[0], 1))
        else:
            fts = pts[:, 3:]
        if self.local_info:
            dens = np.full(shape=(fts.shape[0], 1), fill_value=local_info['density'])
            bs = np.full(shape=(fts.shape[0], 1), fill_value=local_info['bs'])
            fts = np.hstack((fts, dens, bs))

        fts = fts.astype(np.float32)

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        indices = torch.from_numpy(indices).long()

        return pts, fts, indices

    def multi_scale(self, pt):
        # First computation of mask and selection of points.
        mask, mx, my = compute_mask(self.xyzni, pt, self.bs)
        pts = self.xyzni[mask]
        local_density = max(int(pts.shape[0] / self.bs ** 2), 1)

        # Random selection of npoints in the masked points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # Second computation of mask and selection of points. (second scale)
        mask_2 = compute_large_mask(self.xyzni, pt, self.bs * 2, mx, my)
        pts_2 = self.xyzni[mask_2]

        # Random selection of npoints in the masked points
        choice_2 = np.random.choice(pts_2.shape[0], self.npoints, replace=True)
        pts = np.concatenate((pts_2[choice], pts))
        # mask = np.concatenate((mask, mask_2))

        # indices in the original point cloud
        indices = np.where(mask)[0][choice]
        indices_2 = np.where(mask_2)[0][choice_2]
        np.concatenate((indices, indices_2))

        return pts, {'density': local_density, 'bs': self.bs}, indices

    # def adapt_mask(self, pt):
    #     # First computation of mask and selection of points.
    #     mask = compute_mask(self.xyzni, pt, self.bs)
    #     pts = self.xyzni[mask]
    #
    #     # Check if total number of points in the first mask is within tolerance.
    #     local_pt_num = max(pts.shape[0], 1)
    #     local_density = max(int(local_pt_num / self.bs ** 2), 1)
    #     pts_num_ratio = self.npoints / local_pt_num
    #
    #     # Recompute mask with new block size if outside the tolerance.
    #     if (local_pt_num > (1 + int(self.tolerance_range[1]) / 100) * self.npoints) or \
    #             (local_pt_num < (1 - int(self.tolerance_range[0]) / 100) * self.npoints) or \
    #             (pts.shape[0] == 0):
    #         bs = max(sqrt(pts_num_ratio) * self.bs, 1)
    #         mask = compute_mask(self.xyzni, pt, bs)
    #         pts = self.xyzni[mask]
    #     else:
    #         bs = self.bs
    #
    #     return pts, {'density': local_density, 'bs': bs}, mask

    def __len__(self):
        if self.pts is None:
            with h5py.File(self.h5file, 'r') as data_file:
                self.xyzni = data_file["xyzni"][:]
                self.labels = data_file['labels'][:]

                discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
                self.pts = np.unique(discretized, axis=0)
                self.pts = self.pts.astype(np.float) * self.step
        return len(self.pts)
