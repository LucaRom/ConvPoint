# Airborne lidar example with ConvPoint

# add the parent folder to the python path to access convpoint library
import sys
import warnings
sys.path.append('/space/partner/nrcan/geobase/work/transfer/work/deep_learning/lidar/CMM_2018/convpoint_tests/ConvPoint')

import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time
import torch
import torch.utils.data
import torch.nn.functional as F
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors
import utils.metrics as metrics
from examples.airborne_lidar.airborne_lidar_utils import InformationLogger, print_metric, write_config, read_parameters, wblue, wgreen
from pathlib import Path
from examples.airborne_lidar.airborne_lidar_viz import prediction2ply, error2ply
from examples.airborne_lidar.airborne_lidar_datasets import PartDatasetTrainVal, PartDatasetTest
from mlflow import log_params, set_tracking_uri, set_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./config_template.yaml', type=str)
    args = parser.parse_args()
    conf = args.config
    args = read_parameters(conf)
    args['global']['config_file'] = conf
    return args


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


def get_model(nb_classes, args):
    # Select the model
    if args['training']['model'] == "SegBig":
        from networks.network_seg import SegBig as Net
    else:
        raise NotImplemented(f"The model {args['training']['model']} does not exist. Only SegBig is available at this time.")

    # Number of features as input
    if args['training']['features'] == "xyzni":
        input_channels = 2
        features = True
        if args['training']['local_features']:
            input_channels = 4
    elif args['training']['features'] == "xyz":
        input_channels = 1
        features = False
        if args['training']['local_features']:
            input_channels = 3
    else:
        raise NotImplemented(f"Features {args['training']['features']} are not supported. Only xyzni or xyz, at this time.")

    return Net(input_channels, output_channels=nb_classes, args=args), features


def train(args, dataset_dict, info_class):

    nb_class = info_class['nb_class']
    print("Creating network...")
    net, features = get_model(nb_class, args)
    net.cuda()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    print("Creating dataloader and optimizer...", end="")
    ds_trn = PartDatasetTrainVal(filelist=dataset_dict['trn'], folder=args['global']['rootdir'], training=True,
                                 block_size=args['training']['blocksize'], npoints=args['training']['npoints'],
                                 iteration_number=args['training']['batchsize'] * args['training']['trn_iter'], features=features,
                                 class_info=info_class['class_info'], tolerance_range=args['training']['tolerance'],
                                 local_info=args['training']['local_features'])
    train_loader = torch.utils.data.DataLoader(ds_trn, batch_size=args['training']['batchsize'], shuffle=True,
                                               num_workers=args['training']['num_workers'])

    ds_val = PartDatasetTrainVal(filelist=dataset_dict['val'], folder=args['global']['rootdir'], training=False,
                                 block_size=args['training']['blocksize'], npoints=args['training']['npoints'],
                                 iteration_number=args['training']['batchsize'] * args['training']['val_iter'], features=features,
                                 class_info=info_class['class_info'], tolerance_range=args['training']['tolerance'],
                                 local_info=args['training']['local_features'])
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args['training']['batchsize'], shuffle=False,
                                             num_workers=args['training']['num_workers'])

    optimizer = torch.optim.Adam(net.parameters(), lr=float(args['training']['lr']))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args['training']['step_size'], gamma=args['training']['gamma'])
    print("done")

    # create the root folder
    print("Creating results folder...", end="")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = Path(f"{args['global']['savedir']}/{args['training']['model']}_{args['training']['npoints']}_"
                       f"mode{args['training']['mode']}_{time_string}")
    root_folder.mkdir(exist_ok=True)
    args['data'] = dataset_dict
    write_config(args['global']['config_file'], root_folder)
    print("done at", root_folder)

    # create the log file
    trn_logs = InformationLogger('trn')
    val_logs = InformationLogger('val')

    # iterate over epochs
    for epoch in range(args['training']['nepochs']):

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

        lr_scheduler.step()
        trn_logs.add_values({'loss': f"{train_loss / cm.sum():.4e}", 'iou': iou[0]}, epoch)
        trn_logs.add_values({'iou': iou[1]}, epoch, classwise=True)

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
            'args': args
        }
        torch.save(state, root_folder / "state_dict.pth")

        # write the logs
        val_metrics_values = {'loss': f"{val_loss / cm_val.sum():.4e}", 'acc': acc_val[0], 'iou': iou_val[0], 'fscore': fscore_val[0]}
        val_class_score = {'acc': acc_val[1], 'iou': iou_val[1], 'fscore': fscore_val[1]}

        val_logs.add_values(val_metrics_values, epoch)
        val_logs.add_values(val_class_score, epoch, classwise=True)
        print_metric('Validation', 'F1-Score', fscore_val)

    return root_folder


def format_classes(labels, class_info):
    """Format labels array to match the classes of interest.
    Labels with keys not defined in the coi dict will be set to 0.
    """
    labels2 = np.full(shape=labels.shape, fill_value=0, dtype=int)
    for key, value in class_info.items():
        labels2[labels == int(key)] = value['mode']

    return labels2


def test(args, filename, model_folder, info_class, file_idx):
    nb_class = info_class['nb_class']
    # create the network
    print("Creating network...")
    net, features = get_model(nb_class, args)
    state = torch.load(model_folder / "state_dict.pth")
    net.load_state_dict(state['state_dict'])
    net.cuda()
    net.eval()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    print(filename)
    ds_tst = PartDatasetTest(filename, args['global']['rootdir'], block_size=args['training']['blocksize'], npoints=args['training']['npoints'],
                             step=args['test']['test_step'], features=features, tolerance=args['training']['tolerance'],
                             local_features=args['training']['local_features'])
    tst_loader = torch.utils.data.DataLoader(ds_tst, batch_size=args['training']['batchsize'], shuffle=False,
                                             num_workers=args['training']['num_workers'])

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
            t.set_postfix(time=f"{total_time / (iter_nb * args['training']['batchsize']):05e}")

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
    if args['test']['test_labels']:
        tst_logs = InformationLogger('tst')
        lbl = format_classes(ds_tst.labels[:, :], class_info=info_class['class_info'])

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
        tst_logs.add_values(tst_avg_score, file_idx)
        tst_logs.add_values(tst_class_score, file_idx, classwise=True)

        # write error file.
        # error2ply(model_folder / f"{filename}_error.ply", xyz=xyz, labels=lbl, prediction=scores, info_class=info_class['class_info'])

    if args['test']['savepts']:
        # Save predictions
        out_folder = model_folder / 'tst'
        out_folder.mkdir(exist_ok=True)
        prediction2ply(model_folder / f"{filename}_predictions.ply", xyz=xyz, prediction=scores, info_class=info_class['class_info'])


def main():
    args = parse_args()
    # mlflow settings
    set_tracking_uri(args['global']['mlruns_dir'])
    set_experiment(args['global']['exp_name'])
    log_params(args['global'])
    log_params(args['training'])
    log_params(args['test'])

    # create the file lists (trn / val / tst)
    print("Create file list...")
    base_dir = Path(args['global']['rootdir'])
    dataset_dict = {'trn': [], 'val': [], 'tst': []}

    for dataset in dataset_dict.keys():
        for file in (base_dir / dataset).glob('*.hdfs'):
            dataset_dict[dataset].append(f"{dataset}/{file.stem}")

        if len(dataset_dict[dataset]) == 0:
            warnings.warn(f"{base_dir / dataset} is empty")

    print(f"Las files per dataset:\n Trn: {len(dataset_dict['trn'])} \n Val: {len(dataset_dict['val'])} \n Tst: {len(dataset_dict['tst'])}")

    info_class = class_mode(args['training']['mode'])
    if args['test']['test_model'] is None:
        # Train + Validate model
        model_folder = train(args, dataset_dict, info_class)

    else:
        # Test only
        model_folder = Path(args['test']['test_model'])

    # Test model
    if args['test']['test']:
        for filename in dataset_dict['tst']:
            test(args, filename, model_folder, info_class, dataset_dict['tst'].index(filename))


if __name__ == '__main__':
    main()
