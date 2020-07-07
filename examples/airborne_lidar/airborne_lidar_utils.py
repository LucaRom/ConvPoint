import h5py
import os
import warnings
import yaml
from mlflow import log_metric


def tsv_line(*args):
    return '\t'.join(map(str, args)) + '\n'


def write_features(file_name, xyzni, labels= None):
    """write the geometric features, labels and clouds in a h5 file"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('xyzni', data=xyzni, dtype='float16')
    if labels is not None:
        data_file.create_dataset('labels', data=labels, dtype='uint8')
    data_file.close()


class InformationLogger(object):
    def __init__(self, mode):
        self.mode = mode

    def add_values(self, metrics, epoch, classwise=False):
        """Add new information to the logs."""
        if classwise:
            for key, val in metrics.items():
                for cls in val:
                    log_metric(key=f"{self.mode}_{key}_{val.index(cls)}", value=cls.avg, step=epoch)

        else:
            for key, val in metrics.items():
                log_metric(key=f"{self.mode}_{key}", value=val, step=epoch)


def print_metric(mode, metric, values):
    print(f"\n{mode} {metric}:\n  Overall: {values[0]:.3f}\n  Per class: {values[1]}")


def write_config(folder, args):
    with open(folder / 'config.yaml', 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
