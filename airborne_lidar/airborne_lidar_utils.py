import h5py
import yaml
from mlflow import log_metric
from pathlib import Path
import shutil
import warnings
import numpy as np

try:
    from pynvml import *
except ModuleNotFoundError:
    warnings.warn(f"The python Nvidia management library could not be imported. Ignore if running on CPU only.")


def gpu_stats(device=0):
    """
    Provides GPU utilization (%) and RAM usage
    :return: res.gpu, res.memory
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device)
    res = nvmlDeviceGetUtilizationRates(handle)
    mem = nvmlDeviceGetMemoryInfo(handle)

    return res, mem


def read_parameters(param_file):
    """Read and return parameters in .yaml file
    Args:
        param_file: Full file path of the parameters file
    Returns:
        YAML: CommentedMap dict-like object
    """
    with open(param_file) as yamlfile:
        params = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return params


def tsv_line(*args):
    return '\t'.join(map(str, args)) + '\n'


def write_features(file_name, xyzni, labels=None):
    """write the geometric features, labels and clouds in a h5 file"""
    if Path.is_file(file_name):
        Path.unlink(file_name)
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('xyzni', data=xyzni, dtype='float16')
    if labels is not None:
        data_file.create_dataset('labels', data=labels, dtype='uint8')
    else:
        data_file.create_dataset('labels', data=np.zeros((1, 1)))
    data_file.close()


class InformationLogger(object):
    def __init__(self, mode):
        self.mode = mode

    def add_values(self, metrics, step, classwise=False):
        """Add new information to the logs."""
        if classwise:
            for key, val in metrics.items():
                for cls in val:
                    log_metric(key=f"{self.mode}_{key}_{list(val).index(cls)}", value=float(cls), step=step)

        else:
            for key, val in metrics.items():
                if type(val) is list:
                    warnings.warn(f"Provided class metric value ({val}) is a list. Will not be able to process.")  # Sometimes happens. Don't know why
                else:
                    log_metric(key=f"{self.mode}_{key}", value=float(val), step=step)


def print_metric(mode, metric, values):
    print(f"\n{mode} {metric}:\n  Overall: {values[0]:.3f}\n  Per class: {values[1]}")


def write_config(from_file, to_folder):
    shutil.copy(str(from_file), f"{to_folder}/config.yaml")


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
