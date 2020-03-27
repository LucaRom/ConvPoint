import h5py
import os
import warnings
import yaml


def tsv_line(*args):
    return '\t'.join(map(str, args)) + '\n'


def write_features(file_name, xyzni, labels):
    """write the geometric features, labels and clouds in a h5 file"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('xyzni', data=xyzni, dtype='float16')
    data_file.create_dataset('labels', data=labels, dtype='uint8')
    data_file.close()


class InformationLogger(object):
    def __init__(self, log_folder, mode):
        # List of metrics names
        self.metrics = ['loss', 'iou', 'acc', 'fscore']
        self.metrics_classwise = ['iou', 'acc', 'fscore']
        self.mode = mode

        # Dicts of logs
        def open_log(metric_name, fmt_str="metric_{}_{}.log"):
            filename = fmt_str.format(mode, metric_name)
            return open(log_folder / filename, "a", buffering=1)

        self.metric_values = {m: open_log(m) for m in self.metrics}
        self.class_scores = {m: open_log(m, fmt_str="metric_classwise_{}_{}.log") for m in self.metrics_classwise}

    def add_metric_values(self, values, epoch):
        """Add new information to the averaged logs."""
        for key in values:
            if key in self.metric_values:
                self.metric_values[key].write(tsv_line(epoch, values[key]))
            else:
                warnings.warn(f"Unknown metric {key}")

    def add_class_scores(self, values, epoch):
        """Add new information to the classwise logs."""
        for key, value in values.items():
            if key in self.class_scores:
                counter = 0
                for num in value:
                    self.class_scores[key].write(tsv_line(epoch, counter, num))
                    counter += 1
            else:
                warnings.warn(f"Unknown metric {key}")


def print_metric(mode, metric, values):
    print(f"\n{mode} {metric}:\n  Overall: {values[0]:.3f}\n  Per class: {values[1]}")


def write_config(folder, args):
    with open(folder / 'config.yaml', 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
