import h5py
import os

def get_airborne_lidar_info():
    obj_classes = {
        'other': 0,
        'building': 1,
        'water': 2,
        'ground': 3}

    return obj_classes


def write_features(file_name, xyzni, labels):
    """write the geometric features, labels and clouds in a h5 file"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('xyzni', data=xyzni, dtype='float16')
    data_file.create_dataset('labels', data=labels, dtype='uint8')
    data_file.close()
