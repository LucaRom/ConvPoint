# ConvPoint: Generalizing discrete convolutions for unstructured point clouds

## Introduction

This repository was forked from the [original repo](https://github.com/aboulch/ConvPoint).  
Python scripts for point cloud classification and segmentation. The library is coded with PyTorch.  
It has been updated to fit the case of airborne lidar classification. 

## License

Code is released under dual license depending on applications, research or commercial. Reseach and non-commercial license is GPLv3.
See the [license](LICENSE.md).

## Citation

If you use this code in your research, please consider citing the original author:

```
@inproceedings {or.20191064,
booktitle = {Eurographics Workshop on 3D Object Retrieval},
title = {{Generalizing Discrete Convolutions for Unstructured Point Clouds}},
author = {Boulch, Alexandre},
year = {2019},
publisher = {The Eurographics Association},
ISSN = {1997-0471},
ISBN = {978-3-03868-077-2},
DOI = {10.2312/3dor.20191064}
}
```

## Platform

The code was tested on Ubuntu 18.04 with Anaconda.

## Dependencies

- Pytorch
- MLFlow
- Scikit-learn
- TQDM
- PlyFile
- H5py
- Cython

All these dependencies can be install via conda in an Anaconda environment or via pip.

### Installation
Procedure to install all the dependencies:
```bash
conda create --name <name_of_your_env> python=3.6
conda activate <name_of_your_env>
conda install pytorch -c pytorch
conda install scikit-learn tqdm h5py cython pyyaml
conda install -c conda-forge laspy
pip install plyfile
```

## The library

### Nearest neighbor module

The `knn` directory contains a very small wrapper for [NanoFLANN](https://github.com/jlblancoc/nanoflann) with OpenMP.
To compile the module:
```
cd convpoint/knn
python setup.py install --home="."
```

In the case, you do not want to use this C++/Python wrapper. You still can use the previous version of the nearest neighbors computation with Scikit Learn and Multiprocessing, python only version (slower). To do so, add the following lines at the start of your main script (e.g. ```modelnet_classif.py```):
```
from global_tags import GlobalTags
GlobalTags.legacy_layer_base(True)
```

## Usage
Step 1: Prepare las files.  
```
python airborne_lidar/prepare_airborne_lidar_label.py --folder /path/to/las/folder --dest path/to/input/folder --csv file.csv
```
Step 2: Prepare a config file and train a model.  
A config file template can be found [here](airborne_lidar/config_template.yaml).
```
python airborne_lidar/airborne_lidar_seg.py --config /path/to/config.yaml
```
Step 3: Inference on new las files.
```
python airborne_lidar/airborne_lidar_inference.py --modeldir /path/to/model/folder --rootdir path/to/input/las --test_step int
```  
