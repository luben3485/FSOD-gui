# FSOD

## Installation


### create soft link for dataset

```bash
../FSOD$ mkdir data
../FSOD/data$ ln -s <PATH_OF_DATASET> <NAME_OF_DATASET>
```
find detailed description in https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models


### install conda packages

befor the installation, make sure your cuda version is 10.0
```bash
conda env create -f env.yml
```


### install coco api

```bash
$ git clone https://github.com/pdollar/coco.git
$ cd ../coco/PythonAPI
$ make
$ make install
$ cd ../FSOD/lib
../FSOD/lib$ mv <PATH_TO_FRESH_GIT_CLONE>/coco/PythonAPI/pycocotools .
```


### install _C related packages

```bash
$ cd ../FSOD/lib
$ python setup.py build develop
```
