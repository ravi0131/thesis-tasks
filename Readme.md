# About
This repo is a collection of notebooks and code for the preparatory tasks for my thesis

To run either the code or notebooks, you'll need to download the semanticKITTI dataset from http://www.semantic-kitti.org/dataset. (velodyne point clouds, calibration and label data)

Extract all the data to a directory called `dataset` and ensure that you have the following directory structure

```
├── task2.ipynb
├── task2.py
...
└── dataset
    └── sequences
        └── 00
            ├── labels
            ├── velodyne
            ├── calib.txt
            ├── poses.txt
            └── times.txt
        └── 01
        ...
```
