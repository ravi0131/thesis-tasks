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

## Task 2
Implement a visualization of LiDAR point cloud sequence from the SemanticKITTI dataset. 
Your implementation should
1) Allow you to navigate forward and backward in time using the left and right arrow keys
2) Visualize the semantic labels for each point

SemanticKITTIVisualizer is a class that loads loads point clouds and visualizes them using open3D

It consists of the following functions
* load_color_map() => loads color map from color_map.json

* read_bin() =>  reads the binary file containing point cloud data

* read_label() => reads the binary file containing semantic labels

* update_point_cloud() => adds the next point cloud to the renderer to be visualized

* set_view_status() => sets the viewing angle of the camera 

* visualize() => responsible for visualizing. It sets up a open3D visualizer, registers callbacks for navigation and starts the visualization loop

Note: the above description is a short summary. For brevity, the parameter list for functions was omitted. Refer to documenation in code for further details


## Task 3
LiDAR scan always positions the scanning sensor at the origin (0,0,0). This is fine in stationary settings but when data is captured from a moving vehicle, then it sometimes necessary to have a global coordinate system for all scans. This is referred to as ego-motion compensation. <br>
Using the pose data in semanticKITTI dataset, transform coordinate system for each scan into a global one common for all scans of the sequence.<br>
 Create a visualization without ego-motion compensation and one with ego-motion compensation to fully understand the difference between the two


The implementation also uses the same class `SemanticKITTIVisualizer` but with some extra methods

* read_poses() => reads the pose file and returns a list of transformation matrices
* apply_transform() => applies the transformation on a set of points (point cloud)
* update_point_cloud_with_ego_motion() => adds the next point cloud to be visualized after performing coordinate transformation on it
* visualize_with_ego_motion() => visualizes point clouds with ego compensation

NOTE: Similar to task 2, the parameter lists for functions have been omitted for brevity. 
