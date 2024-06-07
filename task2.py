import open3d as o3d
import numpy as np
import os
import json
import logging
from typing import Dict, List
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticKITTIVisualizer:
    """
    A class to visualize SemanticKITTI dataset point clouds with semantic labels.

    Attributes:
    base_dir (str): The base directory of the dataset. (e.g. "./dataset/sequences/")
    seq_idx (str): The sequence index. (e.g. "01")
    color_map (Dict[int, np.ndarray]): A dictionary mapping labels to RGB colors.
    scan_dir (str): Directory containing scan files.   (e.g. "./dataset/sequences/01/velodyne")
    label_dir (str): Directory containing label files.  (e.g. "./dataset/sequences/01/labels")
    scan_files (List[str]): Sorted list of file names from scan_dir
    label_files (List[str]): Sorted list of file names from label_dir
    pcd (o3d.geometry.PointCloud): The point cloud object.
    geometry_added (bool): Flag indicating if geometry has been added to the visualizer.
    view_status_file(str): path to file containing view_status config of the visualizer
    point_size(int): size of each point
    """

    def __init__(self, base_dir: str, seq_idx: str, color_map_file: str, view_status_file: str, point_size: int) -> None:
        """
        Initialize the SemanticKITTIVisualizer.

        Args:
        base_dir (str): The base directory of the dataset.
        seq_idx (str): The sequence index.
        color_map_file (str): Path to the JSON file containing the color map.
        """
        self.base_dir = base_dir
        self.seq_idx = seq_idx
        self.color_map = self.load_color_map(color_map_file)
        self.view_status_file = view_status_file
        self.point_size = point_size
        self.scan_dir = os.path.join(base_dir, seq_idx, "velodyne")
        self.label_dir = os.path.join(base_dir, seq_idx, "labels")
        self.scan_files = sorted(os.listdir(self.scan_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        self.pcd = o3d.geometry.PointCloud()
        self.geometry_added = False
        logging.info("SemanticKITTIVisualizer initialized")

    def load_color_map(self, color_map_file: str) -> Dict[int, np.ndarray]:
        """
        Load the color map from a JSON file.

        Args:
        color_map_file (str): Path to the JSON file containing the color map.

        Returns:
        Dict[int, np.ndarray]: A dictionary mapping labels to RGB colors.
        """
        with open(color_map_file) as f:
            color_map = json.load(f)
        logging.info("Color map loaded")
        return {int(k): np.array(v, dtype=np.float32) / 255.0 for k, v in color_map.items()}

    def read_bin(self, file_path: str) -> np.ndarray:
        """
        Read a binary file containing point cloud data.

        Args:
        file_path (str): Path to the binary file.

        Returns:
        np.ndarray: An array of point cloud data.
        """
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

    def read_label(self, file_path: str) -> np.ndarray:
        """
        Read a binary file containing semantic labels.

        Args:
        file_path (str): Path to the label file.

        Returns:
        np.ndarray: An array of semantic labels.
        """
        label = np.fromfile(file_path, dtype=np.uint32)
        sem_label = label & 0xFFFF  # Semantic label in lower half
        return sem_label

    def update_point_cloud(self, vis: o3d.visualization.Visualizer, idx: int) -> None:
        """
        Update the point cloud with data from the given index.

        Args:
        vis (o3d.visualization.Visualizer): The Open3D visualizer.
        idx (int): The index of the scan and label files to be visualized.
        """
        
        scan_file = os.path.join(self.scan_dir, self.scan_files[idx])
        label_file = os.path.join(self.label_dir, self.label_files[idx])
        
        points = self.read_bin(scan_file)[:, :3]  # select all rows and first 3 columns
        labels = self.read_label(label_file)
        
        if points.shape[0] == 0:
            logging.warning(f"No points found in scan {scan_file}")
            return
        
        colors = np.array([self.color_map.get(label, [0.5, 0.5, 0.5]) for label in labels])
        
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if self.geometry_added:
            vis.remove_geometry(self.pcd, reset_bounding_box=False)
        
        vis.add_geometry(self.pcd, reset_bounding_box=not self.geometry_added)
        self.geometry_added = True
        
        vis.update_renderer()

        logging.info(f"Frame {idx + 1}/{len(self.scan_files)} computed")


    def set_view_status(self, vis: o3d.visualization.Visualizer) -> None:
        """
        Sets the camera view of the visualizer

        Args:
        vis (o3d.visualization.Visualizer): The Open3D visualizer.
        """
        with open(self.view_status_file) as f:
            view_status = json.load(f)
        sio = StringIO()
        json.dump(view_status,sio)
        view_status_string = sio.getvalue()
        vis.set_view_status(view_status_string)
        logging.info("View status loaded")
        

    def visualize(self) -> None:
        """
        Visualize the SemanticKITTI dataset using Open3D.

        This method sets up the Open3D visualizer, registers key callbacks for navigation, 
        and starts the visualization loop.
        """
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=1920, height=1055)  # Set window size
        
        current_idx = [0]
        self.update_point_cloud(vis, current_idx[0])
        
        def next(vis: o3d.visualization.Visualizer) -> None:
            if current_idx[0] < len(self.scan_files) - 1:
                current_idx[0] += 1
                self.update_point_cloud(vis, current_idx[0])
        
        def prev(vis: o3d.visualization.Visualizer) -> None:
            if current_idx[0] > 0:
                current_idx[0] -= 1
                self.update_point_cloud(vis, current_idx[0])
        
        vis.register_key_callback(262, next)  # Right arrow key
        vis.register_key_callback(263, prev)  # Left arrow key
        
        vis.get_render_option().background_color = np.array([0, 0, 0])  # Set background to black
        vis.get_render_option().point_size = self.point_size  
        
        def print_view_status(vis: o3d.visualization.Visualizer) -> None:
             logging.info(f"View status of visualizer: {vis.get_view_status()}")

        vis.register_key_callback(ord("V"), print_view_status)    # 'V' key to print view status of visualizer
        
        vis.poll_events()
        vis.update_renderer()
        self.set_view_status(vis)

        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    base_dir = "./dataset/sequences/"
    seq_idx = "01"
    color_map_file = "./color_map.json"
    view_status_file = "./view_status.json"
    point_size = 3
    
    visualizer = SemanticKITTIVisualizer(base_dir, seq_idx, color_map_file, view_status_file, point_size)
    visualizer.visualize()
