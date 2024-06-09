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
    def __init__(self, base_dir: str, seq_idx: str, color_map_file: str, view_status_file: str, point_size: int) -> None:
        self.base_dir = base_dir
        self.seq_idx = seq_idx
        self.color_map = self.load_color_map(color_map_file)
        self.view_status_file = view_status_file
        self.point_size = point_size
        self.scan_dir = os.path.join(base_dir, seq_idx, "velodyne")
        self.label_dir = os.path.join(base_dir, seq_idx, "labels")
        self.pose_file = os.path.join(base_dir, seq_idx, "poses.txt")
        self.scan_files = sorted(os.listdir(self.scan_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        self.pcd = o3d.geometry.PointCloud()
        self.geometry_added = False
        logging.info("SemanticKITTIVisualizer initialized")

    def load_color_map(self, color_map_file: str) -> Dict[int, np.ndarray]:
        with open(color_map_file) as f:
            color_map = json.load(f)
        logging.info("Color map loaded")
        return {int(k): np.array(v, dtype=np.float32) / 255.0 for k, v in color_map.items()}

    def read_bin(self, file_path: str) -> np.ndarray:
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

    def read_label(self, file_path: str) -> np.ndarray:
        label = np.fromfile(file_path, dtype=np.uint32)
        sem_label = label & 0xFFFF  # Semantic label in lower half
        return sem_label

    def read_poses(self, file_path: str) -> List[np.ndarray]:
        poses = []
        with open(file_path) as f:
            for line in f:
                pose = np.fromstring(line, dtype=float, sep=' ').reshape(3, 4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                poses.append(pose)
        return poses

    def apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        hom_points = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = hom_points @ transform.T
        return transformed_points[:, :3]

    def update_point_cloud(self, vis: o3d.visualization.Visualizer, idx: int) -> None:
        scan_file = os.path.join(self.scan_dir, self.scan_files[idx])
        label_file = os.path.join(self.label_dir, self.label_files[idx])
        
        points = self.read_bin(scan_file)[:, :3]
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
        self.set_view_status(vis)
        logging.info(f"Frame {idx + 1}/{len(self.scan_files)} computed")

    def update_point_cloud_with_ego_motion(self, vis: o3d.visualization.Visualizer, idx: int) -> None:
        scan_file = os.path.join(self.scan_dir, self.scan_files[idx])
        label_file = os.path.join(self.label_dir, self.label_files[idx])
        
        points = self.read_bin(scan_file)[:, :3]
        labels = self.read_label(label_file)
        
        if points.shape[0] == 0:
            logging.warning(f"No points found in scan {scan_file}")
            return
        
        poses = self.read_poses(self.pose_file)
        transformed_points = self.apply_transform(points, poses[idx])
        
        colors = np.array([self.color_map.get(label, [0.5, 0.5, 0.5]) for label in labels])
        
        self.pcd.points = o3d.utility.Vector3dVector(transformed_points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if self.geometry_added:
            vis.remove_geometry(self.pcd, reset_bounding_box=False)
        
        vis.add_geometry(self.pcd, reset_bounding_box=not self.geometry_added)
        self.geometry_added = True
        
        vis.update_renderer()
        self.set_view_status(vis)
        logging.info(f"Frame {idx + 1}/{len(self.scan_files)} computed with ego-motion compensation")

    def set_view_status(self, vis: o3d.visualization.Visualizer) -> None:
        with open(self.view_status_file) as f:
            view_status = json.load(f)
        sio = StringIO()
        json.dump(view_status,sio)
        view_status_string = sio.getvalue()
        vis.set_view_status(view_status_string)
        logging.info("View status loaded")

    def visualize(self) -> None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=1920, height=1055)
        
        current_idx = [0]
        self.update_point_cloud(vis, current_idx[0])
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        def next(vis: o3d.visualization.Visualizer) -> None:
            if current_idx[0] < len(self.scan_files) - 1:
                current_idx[0] += 1
                self.update_point_cloud(vis, current_idx[0])
        
        def prev(vis: o3d.visualization.Visualizer) -> None:
            if current_idx[0] > 0:
                current_idx[0] -= 1
                self.update_point_cloud(vis, current_idx[0])
        
        def next20(vis: o3d.visualization.Visualizer):
            stepsize = 20
            if current_idx[0] < len(self.scan_files) -1 - stepsize:
                current_idx[0] += stepsize
                self.update_point_cloud(vis, current_idx[0])
       
        def prev20(vis: o3d.visualization.Visualizer):
            stepsize = 20
            if current_idx[0] < len(self.scan_files) -1 - stepsize:
                current_idx[0] -= stepsize
                self.update_point_cloud(vis, current_idx[0])
        
        vis.register_key_callback(262, next)
        vis.register_key_callback(263, prev)
        vis.register_key_callback(ord('N'),next20)
        vis.register_key_callback(ord('B'),prev20)
        
        vis.get_render_option().background_color = np.array([0, 0, 0])
        vis.get_render_option().point_size = self.point_size
        
        vis.poll_events()
        vis.update_renderer()
        self.set_view_status(vis)

        vis.run()
        vis.destroy_window()

    def visualize_with_ego_motion(self) -> None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=1920, height=1055)
        
        current_idx = [0]
        self.update_point_cloud_with_ego_motion(vis, current_idx[0])
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        def next(vis: o3d.visualization.Visualizer) -> None:
            stepsize = 1
            if current_idx[0] < len(self.scan_files) -1 - stepsize:
                current_idx[0] += stepsize
                self.update_point_cloud_with_ego_motion(vis, current_idx[0])
        
        def prev(vis: o3d.visualization.Visualizer) -> None:
            stepsize = 1
            if current_idx[0] >= stepsize :
                current_idx[0] -= stepsize
                self.update_point_cloud_with_ego_motion(vis, current_idx[0])
        

        def next20(vis: o3d.visualization.Visualizer):
            stepsize = 20
            if current_idx[0] < len(self.scan_files) -1 - stepsize:
                current_idx[0] += stepsize
                self.update_point_cloud_with_ego_motion(vis, current_idx[0])
       
        def prev20(vis: o3d.visualization.Visualizer):
            stepsize = 20
            if current_idx[0] < len(self.scan_files) -1 - stepsize:
                current_idx[0] -= stepsize
                self.update_point_cloud_with_ego_motion(vis, current_idx[0])
        
        vis.register_key_callback(262, next)
        vis.register_key_callback(263, prev)
        vis.register_key_callback(ord('N'),next20)
        vis.register_key_callback(ord('B'),prev20)
        
        vis.get_render_option().background_color = np.array([0, 0, 0])
        vis.get_render_option().point_size = self.point_size
        
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
    
    logging.info("Visualizing without ego-motion compensation")
    visualizer.visualize()

    logging.info("Visualizing with ego-motion compensation")
    visualizer.visualize_with_ego_motion()
