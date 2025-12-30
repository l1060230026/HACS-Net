"""
Multi-sensor virtual scan simulator
Generates point cloud data simulating LiDAR scans, supports intelligent multi-sensor layout and occlusion simulation
"""

from re import T
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import os
import json
from typing import Tuple, Optional, Union, Dict, Any
import multiprocessing as mp
from functools import partial
import time


class ScanSimConfig:
    """Scan simulation configuration class"""
    def __init__(
        self,
        h_fov_min_deg: float = -180.0,
        h_fov_max_deg: float = 180.0,
        v_fov_min_deg: float = -25.0,
        v_fov_max_deg: float = 2.0,
        h_res_deg: float = 0.35,
        v_res_deg: float = 0.4,
        max_range: float = 30.0,
        range_noise_std: float = 0.01,
        dropout_prob: float = 0.05,
        sensor_height: float = 1.2,
        sensor_offset_xy: tuple = (0.0, -0.5),
    ) -> None:
        self.h_fov_min_deg = h_fov_min_deg
        self.h_fov_max_deg = h_fov_max_deg
        self.v_fov_min_deg = v_fov_min_deg
        self.v_fov_max_deg = v_fov_max_deg
        self.h_res_deg = max(1e-6, h_res_deg)
        self.v_res_deg = max(1e-6, v_res_deg)
        self.max_range = max_range
        self.range_noise_std = range_noise_std
        self.dropout_prob = np.clip(dropout_prob, 0.0, 1.0)
        self.sensor_height = sensor_height
        self.sensor_offset_xy = sensor_offset_xy


class VirtualScanSimulator:
    """Main virtual scan simulator class"""
    
    def __init__(self, config: Optional[ScanSimConfig] = None):
        """
        Initialize virtual scan simulator
        
        Args:
            config: Scan configuration, uses default config if None
        """
        self.config = config if config is not None else ScanSimConfig()
        
    def _cartesian_to_spherical(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert Cartesian coordinates to spherical coordinates"""
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2]
        r = np.linalg.norm(points_xyz, axis=1) + 1e-9
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
        return r, az, el

    def _spherical_to_cartesian(self, r: np.ndarray, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        """Convert spherical coordinates to Cartesian coordinates"""
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return np.stack([x, y, z], axis=1)

    def simulate_scan_block(
        self,
        block_points_xyz_abs: np.ndarray,
        block_center_abs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate LiDAR scan for a point cloud block
        
        Args:
            block_points_xyz_abs: Point cloud block data in absolute coordinates
            block_center_abs: Block center position in absolute coordinates
            
        Returns:
            (Simulated scanned point cloud, indices of retained points)
        """
        if block_points_xyz_abs.size == 0:
            return block_points_xyz_abs, np.arange(block_points_xyz_abs.shape[0], dtype=np.int64)

        # Sensor position (relative to block center)
        sensor_xy = np.array([
            block_center_abs[0] + self.config.sensor_offset_xy[0],
            block_center_abs[1] + self.config.sensor_offset_xy[1],
        ])
        sensor_pos = np.array([sensor_xy[0], sensor_xy[1], block_center_abs[2] + self.config.sensor_height], dtype=np.float32)

        # Transform points to sensor coordinate system
        pts_sensor = block_points_xyz_abs.astype(np.float32) - sensor_pos[None, :]

        # Convert to spherical coordinates
        r, az_deg, el_deg = self._cartesian_to_spherical(pts_sensor)

        # FOV and distance filtering
        h_ok = (az_deg >= self.config.h_fov_min_deg) & (az_deg <= self.config.h_fov_max_deg)
        v_ok = (el_deg >= self.config.v_fov_min_deg) & (el_deg <= self.config.v_fov_max_deg)
        d_ok = r <= self.config.max_range
        mask = h_ok & v_ok & d_ok
        
        if not np.any(mask):
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

        r = r[mask]
        az_deg = az_deg[mask]
        el_deg = el_deg[mask]
        pts_sensor = pts_sensor[mask]
        idxs = np.nonzero(mask)[0]

        # Angle binning (keep nearest point in each angle bin)
        h_bins = int(np.ceil((self.config.h_fov_max_deg - self.config.h_fov_min_deg) / self.config.h_res_deg))
        v_bins = int(np.ceil((self.config.v_fov_max_deg - self.config.v_fov_min_deg) / self.config.v_res_deg))
        h_bin = np.floor((az_deg - self.config.h_fov_min_deg) / self.config.h_res_deg).astype(np.int32)
        v_bin = np.floor((el_deg - self.config.v_fov_min_deg) / self.config.v_res_deg).astype(np.int32)
        h_bin = np.clip(h_bin, 0, max(0, h_bins - 1))
        v_bin = np.clip(v_bin, 0, max(0, v_bins - 1))
        lin_bin = v_bin * max(1, h_bins) + h_bin

        order = np.argsort(r)
        keep_mask_local = np.zeros(order.shape[0], dtype=bool)
        seen = set()
        for pos in order:
            lb = lin_bin[pos]
            if lb in seen:
                continue
            seen.add(lb)
            keep_mask_local[pos] = True

        r = r[keep_mask_local]
        az_deg = az_deg[keep_mask_local]
        el_deg = el_deg[keep_mask_local]
        kept_block_indices = idxs[keep_mask_local]

        # Add range noise
        if self.config.range_noise_std > 0:
            r = r + np.random.normal(0.0, self.config.range_noise_std, size=r.shape[0]).astype(np.float32)
            r = np.clip(r, 0.0, self.config.max_range)

        # Random dropout
        if self.config.dropout_prob > 0:
            keep = np.random.rand(r.shape[0]) > self.config.dropout_prob
            if np.any(keep):
                r = r[keep]
                az_deg = az_deg[keep]
                el_deg = el_deg[keep]
                kept_block_indices = kept_block_indices[keep]
            else:
                return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

        # Convert back to Cartesian coordinates in sensor frame, then to absolute coordinates
        pts_sensor_sim = self._spherical_to_cartesian(r, az_deg, el_deg).astype(np.float32)
        pts_abs_sim = pts_sensor_sim + sensor_pos[None, :]
        return pts_abs_sim, kept_block_indices

    def simulate_scan_with_occlusion(
        self, 
        point_cloud: Union[np.ndarray, any], 
        sensor_position: np.ndarray, 
        voxel_size: float = 0.05, 
        max_range: float = 30.0, 
        fov_horizontal: float = 360.0, 
        fov_vertical: float = 30.0,
        seed: Optional[int] = None,
        add_noise: bool = True,
        add_dropout: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precise occlusion simulation scan algorithm based on ray casting and 3D voxel grid
        
        Args:
            point_cloud: Point cloud data with shape (N, 3) or (N, 6)
            sensor_position: Sensor position [x, y, z]
            voxel_size: Voxel size
            max_range: Maximum scan range
            fov_horizontal: Horizontal field of view (degrees)
            fov_vertical: Vertical field of view (degrees)
            seed: Random seed
            add_noise: Whether to add range noise
            add_dropout: Whether to add random dropout
            
        Returns:
            (Simulated scanned point cloud data, original indices of retained points)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Input processing
        if hasattr(point_cloud, 'cpu') and hasattr(point_cloud, 'numpy'):
            original_pcd_np = point_cloud.cpu().numpy()
        else:
            original_pcd_np = np.asarray(point_cloud)

        if original_pcd_np.size == 0:
            return np.empty((0, original_pcd_np.shape[1]), dtype=original_pcd_np.dtype), np.array([], dtype=np.int64)

        points_xyz = original_pcd_np[:, :3]
        sensor_position = np.asarray(sensor_position)

        print("1. Performing distance and FOV pre-filtering...")
        
        # Calculate relative positions and distances from points to sensor
        points_relative = points_xyz - sensor_position
        distances = np.linalg.norm(points_relative, axis=1)

        # Distance filtering
        range_mask = distances <= max_range
        
        # FOV filtering
        distances_safe = np.where(distances == 0, 1e-6, distances)
        
        # Calculate vertical angle (elevation)
        vertical_angles = np.arcsin(points_relative[:, 2] / distances_safe) * 180.0 / np.pi
        v_fov_mask = np.abs(vertical_angles) <= fov_vertical / 2.0
        
        # Calculate horizontal angle (azimuth)
        if fov_horizontal < 360.0:
            horizontal_angles = np.arctan2(points_relative[:, 1], points_relative[:, 0]) * 180.0 / np.pi
            h_fov_mask = np.abs(horizontal_angles) <= fov_horizontal / 2.0
        else:
            h_fov_mask = np.ones_like(range_mask, dtype=bool)

        # Combine all filtering conditions
        pre_filter_mask = range_mask & v_fov_mask & h_fov_mask
        
        pre_filtered_indices = np.where(pre_filter_mask)[0]
        if pre_filtered_indices.size == 0:
            return np.empty((0, original_pcd_np.shape[1]), dtype=original_pcd_np.dtype), np.array([], dtype=np.int64)
            
        pre_filtered_points = points_xyz[pre_filtered_indices]
        print(f"   - {len(pre_filtered_points)} points remaining after pre-filtering.")
        
        # Create voxel grid for pre-filtered points
        print("2. Creating voxel grid for filtered points...")
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pre_filtered_points)
        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size=voxel_size)
        occupied_voxels = voxel_grid.get_voxels()
        occupied_indices_set = {tuple(v.grid_index) for v in occupied_voxels}
        print(f"   - Created {len(occupied_indices_set)} voxels.")
        
        # Ray casting to detect occlusion
        print("3. Performing ray casting to detect occlusion...")
        visible_voxel_indices = set()

        for voxel in tqdm(occupied_voxels, desc="  - Checking voxel visibility"):
            voxel_index = tuple(voxel.grid_index)
            voxel_center = voxel_grid.origin + (np.array(voxel_index) + 0.5) * voxel_grid.voxel_size
            
            direction = voxel_center - sensor_position
            ray_length = np.linalg.norm(direction)
            if ray_length < 1e-6: 
                continue
            direction_normalized = direction / ray_length

            is_occluded = False
            step_size = voxel_size * 0.5
            num_steps = int(ray_length / step_size)

            for i in range(1, num_steps):
                current_pos = sensor_position + direction_normalized * (i * step_size)
                current_voxel_idx = voxel_grid.get_voxel(current_pos)
                
                if tuple(current_voxel_idx) in occupied_indices_set:
                    is_occluded = True
                    break

            if not is_occluded:
                visible_voxel_indices.add(voxel_index)

        print(f"   - Found {len(visible_voxel_indices)} visible voxels.")
        
        # Collect all points in visible voxels
        print("4. Collecting points in visible voxels...")
        final_indices = []
        
        for i, point in enumerate(pre_filtered_points):
            point_voxel_idx = voxel_grid.get_voxel(point)
            if tuple(point_voxel_idx) in visible_voxel_indices:
                final_indices.append(pre_filtered_indices[i])

        final_indices = np.array(final_indices, dtype=np.int64)
        
        if final_indices.size == 0:
            return np.empty((0, original_pcd_np.shape[1]), dtype=original_pcd_np.dtype), np.array([], dtype=np.int64)

        simulated_points = original_pcd_np[final_indices]
        print(f"   - {len(simulated_points)} points retained after occlusion detection.")
        
        # Add perturbation simulation
        if len(simulated_points) > 0:
            # 1. Add range noise
            if add_noise and hasattr(self.config, 'range_noise_std') and self.config.range_noise_std > 0:
                print("5. Adding range noise...")
                points_xyz = simulated_points[:, :3]
                distances = np.linalg.norm(points_xyz - sensor_position, axis=1)
                
                # Add Gaussian noise
                noise = np.random.normal(0.0, self.config.range_noise_std, size=distances.shape[0])
                distances_noisy = distances + noise
                distances_noisy = np.clip(distances_noisy, 0.0, max_range)
                
                # Recalculate coordinates (keep direction unchanged, only change distance)
                directions = (points_xyz - sensor_position) / (distances[:, np.newaxis] + 1e-9)
                points_xyz_noisy = sensor_position + directions * distances_noisy[:, np.newaxis]
                
                # Update point cloud coordinates
                simulated_points = simulated_points.copy()
                simulated_points[:, :3] = points_xyz_noisy
                print(f"   - Range noise added (std={self.config.range_noise_std})")
            
            # 2. Random dropout
            if add_dropout and hasattr(self.config, 'dropout_prob') and self.config.dropout_prob > 0:
                print("6. Adding random dropout...")
                keep_mask = np.random.rand(len(simulated_points)) > self.config.dropout_prob
                if np.any(keep_mask):
                    simulated_points = simulated_points[keep_mask]
                    final_indices = final_indices[keep_mask]
                    print(f"   - Random dropout completed (dropout_prob={self.config.dropout_prob}), {len(simulated_points)} points remaining")
                else:
                    print("   - Warning: All points were randomly dropped")
                    return np.empty((0, original_pcd_np.shape[1]), dtype=original_pcd_np.dtype), np.array([], dtype=np.int64)

        print(f"   - {len(simulated_points)} points retained in final result.")

        return simulated_points, final_indices


    def save_config(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            'h_fov_min_deg': self.config.h_fov_min_deg,
            'h_fov_max_deg': self.config.h_fov_max_deg,
            'v_fov_min_deg': self.config.v_fov_min_deg,
            'v_fov_max_deg': self.config.v_fov_max_deg,
            'h_res_deg': self.config.h_res_deg,
            'v_res_deg': self.config.v_res_deg,
            'max_range': self.config.max_range,
            'range_noise_std': self.config.range_noise_std,
            'dropout_prob': self.config.dropout_prob,
            'sensor_height': self.config.sensor_height,
            'sensor_offset_xy': self.config.sensor_offset_xy,
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to: {filepath}")

    def load_config(self, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.config = ScanSimConfig(**config_dict)
        print(f"Configuration loaded from {filepath}")


def find_point_cloud_files(input_dir: str) -> list:
    """
    Find all supported point cloud files in the input directory
    
    Args:
        input_dir: Input directory path
        
    Returns:
        List of point cloud file paths
    """
    supported_extensions = ['.ply', '.pcd', '.txt', '.npy']
    point_cloud_files = []
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if os.path.isfile(input_dir):
        # If input is a single file, return directly
        if any(input_dir.lower().endswith(ext) for ext in supported_extensions):
            return [input_dir]
        else:
            raise ValueError(f"Unsupported file format: {input_dir}")
    
    # Traverse directory to find point cloud files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                point_cloud_files.append(os.path.join(root, file))
    
    return sorted(point_cloud_files)


def load_point_cloud(filepath: str) -> np.ndarray:
    """
    Load point cloud file
    
    Args:
        filepath: Point cloud file path
        
    Returns:
        Point cloud data array
    """
    print(f"Loading point cloud file: {filepath}")
    
    if filepath.endswith('.npy'):
        points = np.load(filepath)
    elif filepath.endswith('.txt'):
        points = np.loadtxt(filepath)
    elif filepath.endswith('.ply') or filepath.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    print(f"Loaded {len(points)} points")
    return points


def analyze_point_cloud_scene(points: np.ndarray, verbose: bool = False) -> dict:
    """
    Analyze statistical information of point cloud scene
    
    Args:
        points: Point cloud data with shape (N, 3)
        
    Returns:
        Dictionary containing scene statistical information
    """
    if len(points) == 0:
        return {}
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center_coords = (min_coords + max_coords) / 2.0
    dimensions = max_coords - min_coords
    
    scene_info = {
        'num_points': len(points),
        'bounds': {
            'min': min_coords,
            'max': max_coords,
            'center': center_coords,
            'dimensions': dimensions
        },
        'height_stats': {
            'min_height': min_coords[2],
            'max_height': max_coords[2],
            'avg_height': np.mean(points[:, 2]),
            'height_range': dimensions[2]
        }
    }
    
    if verbose:
        print(f"Point cloud scene analysis:")
        print(f"  - Number of points: {scene_info['num_points']}")
        print(f"  - Scene dimensions: [{dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}]")
        print(f"  - Scene center: [{center_coords[0]:.2f}, {center_coords[1]:.2f}, {center_coords[2]:.2f}]")
        print(f"  - Height range: {min_coords[2]:.2f} ~ {max_coords[2]:.2f} (average: {np.mean(points[:, 2]):.2f})")
    
    return scene_info


def save_point_cloud(points: np.ndarray, filepath: str):
    """
    Save point cloud file
    PLY format is recommended for smaller file size and better compatibility
    
    Args:
        points: Point cloud data with shape (N, 3) or (N, 6)
        filepath: Save path, supports .ply, .pcd, .txt, .npy formats
    """
    print(f"Saving results to: {filepath}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.npy'):
        np.save(filepath, points)
    elif filepath.endswith('.txt'):
        # For txt files, maintain original data precision without formatting
        if points.shape[1] >= 6:  # Contains x,y,z,r,g,b
            # Save with header format, maintaining original precision
            header = "x y z r g b"
            points_to_save = points[:, :6]  # Only save first 6 columns: x,y,z,r,g,b
            np.savetxt(filepath, points_to_save, header=header, comments='')
        else:
            # Only xyz coordinates, maintain original precision
            np.savetxt(filepath, points)
    elif filepath.endswith('.ply'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] >= 6:  # Contains color information
            # Ensure color values are in [0,1] range
            colors = points[:, 3:6].astype(np.float32)
            if colors.max() > 1.0:  # If color values > 1, they are in 0-255 range
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save PLY file directly, Open3D will automatically choose binary format
        o3d.io.write_point_cloud(filepath, pcd)
    elif filepath.endswith('.pcd'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] >= 6:  # Contains color information
            # Ensure color values are in [0,1] range
            colors = points[:, 3:6].astype(np.float32)
            if colors.max() > 1.0:  # If color values > 1, they are in 0-255 range
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save PCD file directly
        o3d.io.write_point_cloud(filepath, pcd)
    else:
        raise ValueError(f"Unsupported output file format: {filepath}")


def check_processed_files(input_files: list, output_dir: str, output_format: str) -> Tuple[list, list]:
    """
    Check which files have been processed, return lists of files to process and already processed files
    
    Args:
        input_files: Input file list
        output_dir: Output directory
        output_format: Output file format
        
    Returns:
        (List of files to process, List of already processed files)
    """
    if not os.path.exists(output_dir):
        return input_files, []
    
    # Get existing files in output directory
    existing_files = set()
    for file in os.listdir(output_dir):
        if file.endswith(f'_scanned.{output_format}') or file.endswith('.npy'):
            # Extract original filename (remove _scanned suffix)
            if file.endswith(f'_scanned.{output_format}'):
                base_name = file.replace(f'_scanned.{output_format}', '')
            else:  # .npy file
                base_name = file.replace('.npy', '')
            existing_files.add(base_name)
    
    # Separate files to process and already processed files
    files_to_process = []
    already_processed = []
    
    for input_file in input_files:
        input_filename = os.path.basename(input_file)
        input_name, _ = os.path.splitext(input_filename)
        
        if input_name in existing_files:
            already_processed.append(input_file)
        else:
            files_to_process.append(input_file)
    
    return files_to_process, already_processed


def create_default_configs():
    """Create some commonly used preset configurations"""
    configs = {}
    
    # Standard LiDAR configuration
    configs['standard_lidar'] = ScanSimConfig(
        h_fov_min_deg=-180.0,
        h_fov_max_deg=180.0,
        v_fov_min_deg=-25.0,
        v_fov_max_deg=2.0,
        h_res_deg=0.35,
        v_res_deg=0.4,
        max_range=30.0,
        range_noise_std=0.01,
        dropout_prob=0.05,
        sensor_height=1.5,
        sensor_offset_xy=(0.0, -0.5)
    )
    
    # High resolution configuration
    configs['high_resolution'] = ScanSimConfig(
        h_fov_min_deg=-180.0,
        h_fov_max_deg=180.0,
        v_fov_min_deg=-25.0,
        v_fov_max_deg=2.0,
        h_res_deg=0.1,
        v_res_deg=0.1,
        max_range=50.0,
        range_noise_std=0.005,
        dropout_prob=0.02,
        sensor_height=1.2,
        sensor_offset_xy=(0.0, -0.5)
    )
    
    # Indoor environment configuration
    configs['indoor'] = ScanSimConfig(
        h_fov_min_deg=-180.0,
        h_fov_max_deg=180.0,
        v_fov_min_deg=-30.0,
        v_fov_max_deg=10.0,
        h_res_deg=0.5,
        v_res_deg=0.5,
        max_range=20.0,
        range_noise_std=0.02,
        dropout_prob=0.1,
        sensor_height=1.5,
        sensor_offset_xy=(0.0, 0.0)
    )
    
    return configs


def _process_file_worker(args_tuple):
    """Worker function for processing a single file in multi-sensor mode (for file-level multiprocessing)."""
    (input_file, output_dir, config_dict, layout_params,
     scan_params, multi_sensor_params, io_params) = args_tuple

    start_time = time.time()
    result = {
        'success': False,
        'input_file': input_file,
        'output_files': [],
        'processing_time': 0.0,
        'error': None,
        'points_before': 0,
        'points_after': 0
    }

    try:
        # Construct configuration and simulator
        config = ScanSimConfig(**config_dict)
        from multi_sensor import MultiSensorScanSimulator  # Lazy import
        multi_simulator = MultiSensorScanSimulator(config)
        # Apply layout parameters
        multi_simulator.layout_planner.grid_resolution = layout_params['grid_resolution']
        multi_simulator.layout_planner.min_sensor_distance = layout_params['min_sensor_distance']
        multi_simulator.layout_planner.max_sensor_distance = layout_params['max_sensor_distance']
        multi_simulator.layout_planner.max_sensors = layout_params['max_sensors']
        multi_simulator.layout_planner.min_coverage_threshold = layout_params['min_coverage_threshold']

        # Load point cloud
        points = load_point_cloud(input_file)
        result['points_before'] = len(points)
        # Extract label column (if exists) for more accurate layout analysis
        labels = points[:, 6].astype(int) if (isinstance(points, np.ndarray) and points.shape[1] > 6) else None

        # Execute planning and scanning (linear execution internally, no longer multiprocessing)
        merged_cloud, sensor_positions, metrics = multi_simulator.plan_and_simulate_multi_sensor(
            points,
            labels=labels,
            max_sensors=layout_params['max_sensors'],
            min_coverage_threshold=layout_params['min_coverage_threshold'],
            use_multiprocessing=False,
            num_processes=1,
            voxel_size=scan_params['voxel_size'],
            max_range=scan_params['max_range'],
            fov_horizontal=scan_params['fov_horizontal'],
            fov_vertical=scan_params['fov_vertical'],
            seed=scan_params['seed'],
            add_noise=not scan_params['disable_noise'],
            add_dropout=not scan_params['disable_dropout'],
            # New: Multi-sensor scan improvement parameters
            use_iterative_optimization=multi_sensor_params['use_iterative_optimization'],
            target_sensor_count=multi_sensor_params['target_sensor_count'],
            max_iterations=multi_sensor_params['max_iterations'],
            coverage_improvement_threshold=multi_sensor_params['coverage_improvement_threshold']
        )

        result['points_after'] = len(merged_cloud)

        # Save output
        input_filename = os.path.basename(input_file)
        input_name, _ = os.path.splitext(input_filename)
        if len(merged_cloud) > 0:
            scanned_filename = f"{input_name}_scanned.{io_params['output_format']}"
            scanned_file = os.path.join(output_dir, scanned_filename)
            save_point_cloud(merged_cloud, scanned_file)
            result['output_files'].append(scanned_file)

            npy_filename = f"{input_name}.npy"
            npy_file = os.path.join(output_dir, npy_filename)
            np.save(npy_file, merged_cloud)
            result['output_files'].append(npy_file)

        result['success'] = True
    except Exception as e:
        result['error'] = str(e)
        print(f"Error processing file {os.path.basename(input_file)}: {e}")
    finally:
        result['processing_time'] = time.time() - start_time

    return result


def main():
    """Main function - Multi-sensor scan command line interface"""
    parser = argparse.ArgumentParser(description='Multi-sensor virtual scan simulator')
    parser.add_argument('--input', type=str, default='data/bim', help='Input point cloud folder path')
    parser.add_argument('--output', type=str, default='multi_scan_sim', help='Output simulated scan results folder path')
    parser.add_argument('--config', type=str, help='Configuration file path (JSON format)')
    parser.add_argument('--preset', type=str, choices=['standard_lidar', 'high_resolution', 'indoor'], 
                       default='standard_lidar', help='Use preset configuration')
    parser.add_argument('--output_format', type=str, choices=['txt', 'ply', 'pcd', 'npy'], 
                       default='ply', help='Output file format (default PLY format, smaller size)')
    parser.add_argument('--num_processes', type=int, default=16,
                       help='Number of parallel processes (defaults to CPU core count)')
    parser.add_argument('--disable_multiprocess', action='store_true', default=True,
                       help='Disable multiprocessing, use single process')
    
    # Multi-sensor scan options
    parser.add_argument('--max_sensors', type=int, default=6,
                       help='Maximum number of sensors')
    parser.add_argument('--min_coverage_threshold', type=float, default=0.85,
                       help='Minimum coverage threshold')
    parser.add_argument('--grid_resolution', type=float, default=0.2,
                       help='Grid resolution')
    parser.add_argument('--min_sensor_distance', type=float, default=2.0,
                       help='Minimum sensor distance')
    parser.add_argument('--max_sensor_distance', type=float, default=8.0,
                       help='Maximum sensor scan distance')
    
    # New: Multi-sensor scan improvement options
    parser.add_argument('--use_iterative_optimization', action='store_true', default=True,
                       help='Enable iterative optimization strategy, dynamically adjust sensor positions based on actual scan results')
    parser.add_argument('--target_sensor_count', type=int, default=None,
                       help='Manually specify sensor count, if specified will force use this count')
    parser.add_argument('--max_iterations', type=int, default=5,
                       help='Maximum number of iterations for iterative optimization')
    parser.add_argument('--coverage_improvement_threshold', type=float, default=0.05,
                       help='Coverage improvement threshold, stop iteration if below this value')
    
    # Scan simulation parameters
    parser.add_argument('--voxel_size', type=float, default=0.05, 
                       help='Voxel size')
    parser.add_argument('--max_range', type=float, default=10.0, 
                       help='Maximum scan range')
    parser.add_argument('--fov_horizontal', type=float, default=360.0, 
                       help='Horizontal field of view (degrees)')
    parser.add_argument('--fov_vertical', type=float, default=300.0, 
                       help='Vertical field of view (degrees)')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--disable_noise', action='store_true',
                       help='Disable range noise simulation')
    parser.add_argument('--disable_dropout', action='store_true', default=True,
                       help='Disable random dropout simulation')
    parser.add_argument('--sensor_height', type=float, default=1.5,
                       help='Sensor height (meters). If provided, will override height in config and layout planner')
    
    # Resume functionality options
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Enable resume functionality, automatically skip already processed files. By default will reprocess all files')
    
    args = parser.parse_args()
    
    # Find all point cloud files in input folder
    print(f"Searching for point cloud files in input folder: {args.input}")
    print(f"Output format: {args.output_format} (PLY format recommended, smaller size)")
    input_files = find_point_cloud_files(args.input)
    
    if not input_files:
        print("No point cloud files found!")
        return
    
    print(f"Found {len(input_files)} point cloud files:")
    for i, file_path in enumerate(input_files):
        print(f"  {i+1}. {file_path}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Resume functionality processing
    if args.resume:
        print("\n=== Resume Mode ===")
        files_to_process, already_processed = check_processed_files(input_files, args.output, args.output_format)
        if already_processed:
            print(f"Found {len(already_processed)} already processed files, will skip:")
            for i, file_path in enumerate(already_processed):
                print(f"  {i+1}. {os.path.basename(file_path)}")
        print(f"{len(files_to_process)} files to process")
    else:
        print("\n=== Normal Processing Mode ===")
        files_to_process = input_files
        already_processed = []
        print("Tip: Use --resume parameter to enable resume functionality, automatically skip already processed files")
    
    if not files_to_process:
        print("All files have been processed!")
        return
    
    print(f"\n=== Multi-sensor Scan Mode ===")
    print(f"Maximum number of sensors: {args.max_sensors}")
    print(f"Minimum coverage threshold: {args.min_coverage_threshold}")
    print(f"Grid resolution: {args.grid_resolution}m")
    print(f"Minimum sensor distance: {args.min_sensor_distance}m")
    
    # Display new multi-sensor scan improvement options
    if args.target_sensor_count is not None:
        print(f"Manually specified sensor count: {args.target_sensor_count}")
    if args.use_iterative_optimization:
        print(f"Iterative optimization: Enabled (max iterations: {args.max_iterations}, improvement threshold: {args.coverage_improvement_threshold})")
    else:
        print(f"Iterative optimization: Disabled (using traditional planning method)")
    
    # Create multi-sensor scan simulator
    if args.config:
        config = ScanSimConfig()
        simulator = VirtualScanSimulator()
        simulator.load_config(args.config)
        multi_simulator = MultiSensorScanSimulator(simulator.config)
    else:
        configs = create_default_configs()
        multi_simulator = MultiSensorScanSimulator(configs[args.preset])

    # Override sensor height (applies to both config and layout planner)
    if args.sensor_height is not None:
        try:
            multi_simulator.config.sensor_height = float(args.sensor_height)
        except Exception:
            pass
        try:
            multi_simulator.layout_planner.sensor_height = float(args.sensor_height)
        except Exception:
            pass

    print(f"Applied sensor height: config={multi_simulator.config.sensor_height}, layout_planner={multi_simulator.layout_planner.sensor_height}")
    
    # Configure layout planner parameters
    multi_simulator.layout_planner.grid_resolution = args.grid_resolution
    multi_simulator.layout_planner.min_sensor_distance = args.min_sensor_distance
    multi_simulator.layout_planner.max_sensor_distance = args.max_sensor_distance
    multi_simulator.layout_planner.max_sensors = args.max_sensors
    multi_simulator.layout_planner.min_coverage_threshold = args.min_coverage_threshold
    
    # Prepare configuration dictionary (for subprocess to construct config)
    if args.config:
        tmp_sim = VirtualScanSimulator()
        tmp_sim.load_config(args.config)
        config_dict = {
            'h_fov_min_deg': tmp_sim.config.h_fov_min_deg,
            'h_fov_max_deg': tmp_sim.config.h_fov_max_deg,
            'v_fov_min_deg': tmp_sim.config.v_fov_min_deg,
            'v_fov_max_deg': tmp_sim.config.v_fov_max_deg,
            'h_res_deg': tmp_sim.config.h_res_deg,
            'v_res_deg': tmp_sim.config.v_res_deg,
            'max_range': tmp_sim.config.max_range,
            'range_noise_std': tmp_sim.config.range_noise_std,
            'dropout_prob': tmp_sim.config.dropout_prob,
            'sensor_height': (float(args.sensor_height) if args.sensor_height is not None else tmp_sim.config.sensor_height),
            'sensor_offset_xy': tmp_sim.config.sensor_offset_xy,
        }
    else:
        preset_configs = create_default_configs()
        preset_config = VirtualScanSimulator(preset_configs[args.preset]).config
        config_dict = {
            'h_fov_min_deg': preset_config.h_fov_min_deg,
            'h_fov_max_deg': preset_config.h_fov_max_deg,
            'v_fov_min_deg': preset_config.v_fov_min_deg,
            'v_fov_max_deg': preset_config.v_fov_max_deg,
            'h_res_deg': preset_config.h_res_deg,
            'v_res_deg': preset_config.v_res_deg,
            'max_range': preset_config.max_range,
            'range_noise_std': preset_config.range_noise_std,
            'dropout_prob': preset_config.dropout_prob,
            'sensor_height': (float(args.sensor_height) if args.sensor_height is not None else preset_config.sensor_height),
            'sensor_offset_xy': preset_config.sensor_offset_xy,
        }

    # Layout and scan parameters
    layout_params = {
        'grid_resolution': args.grid_resolution,
        'min_sensor_distance': args.min_sensor_distance,
        'max_sensor_distance': args.max_sensor_distance,
        'max_sensors': args.max_sensors,
        'min_coverage_threshold': args.min_coverage_threshold,
    }
    scan_params = {
        'voxel_size': args.voxel_size,
        'max_range': args.max_range,
        'fov_horizontal': args.fov_horizontal,
        'fov_vertical': args.fov_vertical,
        'seed': args.seed,
        'disable_noise': args.disable_noise,
        'disable_dropout': args.disable_dropout,
    }
    # New: Multi-sensor scan improvement parameters
    multi_sensor_params = {
        'use_iterative_optimization': args.use_iterative_optimization,
        'target_sensor_count': args.target_sensor_count,
        'max_iterations': args.max_iterations,
        'coverage_improvement_threshold': args.coverage_improvement_threshold,
    }
    io_params = {
        'output_format': args.output_format,
    }

    # Organize parameters (only process files that need processing)
    worker_args = [
        (input_file, args.output, config_dict, layout_params, scan_params, multi_sensor_params, io_params)
        for input_file in files_to_process
    ]

    # File-level processing (can use multiprocessing)
    if args.disable_multiprocess:
        results = []
        for i, wargs in enumerate(worker_args):
            print(f"\nProcessing file {i+1}/{len(worker_args)}: {os.path.basename(wargs[0])}")
            results.append(_process_file_worker(wargs))
    else:
        num_proc = args.num_processes if args.num_processes else mp.cpu_count()
        print(f"\nUsing file-level multiprocessing, number of processes: {num_proc}")
        with mp.Pool(processes=num_proc) as pool:
            results = list(tqdm(
                pool.imap(_process_file_worker, worker_args),
                total=len(worker_args),
                desc="Processing point cloud files",
                unit="file"
            ))

    # Summary
    success_count = sum(1 for r in results if r.get('success'))
    fail_count = len(results) - success_count
    skipped_count = len(already_processed)
    
    print(f"\nMulti-sensor scan processing completed! Results saved to: {args.output}")
    print(f"Success: {success_count}, Failed: {fail_count}")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} (already processed)")
    
    if args.resume and skipped_count > 0:
        print(f"\nResume functionality: Automatically skipped {skipped_count} already processed files")
        print("To reprocess all files, do not use --resume parameter")


# Import multi-sensor functionality

from multi_sensor import MultiSensorScanSimulator, MultiSensorLayoutPlanner
from multi_sensor.scan_simulator import _simulate_single_sensor_worker



if __name__ == '__main__':
    main()
