"""
多传感器虚拟扫描模拟器
用于生成LiDAR扫描模拟的点云数据，支持多传感器智能布局和遮挡模拟
作者: Benny
日期: 2024
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
    """扫描模拟配置类"""
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
    """虚拟扫描模拟器主类"""
    
    def __init__(self, config: Optional[ScanSimConfig] = None):
        """
        初始化虚拟扫描模拟器
        
        Args:
            config: 扫描配置，如果为None则使用默认配置
        """
        self.config = config if config is not None else ScanSimConfig()
        
    def _cartesian_to_spherical(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将笛卡尔坐标转换为球坐标"""
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2]
        r = np.linalg.norm(points_xyz, axis=1) + 1e-9
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
        return r, az, el

    def _spherical_to_cartesian(self, r: np.ndarray, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        """将球坐标转换为笛卡尔坐标"""
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
        对点云块进行LiDAR扫描模拟
        
        Args:
            block_points_xyz_abs: 点云块数据，绝对坐标系
            block_center_abs: 块中心位置，绝对坐标系
            
        Returns:
            (模拟扫描后的点云, 保留点的索引)
        """
        if block_points_xyz_abs.size == 0:
            return block_points_xyz_abs, np.arange(block_points_xyz_abs.shape[0], dtype=np.int64)

        # 传感器位置（相对于块中心）
        sensor_xy = np.array([
            block_center_abs[0] + self.config.sensor_offset_xy[0],
            block_center_abs[1] + self.config.sensor_offset_xy[1],
        ])
        sensor_pos = np.array([sensor_xy[0], sensor_xy[1], block_center_abs[2] + self.config.sensor_height], dtype=np.float32)

        # 将点转换到传感器坐标系
        pts_sensor = block_points_xyz_abs.astype(np.float32) - sensor_pos[None, :]

        # 转换为球坐标
        r, az_deg, el_deg = self._cartesian_to_spherical(pts_sensor)

        # FOV和距离过滤
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

        # 角度分箱（每个角度箱保留最近的点）
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

        # 添加距离噪声
        if self.config.range_noise_std > 0:
            r = r + np.random.normal(0.0, self.config.range_noise_std, size=r.shape[0]).astype(np.float32)
            r = np.clip(r, 0.0, self.config.max_range)

        # 随机丢失
        if self.config.dropout_prob > 0:
            keep = np.random.rand(r.shape[0]) > self.config.dropout_prob
            if np.any(keep):
                r = r[keep]
                az_deg = az_deg[keep]
                el_deg = el_deg[keep]
                kept_block_indices = kept_block_indices[keep]
            else:
                return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

        # 转换回传感器坐标系笛卡尔坐标，然后转为绝对坐标
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
        基于光线投射和3D体素网格的精确遮挡模拟扫描算法
        
        Args:
            point_cloud: 点云数据，形状为 (N, 3) 或 (N, 6)
            sensor_position: 传感器位置 [x, y, z]
            voxel_size: 体素大小
            max_range: 最大扫描距离
            fov_horizontal: 水平视场角（度）
            fov_vertical: 垂直视场角（度）
            seed: 随机种子
            add_noise: 是否添加距离噪声
            add_dropout: 是否添加随机丢失
            
        Returns:
            (模拟扫描后的点云数据, 保留点的原始索引)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # 输入处理
        if hasattr(point_cloud, 'cpu') and hasattr(point_cloud, 'numpy'):
            original_pcd_np = point_cloud.cpu().numpy()
        else:
            original_pcd_np = np.asarray(point_cloud)

        if original_pcd_np.size == 0:
            return np.empty((0, original_pcd_np.shape[1]), dtype=original_pcd_np.dtype), np.array([], dtype=np.int64)

        points_xyz = original_pcd_np[:, :3]
        sensor_position = np.asarray(sensor_position)

        print("1. 正在进行距离和FOV预过滤...")
        
        # 计算点到传感器的相对位置和距离
        points_relative = points_xyz - sensor_position
        distances = np.linalg.norm(points_relative, axis=1)

        # 距离过滤
        range_mask = distances <= max_range
        
        # FOV过滤
        distances_safe = np.where(distances == 0, 1e-6, distances)
        
        # 计算垂直角度 (仰角)
        vertical_angles = np.arcsin(points_relative[:, 2] / distances_safe) * 180.0 / np.pi
        v_fov_mask = np.abs(vertical_angles) <= fov_vertical / 2.0
        
        # 计算水平角度 (方位角)
        if fov_horizontal < 360.0:
            horizontal_angles = np.arctan2(points_relative[:, 1], points_relative[:, 0]) * 180.0 / np.pi
            h_fov_mask = np.abs(horizontal_angles) <= fov_horizontal / 2.0
        else:
            h_fov_mask = np.ones_like(range_mask, dtype=bool)

        # 合并所有过滤条件
        pre_filter_mask = range_mask & v_fov_mask & h_fov_mask
        
        pre_filtered_indices = np.where(pre_filter_mask)[0]
        if pre_filtered_indices.size == 0:
            return np.empty((0, original_pcd_np.shape[1]), dtype=original_pcd_np.dtype), np.array([], dtype=np.int64)
            
        pre_filtered_points = points_xyz[pre_filtered_indices]
        print(f"   - 预过滤后剩余 {len(pre_filtered_points)} 个点。")

        # 为预过滤后的点创建体素网格
        print("2. 正在为过滤后的点创建体素网格...")
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pre_filtered_points)
        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size=voxel_size)
        occupied_voxels = voxel_grid.get_voxels()
        occupied_indices_set = {tuple(v.grid_index) for v in occupied_voxels}
        print(f"   - 创建了 {len(occupied_indices_set)} 个体素。")

        # 光线投射以检测遮挡
        print("3. 正在执行光线投射以检测遮挡...")
        visible_voxel_indices = set()

        for voxel in tqdm(occupied_voxels, desc="  - 检查体素可见性"):
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

        print(f"   - 发现 {len(visible_voxel_indices)} 个可见体素。")

        # 收集可见体素内的所有点
        print("4. 正在收集可见体素内的点...")
        final_indices = []
        
        for i, point in enumerate(pre_filtered_points):
            point_voxel_idx = voxel_grid.get_voxel(point)
            if tuple(point_voxel_idx) in visible_voxel_indices:
                final_indices.append(pre_filtered_indices[i])

        final_indices = np.array(final_indices, dtype=np.int64)
        
        if final_indices.size == 0:
            return np.empty((0, original_pcd_np.shape[1]), dtype=original_pcd_np.dtype), np.array([], dtype=np.int64)

        simulated_points = original_pcd_np[final_indices]
        print(f"   - 遮挡检测后保留 {len(simulated_points)} 个点。")

        # 添加扰动模拟
        if len(simulated_points) > 0:
            # 1. 添加距离噪声
            if add_noise and hasattr(self.config, 'range_noise_std') and self.config.range_noise_std > 0:
                print("5. 正在添加距离噪声...")
                points_xyz = simulated_points[:, :3]
                distances = np.linalg.norm(points_xyz - sensor_position, axis=1)
                
                # 添加高斯噪声
                noise = np.random.normal(0.0, self.config.range_noise_std, size=distances.shape[0])
                distances_noisy = distances + noise
                distances_noisy = np.clip(distances_noisy, 0.0, max_range)
                
                # 重新计算坐标（保持方向不变，只改变距离）
                directions = (points_xyz - sensor_position) / (distances[:, np.newaxis] + 1e-9)
                points_xyz_noisy = sensor_position + directions * distances_noisy[:, np.newaxis]
                
                # 更新点云坐标
                simulated_points = simulated_points.copy()
                simulated_points[:, :3] = points_xyz_noisy
                print(f"   - 添加距离噪声完成 (std={self.config.range_noise_std})")
            
            # 2. 随机丢失
            if add_dropout and hasattr(self.config, 'dropout_prob') and self.config.dropout_prob > 0:
                print("6. 正在添加随机丢失...")
                keep_mask = np.random.rand(len(simulated_points)) > self.config.dropout_prob
                if np.any(keep_mask):
                    simulated_points = simulated_points[keep_mask]
                    final_indices = final_indices[keep_mask]
                    print(f"   - 随机丢失完成 (dropout_prob={self.config.dropout_prob})，剩余 {len(simulated_points)} 个点")
                else:
                    print("   - 警告：所有点都被随机丢失")
                    return np.empty((0, original_pcd_np.shape[1]), dtype=original_pcd_np.dtype), np.array([], dtype=np.int64)

        print(f"   - 最终保留 {len(simulated_points)} 个点。")

        return simulated_points, final_indices


    def save_config(self, filepath: str):
        """保存配置到JSON文件"""
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
        print(f"配置已保存到: {filepath}")

    def load_config(self, filepath: str):
        """从JSON文件加载配置"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.config = ScanSimConfig(**config_dict)
        print(f"配置已从 {filepath} 加载")


def find_point_cloud_files(input_dir: str) -> list:
    """
    在输入目录中查找所有支持的点云文件
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        点云文件路径列表
    """
    supported_extensions = ['.ply', '.pcd', '.txt', '.npy']
    point_cloud_files = []
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    if os.path.isfile(input_dir):
        # 如果输入是单个文件，直接返回
        if any(input_dir.lower().endswith(ext) for ext in supported_extensions):
            return [input_dir]
        else:
            raise ValueError(f"不支持的文件格式: {input_dir}")
    
    # 遍历目录查找点云文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                point_cloud_files.append(os.path.join(root, file))
    
    return sorted(point_cloud_files)


def load_point_cloud(filepath: str) -> np.ndarray:
    """
    加载点云文件
    
    Args:
        filepath: 点云文件路径
        
    Returns:
        点云数据数组
    """
    print(f"正在加载点云文件: {filepath}")
    
    if filepath.endswith('.npy'):
        points = np.load(filepath)
    elif filepath.endswith('.txt'):
        points = np.loadtxt(filepath)
    elif filepath.endswith('.ply') or filepath.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"不支持的文件格式: {filepath}")
    
    print(f"加载了 {len(points)} 个点")
    return points


def analyze_point_cloud_scene(points: np.ndarray, verbose: bool = False) -> dict:
    """
    分析点云场景的统计信息
    
    Args:
        points: 点云数据，形状为 (N, 3)
        
    Returns:
        包含场景统计信息的字典
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
        print(f"点云场景分析:")
        print(f"  - 点数: {scene_info['num_points']}")
        print(f"  - 场景尺寸: [{dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}]")
        print(f"  - 场景中心: [{center_coords[0]:.2f}, {center_coords[1]:.2f}, {center_coords[2]:.2f}]")
        print(f"  - 高度范围: {min_coords[2]:.2f} ~ {max_coords[2]:.2f} (平均: {np.mean(points[:, 2]):.2f})")
    
    return scene_info


def save_point_cloud(points: np.ndarray, filepath: str):
    """
    保存点云文件
    推荐使用PLY格式，具有更小的文件体积和更好的兼容性
    
    Args:
        points: 点云数据，形状为 (N, 3) 或 (N, 6)
        filepath: 保存路径，支持 .ply, .pcd, .txt, .npy 格式
    """
    print(f"正在保存结果到: {filepath}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.npy'):
        np.save(filepath, points)
    elif filepath.endswith('.txt'):
        # 对于txt文件，保持原始数据精度，不做格式化处理
        if points.shape[1] >= 6:  # 包含 x,y,z,r,g,b
            # 保存为带标题的格式，保持原始精度
            header = "x y z r g b"
            points_to_save = points[:, :6]  # 只保存前6列：x,y,z,r,g,b
            np.savetxt(filepath, points_to_save, header=header, comments='')
        else:
            # 只有xyz坐标，保持原始精度
            np.savetxt(filepath, points)
    elif filepath.endswith('.ply'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] >= 6:  # 包含颜色信息
            # 确保颜色值在[0,1]范围内
            colors = points[:, 3:6].astype(np.float32)
            if colors.max() > 1.0:  # 如果颜色值大于1，说明是0-255范围
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 直接保存PLY文件，Open3D会自动选择二进制格式
        o3d.io.write_point_cloud(filepath, pcd)
    elif filepath.endswith('.pcd'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] >= 6:  # 包含颜色信息
            # 确保颜色值在[0,1]范围内
            colors = points[:, 3:6].astype(np.float32)
            if colors.max() > 1.0:  # 如果颜色值大于1，说明是0-255范围
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 直接保存PCD文件
        o3d.io.write_point_cloud(filepath, pcd)
    else:
        raise ValueError(f"不支持的输出文件格式: {filepath}")


def check_processed_files(input_files: list, output_dir: str, output_format: str) -> Tuple[list, list]:
    """
    检查哪些文件已经被处理过，返回需要处理的文件列表和已处理的文件列表
    
    Args:
        input_files: 输入文件列表
        output_dir: 输出目录
        output_format: 输出文件格式
        
    Returns:
        (需要处理的文件列表, 已处理的文件列表)
    """
    if not os.path.exists(output_dir):
        return input_files, []
    
    # 获取输出目录中已有的文件
    existing_files = set()
    for file in os.listdir(output_dir):
        if file.endswith(f'_scanned.{output_format}') or file.endswith('.npy'):
            # 提取原始文件名（去掉_scanned后缀）
            if file.endswith(f'_scanned.{output_format}'):
                base_name = file.replace(f'_scanned.{output_format}', '')
            else:  # .npy文件
                base_name = file.replace('.npy', '')
            existing_files.add(base_name)
    
    # 分离需要处理和已处理的文件
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
    """创建一些常用的预设配置"""
    configs = {}
    
    # 标准LiDAR配置
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
    
    # 高精度配置
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
    
    # 室内环境配置
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
    """多传感器模式下，处理单个文件的工作函数（用于文件级多进程）。"""
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
        # 构造配置与模拟器
        config = ScanSimConfig(**config_dict)
        from multi_sensor import MultiSensorScanSimulator  # 延迟导入
        multi_simulator = MultiSensorScanSimulator(config)
        # 应用布局参数
        multi_simulator.layout_planner.grid_resolution = layout_params['grid_resolution']
        multi_simulator.layout_planner.min_sensor_distance = layout_params['min_sensor_distance']
        multi_simulator.layout_planner.max_sensor_distance = layout_params['max_sensor_distance']
        multi_simulator.layout_planner.max_sensors = layout_params['max_sensors']
        multi_simulator.layout_planner.min_coverage_threshold = layout_params['min_coverage_threshold']

        # 加载点云
        points = load_point_cloud(input_file)
        result['points_before'] = len(points)
        # 提取标签列（若存在），用于更准确的布局分析
        labels = points[:, 6].astype(int) if (isinstance(points, np.ndarray) and points.shape[1] > 6) else None

        # 执行规划与扫描（内部线性执行，不再多进程）
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
            # 新增：多传感器扫描改进参数
            use_iterative_optimization=multi_sensor_params['use_iterative_optimization'],
            target_sensor_count=multi_sensor_params['target_sensor_count'],
            max_iterations=multi_sensor_params['max_iterations'],
            coverage_improvement_threshold=multi_sensor_params['coverage_improvement_threshold']
        )

        result['points_after'] = len(merged_cloud)

        # 保存输出
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
        print(f"处理文件 {os.path.basename(input_file)} 时出错: {e}")
    finally:
        result['processing_time'] = time.time() - start_time

    return result


def main():
    """主函数 - 多传感器扫描命令行接口"""
    parser = argparse.ArgumentParser(description='多传感器虚拟扫描模拟器')
    parser.add_argument('--input', type=str, default='data/bim', help='输入点云文件夹路径')
    parser.add_argument('--output', type=str, default='multi_scan_sim', help='输出模拟扫描结果文件夹路径')
    parser.add_argument('--config', type=str, help='配置文件路径 (JSON格式)')
    parser.add_argument('--preset', type=str, choices=['standard_lidar', 'high_resolution', 'indoor'], 
                       default='standard_lidar', help='使用预设配置')
    parser.add_argument('--output_format', type=str, choices=['txt', 'ply', 'pcd', 'npy'], 
                       default='ply', help='输出文件格式 (默认PLY格式，体积更小)')
    parser.add_argument('--num_processes', type=int, default=16,
                       help='并行处理的进程数 (默认为CPU核心数)')
    parser.add_argument('--disable_multiprocess', action='store_true', default=True,
                       help='禁用多进程，使用单进程处理')
    
    # 多传感器扫描选项
    parser.add_argument('--max_sensors', type=int, default=6,
                       help='最大传感器数量')
    parser.add_argument('--min_coverage_threshold', type=float, default=0.85,
                       help='最小覆盖度阈值')
    parser.add_argument('--grid_resolution', type=float, default=0.2,
                       help='网格分辨率')
    parser.add_argument('--min_sensor_distance', type=float, default=2.0,
                       help='传感器最小距离')
    parser.add_argument('--max_sensor_distance', type=float, default=8.0,
                       help='传感器最大扫描距离')
    
    # 新增：多传感器扫描改进选项
    parser.add_argument('--use_iterative_optimization', action='store_true', default=True,
                       help='启用迭代优化策略，根据实际扫描结果动态调整传感器位置')
    parser.add_argument('--target_sensor_count', type=int, default=None,
                       help='手动指定传感器数量，如果指定将强制使用该数量')
    parser.add_argument('--max_iterations', type=int, default=5,
                       help='迭代优化的最大迭代次数')
    parser.add_argument('--coverage_improvement_threshold', type=float, default=0.05,
                       help='覆盖率改进阈值，低于此值将停止迭代')
    
    # 扫描模拟参数
    parser.add_argument('--voxel_size', type=float, default=0.05, 
                       help='体素大小')
    parser.add_argument('--max_range', type=float, default=10.0, 
                       help='最大扫描距离')
    parser.add_argument('--fov_horizontal', type=float, default=360.0, 
                       help='水平视场角 (度)')
    parser.add_argument('--fov_vertical', type=float, default=300.0, 
                       help='垂直视场角 (度)')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--disable_noise', action='store_true',
                       help='禁用距离噪声模拟')
    parser.add_argument('--disable_dropout', action='store_true', default=True,
                       help='禁用随机丢失模拟')
    parser.add_argument('--sensor_height', type=float, default=1.5,
                       help='传感器高度 (米)。如果提供，将覆盖配置与布局规划器中的高度')
    
    # 断点继续功能选项
    parser.add_argument('--resume', action='store_true', default=True,
                       help='启用断点继续功能，自动跳过已处理的文件。默认情况下会重新处理所有文件')
    
    args = parser.parse_args()
    
    # 查找输入文件夹中的所有点云文件
    print(f"正在查找输入文件夹中的点云文件: {args.input}")
    print(f"输出格式: {args.output_format} (推荐PLY格式，体积更小)")
    input_files = find_point_cloud_files(args.input)
    
    if not input_files:
        print("未找到任何点云文件！")
        return
    
    print(f"找到 {len(input_files)} 个点云文件:")
    for i, file_path in enumerate(input_files):
        print(f"  {i+1}. {file_path}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 断点继续功能处理
    if args.resume:
        print("\n=== 断点继续模式 ===")
        files_to_process, already_processed = check_processed_files(input_files, args.output, args.output_format)
        if already_processed:
            print(f"发现 {len(already_processed)} 个已处理的文件，将跳过:")
            for i, file_path in enumerate(already_processed):
                print(f"  {i+1}. {os.path.basename(file_path)}")
        print(f"需要处理 {len(files_to_process)} 个文件")
    else:
        print("\n=== 正常处理模式 ===")
        files_to_process = input_files
        already_processed = []
        print("提示: 使用 --resume 参数可以启用断点继续功能，自动跳过已处理的文件")
    
    if not files_to_process:
        print("所有文件都已处理完成！")
        return
    
    print(f"\n=== 多传感器扫描模式 ===")
    print(f"最大传感器数量: {args.max_sensors}")
    print(f"最小覆盖度阈值: {args.min_coverage_threshold}")
    print(f"网格分辨率: {args.grid_resolution}m")
    print(f"传感器最小距离: {args.min_sensor_distance}m")
    
    # 显示新的多传感器扫描改进选项
    if args.target_sensor_count is not None:
        print(f"手动指定传感器数量: {args.target_sensor_count}")
    if args.use_iterative_optimization:
        print(f"迭代优化: 启用 (最大迭代次数: {args.max_iterations}, 改进阈值: {args.coverage_improvement_threshold})")
    else:
        print(f"迭代优化: 禁用 (使用传统规划方法)")
    
    # 创建多传感器扫描模拟器
    if args.config:
        config = ScanSimConfig()
        simulator = VirtualScanSimulator()
        simulator.load_config(args.config)
        multi_simulator = MultiSensorScanSimulator(simulator.config)
    else:
        configs = create_default_configs()
        multi_simulator = MultiSensorScanSimulator(configs[args.preset])

    # 覆盖传感器高度（同时作用于配置与布局规划器）
    if args.sensor_height is not None:
        try:
            multi_simulator.config.sensor_height = float(args.sensor_height)
        except Exception:
            pass
        try:
            multi_simulator.layout_planner.sensor_height = float(args.sensor_height)
        except Exception:
            pass

    print(f"应用的传感器高度: config={multi_simulator.config.sensor_height}, layout_planner={multi_simulator.layout_planner.sensor_height}")
    
    # 配置布局规划器参数
    multi_simulator.layout_planner.grid_resolution = args.grid_resolution
    multi_simulator.layout_planner.min_sensor_distance = args.min_sensor_distance
    multi_simulator.layout_planner.max_sensor_distance = args.max_sensor_distance
    multi_simulator.layout_planner.max_sensors = args.max_sensors
    multi_simulator.layout_planner.min_coverage_threshold = args.min_coverage_threshold
    
    # 准备配置字典（用于子进程构造配置）
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

    # 布局与扫描参数
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
    # 新增：多传感器扫描改进参数
    multi_sensor_params = {
        'use_iterative_optimization': args.use_iterative_optimization,
        'target_sensor_count': args.target_sensor_count,
        'max_iterations': args.max_iterations,
        'coverage_improvement_threshold': args.coverage_improvement_threshold,
    }
    io_params = {
        'output_format': args.output_format,
    }

    # 组织参数（只处理需要处理的文件）
    worker_args = [
        (input_file, args.output, config_dict, layout_params, scan_params, multi_sensor_params, io_params)
        for input_file in files_to_process
    ]

    # 文件级处理（可多进程）
    if args.disable_multiprocess:
        results = []
        for i, wargs in enumerate(worker_args):
            print(f"\n处理文件 {i+1}/{len(worker_args)}: {os.path.basename(wargs[0])}")
            results.append(_process_file_worker(wargs))
    else:
        num_proc = args.num_processes if args.num_processes else mp.cpu_count()
        print(f"\n使用文件级多进程，进程数: {num_proc}")
        with mp.Pool(processes=num_proc) as pool:
            results = list(tqdm(
                pool.imap(_process_file_worker, worker_args),
                total=len(worker_args),
                desc="处理点云文件",
                unit="文件"
            ))

    # 汇总
    success_count = sum(1 for r in results if r.get('success'))
    fail_count = len(results) - success_count
    skipped_count = len(already_processed)
    
    print(f"\n多传感器扫描处理完成！结果保存在: {args.output}")
    print(f"成功: {success_count}，失败: {fail_count}")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} (已处理)")
    
    if args.resume and skipped_count > 0:
        print(f"\n断点继续功能: 自动跳过了 {skipped_count} 个已处理的文件")
        print("如需重新处理所有文件，请不使用 --resume 参数")


# 导入多传感器功能

from multi_sensor import MultiSensorScanSimulator, MultiSensorLayoutPlanner
from multi_sensor.scan_simulator import _simulate_single_sensor_worker



if __name__ == '__main__':
    main()
