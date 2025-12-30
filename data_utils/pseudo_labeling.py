import os
import numpy as np
import torch
from typing import List, Tuple, Optional
from tqdm import tqdm

import provider
from .data_util import voxelize


def add_vote(vote_label_pool, point_idx, pred_label, weight, confidence_pool=None, confidence=None):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                point_id = int(point_idx[b, n])
                label_id = int(pred_label[b, n])
                vote_label_pool[point_id, label_id] += 1
                
                # Collect confidence information
                if confidence_pool is not None and confidence is not None:
                    # Use weighted average to accumulate confidence
                    current_weight = vote_label_pool[point_id, label_id]
                    if current_weight == 1:  # First vote
                        confidence_pool[point_id, label_id] = confidence[b, n]
                    else:  # Subsequent votes, use weighted average
                        prev_weight = current_weight - 1
                        confidence_pool[point_id, label_id] = (
                            confidence_pool[point_id, label_id] * prev_weight + confidence[b, n]
                        ) / current_weight
    
    return vote_label_pool



def generate_room_pseudo_labels(
    model,
    dataset,
    num_classes: int,
    batch_size: int = 16,
    device: str = 'cuda',
    num_point: Optional[int] = None,
    num_votes: int = 1,
    block_size: float = 2.0,
    stride: float = 0.5,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate pseudo labels per room using the same block-based rotation and voting
    scheme as in test_with_att_baseline.py.

    Returns:
        (pseudo_labels_per_room, confidence_per_room)
        - pseudo_labels_per_room: list of per-point labels for each room (aligned to original points)
        - confidence_per_room: list of per-point confidences computed as the softmax probability 
          of the predicted label (max value after softmax)
    """

    model = model.to(device)
    model.eval()

    pseudo_labels_per_room = []
    confidence_per_room = []


    with torch.no_grad():
        scene_id = dataset.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(dataset)
        
        for batch_idx in tqdm(range(num_batches), total=num_batches):

            whole_scene_data = dataset.scene_points_list[batch_idx]
            whole_scene_label = dataset.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], num_classes))
            confidence_pool = np.zeros((whole_scene_label.shape[0], num_classes))
            
            for _ in range(num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = dataset[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + batch_size - 1) // batch_size

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * batch_size
                    end_idx = min((sbatch + 1) * batch_size, num_blocks)
                    real_batch_size = end_idx - start_idx
                    
                    # Prepare batch data similar to training
                    batch_data = scene_data[start_idx:end_idx, ...]  # [B, N, 3]
                    batch_label = scene_label[start_idx:end_idx, ...]  # [B, N]
                    batch_point_index = scene_point_index[start_idx:end_idx, ...]  # [B, N]
                    batch_smpw = scene_smpw[start_idx:end_idx, ...]  # [B, N]
                    
                    # Apply rotation augmentation (same as training)
                    points_np = batch_data
                    points_np[:, :, :3] = provider.rotate_point_cloud_z(points_np[:, :, :3])
                    points_t = torch.Tensor(points_np)  # [B, N, C]

                    B = points_t.shape[0]
                    N = points_t.shape[1]

                    # Prepare input for attention model
                    coord = points_t[:, :, :3].contiguous().view(-1, 3).float().cuda(non_blocking=True)  # (B*N, 3)
                    offset = torch.cumsum(torch.tensor([N]*B, dtype=torch.int32, device=coord.device), dim=0).contiguous().int()  # (B,)
                    
                    # Forward pass through attention model
                    seg_pred, trans_feat = model([coord, coord, offset])  # use coords as features (c==3)
                    # seg_pred: (B*N, C)
                    
                    # Get predictions and confidence
                    seg_pred_softmax = torch.softmax(seg_pred, dim=1)  # Apply softmax
                    batch_pred_confidence = seg_pred_softmax.contiguous().cpu().data.max(1)[0].numpy()  # (B*N,) - max confidence
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(1)[1].numpy()  # (B*N,) - predicted label
                    
                    batch_pred_confidence = batch_pred_confidence.reshape(B, N)  # (B, N)
                    batch_pred_label = batch_pred_label.reshape(B, N)  # (B, N)

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...],
                                               confidence_pool, batch_pred_confidence[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)
            
            # Get confidence corresponding to final predicted label for each point
            point_confidence = np.zeros((whole_scene_label.shape[0]))
            for i in range(len(point_confidence)):
                point_confidence[i] = confidence_pool[i, pred_label[i]]
            
            pseudo_labels_per_room.append(pred_label)
            confidence_per_room.append(point_confidence)

    return pseudo_labels_per_room, confidence_per_room


def generate_voxel_based_pseudo_labels(
    model,
    dataset,
    num_classes: int,
    voxel_size: float = 0.05,
    voxel_max: int = 4096,
    batch_size: int = 16,
    device: str = 'cuda',
    num_votes: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Voxel-based pseudo label generation function - consistent with test code logic
    
    Generate corresponding labels for original point cloud of each room in target_dataset. Specifically:
    1. Use same voxelization segmentation method as test code
    2. Process block by block and accumulate prediction results
    3. Finally take maximum value as label
    
    Args:
        model: Trained model
        dataset: Target dataset
        num_classes: Number of classes
        voxel_size: Voxel size
        batch_size: Batch size
        device: Device
        num_votes: Number of votes
        
    Returns:
        (room_pseudo, room_confidence)
        - room_pseudo: Pseudo label list for each room, aligned with original point cloud
        - room_confidence: Confidence list for each room
    """
    
    model = model.to(device)
    model.eval()
    
    pseudo_labels_per_room = []
    confidence_per_room = []
    
    with torch.no_grad():
        scene_id = dataset.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(dataset)

        for batch_idx in range(num_batches):

            whole_scene_data = dataset.scene_points_list[batch_idx]
            whole_scene_label = dataset.semantic_labels_list[batch_idx]


            coord, label, idx_data = data_load(whole_scene_data, whole_scene_label, voxel_size)
            pred = torch.zeros((label.size, num_classes)).cuda()

            idx_size = len(idx_data)
            idx_list, coord_list, feat_list, offset_list  = [], [], [], []
            
            for i in range(idx_size):
                idx_part = idx_data[i]
                coord_part = coord[idx_part]
 
                if voxel_max > 0 and coord_part.shape[0] > voxel_max:
                    coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                    while idx_uni.size != idx_part.shape[0]:
                        init_idx = np.argmin(coord_p)
                        dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                        idx_crop = np.argsort(dist)[:voxel_max]
                        coord_sub, idx_sub = coord_part[idx_crop], idx_part[idx_crop]
                        dist = dist[idx_crop]
                        delta = np.square(1 - dist / np.max(dist))
                        coord_p[idx_crop] += delta
                        coord_sub = input_normalize(coord_sub)
                        idx_list.append(idx_sub), coord_list.append(coord_sub), offset_list.append(idx_sub.size)
                        idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
                else:
                    coord_part = input_normalize(coord_part)
                    idx_list.append(idx_part), coord_list.append(coord_part), offset_list.append(idx_part.size)
            batch_num = int(np.ceil(len(idx_list) / batch_size))
            for i in range(batch_num):
                s_i, e_i = i * batch_size, min((i + 1) * batch_size, len(idx_list))
                idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = model([coord_part, feat_part, offset_part])[0]  # (n, k)
                torch.cuda.empty_cache()
                pred[idx_part, :] += pred_part

            pred_label = pred.max(1)[1].data.cpu().numpy()
            
            confidence = torch.softmax(pred, dim=1).max(1)[0].data.cpu().numpy()

            pseudo_labels_per_room.append(pred_label)
            confidence_per_room.append(confidence)
            
    
    return pseudo_labels_per_room, confidence_per_room


def generate_voxel_based_pseudo_labels_optimized(
    model,
    dataset,
    num_classes: int,
    voxel_size: float = 0.05,
    voxel_max: int = 4096,
    batch_size: int = 16,
    device: str = 'cuda',
    num_votes: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    优化版基于体素采样的伪标签生成函数
    
    通过体素网格采样来加速伪标签生成：
    1. 对每个体素只预测一次，然后将结果分配给该体素内的所有点
    2. 避免对同一体素内的多个点重复计算
    3. 显著提高生成速度，特别是在密集点云场景下
    
    Args:
        model: 训练好的模型
        dataset: 目标数据集
        num_classes: 类别数量
        voxel_size: 体素大小
        voxel_max: 每个体素最大点数
        batch_size: 批处理大小
        device: 设备
        num_votes: 投票次数
        
    Returns:
        (room_pseudo, room_confidence)
        - room_pseudo: 每个房间的伪标签列表，与原始点云对齐
        - room_confidence: 每个房间的置信度列表
    """
    
    model = model.to(device)
    model.eval()
    
    pseudo_labels_per_room = []
    confidence_per_room = []
    
    with torch.no_grad():
        scene_id = dataset.rooms_split
        scene_id = [x[:-4] for x in scene_id]
        num_rooms = len(scene_id)
        
        # 处理每个房间
        for room_idx in tqdm(range(num_rooms),total=num_rooms, desc="Processing rooms"):
           
            whole_scene_data = dataset.get_room_points(room_idx)
            
            # 获取原始点云坐标
            original_coords = whole_scene_data[:, :3]  # N x 3
            
            # 体素化处理
            coord_min = np.min(original_coords, 0)
            coord_normalized = original_coords - coord_min
            
            # 使用体素化函数获取体素索引和计数
            idx_sort, count = voxelize(coord_normalized, voxel_size, mode=1)
            
            # 为每个体素选择代表点（通常是第一个点）
            voxel_representative_indices = []
            voxel_to_points_mapping = {}  # 体素ID -> 该体素内所有点的索引
            
            current_idx = 0
            for voxel_id, voxel_count in enumerate(count):
                # 获取该体素内的所有点索引
                voxel_point_indices = idx_sort[current_idx:current_idx + voxel_count]
                
                # 选择第一个点作为代表点
                representative_idx = voxel_point_indices[0]
                voxel_representative_indices.append(representative_idx)
                
                # 记录体素到点的映射关系
                voxel_to_points_mapping[voxel_id] = voxel_point_indices
                
                current_idx += voxel_count
            
            # 准备体素代表点的坐标
            voxel_coords = coord_normalized[voxel_representative_indices]  # M x 3 (M个体素)
            num_voxels = len(voxel_representative_indices)
            
            # 初始化该房间的预测结果
            room_voxel_pred = torch.zeros((num_voxels, num_classes)).cuda()
            
            # 根据voxel_max决定分批次策略
            if voxel_max == -1:
                # 不受限制：一个房间的所有体素作为一个批次
                num_batches_room = 1
                batch_size_room = num_voxels
            else:
                # 受限制：按voxel_max分批，但不超过房间的体素总数
                num_batches_room = int(np.ceil(num_voxels / voxel_max))
                batch_size_room = voxel_max
           
            # 分批处理该房间的体素
            for batch_i in range(num_batches_room):
                start_idx = batch_i * batch_size_room
                end_idx = min((batch_i + 1) * batch_size_room, num_voxels)
                actual_batch_size = end_idx - start_idx
                
                # 检查是否需要填充到指定批次大小
                if actual_batch_size < batch_size_room and batch_i == num_batches_room - 1:
                    # 最后一个批次且体素数量不足，需要填充
                    padding_needed = batch_size_room - actual_batch_size
                    
                    # 从前面批次中随机选择体素进行填充
                    if start_idx > 0:
                        # 从前面已处理的体素中随机选择
                        padding_indices = np.random.choice(start_idx, size=padding_needed, replace=True)
                        padding_coords = voxel_coords[padding_indices]
                        
                        # 合并当前批次和填充体素s
                        batch_voxel_coords = np.vstack([voxel_coords[start_idx:end_idx], padding_coords])
                        batch_voxel_coords = input_normalize(batch_voxel_coords)
                        
                        # 准备模型输入
                        coord_tensor = torch.FloatTensor(batch_voxel_coords).cuda(non_blocking=True)
                        offset_tensor = torch.IntTensor([batch_voxel_coords.shape[0]]).cuda(non_blocking=True)
                        
                        # 模型预测
                        with torch.no_grad():
                            pred_batch_full = model([coord_tensor, coord_tensor, offset_tensor])[0]  # (batch_size_room, num_classes)
                        
                        # 只使用当前批次体素的预测结果
                        pred_batch = pred_batch_full[:actual_batch_size]
                        del pred_batch_full  # Clean up full batch

                    else:
                        # 如果这是第一个批次且体素数量不足，直接处理
                        batch_voxel_coords = voxel_coords[start_idx:end_idx]
                        batch_voxel_coords = input_normalize(batch_voxel_coords)
                        
                        # 准备模型输入
                        coord_tensor = torch.FloatTensor(batch_voxel_coords).cuda(non_blocking=True)
                        offset_tensor = torch.IntTensor([batch_voxel_coords.shape[0]]).cuda(non_blocking=True)
                        
                        # 模型预测
                        with torch.no_grad():
                            pred_batch = model([coord_tensor, coord_tensor, offset_tensor])[0]  # (actual_batch_size, num_classes)
                        
                else:
                    # 正常批次处理
                    batch_voxel_coords = voxel_coords[start_idx:end_idx]
                    batch_voxel_coords = input_normalize(batch_voxel_coords)
                    
                    # 准备模型输入
                    coord_tensor = torch.FloatTensor(batch_voxel_coords).cuda(non_blocking=True)
                    offset_tensor = torch.IntTensor([batch_voxel_coords.shape[0]]).cuda(non_blocking=True)
                    
                    # 模型预测
                    with torch.no_grad():
                        pred_batch = model([coord_tensor, coord_tensor, offset_tensor])[0]  # (actual_batch_size, num_classes)
                
                # 累积预测结果
                room_voxel_pred[start_idx:end_idx, :] += pred_batch
                
                # Clean up batch tensors
                del coord_tensor, offset_tensor, pred_batch
                if batch_i % 5 == 0:  # Clear cache periodically
                    torch.cuda.empty_cache()
            
            # 获取该房间体素级别的预测结果
            room_voxel_pred_labels = room_voxel_pred.max(1)[1].data.cpu().numpy()  # (num_voxels,)
            room_voxel_pred_confidence = torch.softmax(room_voxel_pred, dim=1).max(1)[0].data.cpu().numpy()  # (num_voxels,)
            
            # 将体素预测结果分配给所有原始点
            num_original_points = len(original_coords)
            point_pred_labels = np.zeros(num_original_points, dtype=np.int32)
            point_pred_confidence = np.zeros(num_original_points, dtype=np.float32)
            
            for voxel_id in range(num_voxels):
                voxel_label = room_voxel_pred_labels[voxel_id]
                voxel_confidence = room_voxel_pred_confidence[voxel_id]
                
                # 将该体素的预测结果分配给体素内的所有点
                point_indices = voxel_to_points_mapping[voxel_id]
                point_pred_labels[point_indices] = voxel_label
                point_pred_confidence[point_indices] = voxel_confidence
            
            pseudo_labels_per_room.append(point_pred_labels)
            confidence_per_room.append(point_pred_confidence)
            
            # Clean up room-level tensors and data structures
            del room_voxel_pred, room_voxel_pred_labels, room_voxel_pred_confidence
            del voxel_to_points_mapping, voxel_coords, voxel_representative_indices
            del whole_scene_data, original_coords, coord_normalized
            
            # Periodic cache clearing
            if room_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    return pseudo_labels_per_room, confidence_per_room


def data_load(points, label, voxel_size):

    coord = points[:, :3]  # N * 3

    idx_data = []

    coord_min = np.min(coord, 0)
    coord -= coord_min
    idx_sort, count = voxelize(coord, voxel_size, mode=1)
    for i in range(count.max()):
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
        idx_part = idx_sort[idx_select]
        idx_data.append(idx_part)

    return coord, label, idx_data


def input_normalize(coord):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    return coord