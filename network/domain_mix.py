"""
LaserMix domain mixing - for Stage 2 SAC_LM
Mix source and target domain point clouds by angle regions
"""
import numpy as np
import torch


def laserMix(src_coord, src_label, tgt_coord, tgt_label=None, num_areas=None):
    """
    Mix source and target domain point clouds by pitch angle regions
    
    Core idea:
    1. Divide point cloud into multiple regions by pitch angle
    2. Randomly select some regions to replace source with target domain
    3. Maintain spatial consistency
    
    Args:
        src_coord: [N_src, 3] Source domain point cloud coordinates
        src_label: [N_src] Source domain labels
        tgt_coord: [N_tgt, 3] Target domain point cloud coordinates
        tgt_label: [N_tgt] Target domain labels (optional, for pseudo labels)
        num_areas: Number of regions, if None then randomly choose 3-6
    
    Returns:
        mixed_coord: [N_mixed, 3] Mixed coordinates
        mixed_label: [N_mixed] Mixed labels
        src_mask: [N_mixed] Boolean mask, True indicates from source domain
        tgt_mask: [N_mixed] Boolean mask, True indicates from target domain
    """
    if num_areas is None:
        num_areas = np.random.choice([3, 4, 5, 6])
    
    # Compute pitch angle for each point: pitch = atan2(z, sqrt(x^2 + y^2))
    src_pitch = np.arctan2(src_coord[:, 2], np.sqrt(src_coord[:, 0]**2 + src_coord[:, 1]**2))
    tgt_pitch = np.arctan2(tgt_coord[:, 2], np.sqrt(tgt_coord[:, 0]**2 + tgt_coord[:, 1]**2))
    
    # Compute angle range
    src_pitch_min, src_pitch_max = src_pitch.min(), src_pitch.max()
    tgt_pitch_min, tgt_pitch_max = tgt_pitch.min(), tgt_pitch.max()
    pitch_min = min(src_pitch_min, tgt_pitch_min)
    pitch_max = max(src_pitch_max, tgt_pitch_max)
    
    # Divide angle range into multiple regions
    angle_list = np.linspace(pitch_max, pitch_min, num_areas + 1)
    
    # Randomly decide whether to replace each region
    replace_mask = np.random.rand(num_areas) > 0.5  # 50% probability to replace
    
    # Build mixed point cloud
    mixed_coords = []
    mixed_labels = []
    src_mask_list = []
    tgt_mask_list = []
    
    for i in range(num_areas):
        angle_start, angle_end = angle_list[i], angle_list[i+1]
        
        if replace_mask[i]:
            # Use target domain points
            tgt_mask = (tgt_pitch >= angle_end) & (tgt_pitch < angle_start)
            if i == num_areas - 1:  # Last region includes upper bound
                tgt_mask = (tgt_pitch >= angle_end) & (tgt_pitch <= angle_start)
            
            if tgt_mask.sum() > 0:
                mixed_coords.append(tgt_coord[tgt_mask])
                if tgt_label is not None:
                    mixed_labels.append(tgt_label[tgt_mask])
                else:
                    mixed_labels.append(np.zeros(tgt_mask.sum(), dtype=np.int64))
                src_mask_list.append(np.zeros(tgt_mask.sum(), dtype=bool))
                tgt_mask_list.append(np.ones(tgt_mask.sum(), dtype=bool))
        else:
            # Use source domain points
            src_mask = (src_pitch >= angle_end) & (src_pitch < angle_start)
            if i == num_areas - 1:  # Last region includes upper bound
                src_mask = (src_pitch >= angle_end) & (src_pitch <= angle_start)
            
            if src_mask.sum() > 0:
                mixed_coords.append(src_coord[src_mask])
                mixed_labels.append(src_label[src_mask])
                src_mask_list.append(np.ones(src_mask.sum(), dtype=bool))
                tgt_mask_list.append(np.zeros(src_mask.sum(), dtype=bool))
    
    if len(mixed_coords) == 0:
        # If no points, return source domain data
        return src_coord, src_label, np.ones(len(src_coord), dtype=bool), np.zeros(len(src_coord), dtype=bool)
    
    # Merge all regions
    mixed_coord = np.concatenate(mixed_coords, axis=0)
    mixed_label = np.concatenate(mixed_labels, axis=0)
    src_mask = np.concatenate(src_mask_list, axis=0)
    tgt_mask = np.concatenate(tgt_mask_list, axis=0)
    
    return mixed_coord, mixed_label, src_mask, tgt_mask


def laserMix_batch(src_coords, src_labels, tgt_coords, tgt_labels=None):
    """
    Batch process LaserMix
    
    Args:
        src_coords: List of [N_i, 3] Source domain point cloud coordinate lists
        src_labels: List of [N_i] Source domain label lists
        tgt_coords: List of [M_i, 3] Target domain point cloud coordinate lists
        tgt_labels: List of [M_i] Target domain label lists (optional)
    
    Returns:
        mixed_coords: List of [K_i, 3] Mixed coordinate lists
        mixed_labels: List of [K_i] Mixed label lists
        src_masks: List of [K_i] Source domain mask lists
        tgt_masks: List of [K_i] Target domain mask lists
    """
    mixed_coords = []
    mixed_labels = []
    src_masks = []
    tgt_masks = []
    
    for i in range(len(src_coords)):
        tgt_label = tgt_labels[i] if tgt_labels is not None else None
        mc, ml, sm, tm = laserMix(
            src_coords[i], src_labels[i],
            tgt_coords[i], tgt_label
        )
        mixed_coords.append(mc)
        mixed_labels.append(ml)
        src_masks.append(sm)
        tgt_masks.append(tm)
    
    return mixed_coords, mixed_labels, src_masks, tgt_masks

