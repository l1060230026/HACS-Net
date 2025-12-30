import numpy as np
import logging
from pathlib import Path
from data_utils.pseudo_labeling import generate_voxel_based_pseudo_labels, generate_voxel_based_pseudo_labels_optimized

logger = logging.getLogger(__name__)


def generate_and_filter_pseudo_labels(
    model,
    target_dataset,
    epoch,
    args,
    num_classes,
    num_point,
    voxel_size,
    config,
    use_voxel_based=True
):

    # Use config parameters if provided, otherwise use function arguments or defaults
    pseudo_config = config.get('pseudo_labels', {})
    batch_size = pseudo_config.get('batch_size', 8)
    device = pseudo_config.get('device', 'cuda')
    threshold_mode = pseudo_config.get('threshold_mode', 'global')  # 'global' or 'class'

    # Use voxel-based sampling method
    room_pseudo, room_confidence = generate_voxel_based_pseudo_labels_optimized(
        model=model,
        dataset=target_dataset,
        num_classes=num_classes,
        voxel_size=voxel_size,
        voxel_max=num_point,
        batch_size=batch_size,
        device=device,
    )

    # Apply thresholding based on configuration
    if threshold_mode == 'global':
        # Global confidence threshold
        global_conf = float(pseudo_config.get('global_confidence_threshold', 0.7))
        if room_confidence is not None:
            filtered = []
            class_ratio = np.zeros(num_classes)
            for labels, conf in zip(room_pseudo, room_confidence):
                labels = labels.copy()
                labels[conf < global_conf] = -1
                filtered.append(labels)
                for cid in range(num_classes):
                    cid_mask = (labels == cid)
                    if np.any(cid_mask):
                        class_ratio[cid] += (cid_mask).sum() / 10000
            class_ratio = class_ratio / class_ratio.sum()
            nonzero_indices = np.nonzero(class_ratio)[0]
            num_class = config.tacm.cuboid_queue.num_class
            tail_class_indices = nonzero_indices[np.argsort(class_ratio[nonzero_indices])[:num_class]]
            room_pseudo = filtered
    else:
        # Per-class ratio thresholds using confidence per point
        ratios = pseudo_config.get('class_ratio_thresholds', [0.3])
        # Normalize ratios list to num_classes
        if len(ratios) == 1:
            ratios = ratios * num_classes
        ratios = [float(r) for r in ratios]

        # Build per-class confidence aggregates (memory-efficient using numpy)
        all_conf = [[] for _ in range(num_classes)]
        for labels, conf in zip(room_pseudo, room_confidence):
            for cid in range(num_classes):
                mask = (labels == cid)
                if np.any(mask):
                    # Convert to list in smaller chunks to avoid memory spike
                    all_conf[cid].append(conf[mask])

        # Determine per-class thresholds at specified top ratios
        thres = []
        for cid in range(num_classes):
            if len(all_conf[cid]) == 0:
                thres.append(0.0)
            else:
                # Concatenate and sort only once per class
                conf_array = np.concatenate(all_conf[cid])
                conf_array.sort()
                conf_array = conf_array[::-1]  # Reverse to get descending order
                k = max(1, int(ratios[cid] * len(conf_array)))
                thres.append(float(conf_array[k - 1]))
                # Free memory immediately
                del conf_array
        
        # Clear all_conf to free memory
        del all_conf

        # Apply per-class thresholds per room
        filtered = []
        class_ratio = np.zeros(num_classes)
        for labels, conf in zip(room_pseudo, room_confidence):
            labels = labels.copy()
            for cid in range(num_classes):
                cid_mask = (labels == cid)
                if np.any(cid_mask):
                    low_mask = conf < thres[cid]
                    labels[cid_mask & low_mask] = -1
                    class_ratio[cid] += (cid_mask & low_mask).sum() / 10000
            filtered.append(labels)
        room_pseudo = filtered

        class_ratio = class_ratio / class_ratio.sum()
        nonzero_indices = np.nonzero(class_ratio)[0]
        num_class = config.tacm.cuboid_queue.num_class
        tail_class_indices = nonzero_indices[np.argsort(class_ratio[nonzero_indices])[:num_class]]

    # Set room pseudo labels to target dataset for point alignment
    return room_pseudo, class_ratio, tail_class_indices
