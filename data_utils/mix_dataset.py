"""
DODA-style TACM dataset implementation
Perfect replication of DODA's Tail-Aware Cuboid Mixing method
"""

import numpy as np
import torch
import random
from typing import List, Dict, Tuple, Optional
from types import SimpleNamespace
from easydict import EasyDict  # optional

from .data_util import data_prepare
from torch.utils.data import Dataset
from .transform import tacm


class Queue(object):
    def __init__(self, size):
        assert size > 0
        self.size = size
        self.queue = [None] * self.size
        self.ptr = 0
        self.cur_size = 0
        self.got = 0

    def update_queue(self, items):
        if len(items) == 0:
            return
        items = items[:self.size]  # update maximum self.size items
        new_ptr = self.ptr + len(items)
        self.queue[self.ptr: min(new_ptr, self.size)] = items[:min(new_ptr, self.size) - self.ptr]
        self.queue[:new_ptr - min(new_ptr, self.size)] = items[min(new_ptr, self.size) - self.ptr:]
        self.cur_size = min(self.cur_size + len(items), self.size)
        self.ptr = new_ptr % self.size

    def get_item(self, n):
        if self.cur_size == 0:
            return []
        n = min(n, self.cur_size)
        items = random.sample(self.queue[:self.cur_size], n)
        self.got += n
        return items
    
    def clear(self):
        """Clear queue to free memory"""
        self.queue = [None] * self.size
        self.ptr = 0
        self.cur_size = 0


class SplitSampler(object):
    def __init__(self, cfg):
        self.total_size = cfg.size
        self.num_c = cfg.num_class

    def init_finish(self):
        return hasattr(self, 'class_ratio')

    def init_class_ratio(self, config):
        self.tail_class_idx = config.tail_class_idx
        self.class_ratio = config.class_ratio
        self.tail_class_ratio = self.class_ratio[self.tail_class_idx]
        self.tail_class_ratio /= self.tail_class_ratio.sum()
        self.queues = []
        self.init_queue()


    def update_cfg(self, cfg):
        cfg.class_ratio = self.class_ratio
        cfg.class_thres = np.ones_like(cfg.class_ratio)
        cfg.class_thres[self.tail_class_idx] = self.class_ratio[self.tail_class_idx]
        cfg.tail_class_idx = self.tail_class_idx

    def init_queue(self):
        for c in range(self.num_c):
            size = max(1, int(self.total_size * self.tail_class_ratio[c]))
            self.queues.append(Queue(size))

    def update(self, items):
        if not self.init_finish():
            raise ValueError('Split sampler is not inited!')
        assert len(items) == self.num_c
        for c in range(self.num_c):
            self.queues[c].update_queue(items[c])

    def get_split(self, n):
        if not self.init_finish():
            raise ValueError('Split sampler is not inited!')
        if n == 0:
            return []
        item_c = np.random.choice(self.num_c, n, p=self.tail_class_ratio)
        items = []
        for c in item_c:
            items.extend(self.queues[c].get_item(1))
        return items

    def update_class_ratio(self, class_ratio):
        if class_ratio.max() > 0.0:
            class_ratio = class_ratio.numpy()
            inverse_class_ratio = 1.0 / (class_ratio + 10e-1)
            inverse_class_ratio /= inverse_class_ratio.sum()
            self.tail_class_ratio = 0.999 * self.tail_class_ratio + 0.001 * inverse_class_ratio

    def load_sampler(self, path):
        buffer = torch.load(path)
        self.queues = buffer['queues']
        self.class_ratio = buffer['class_ratio']
        self.inverse_class_ratio = buffer['inverse_class_ratio']
        self.tail_class_ratio = buffer['tail_class_ratio']
        self.tail_class_idx = buffer['tail_class_idx']

    def save_sampler(self, path):
        torch.save(
            {'queues': self.queues, 'class_ratio': self.class_ratio, 'inverse_class_ratio': self.inverse_class_ratio,
             'tail_class_ratio': self.tail_class_ratio, 'tail_class_idx': self.tail_class_idx}, path
        )
    
    def clear_queues(self):
        """Clear all queues to free memory - useful for periodic cleanup"""
        if hasattr(self, 'queues'):
            for queue in self.queues:
                queue.clear()


class DODATACMDataset(Dataset):
    """DODA-style TACM dataset - perfect replication of DODA implementation"""
    
    def __init__(self, source_dataset, target_dataset,  
        voxel_size,
        voxel_max,
        class_names,
        params,
        transform,
        shuffle_index):

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.shuffle_index = shuffle_index

        self.params = params
        self.transform = transform
        
        # Compute dataset size
        self.source_size = len(source_dataset)
        self.target_size = len(target_dataset)
        
        # Remove unused source_indices variable
        self.class_names = class_names
        self.split_sampler = SplitSampler(self.params.cuboid_queue)
        
    def __len__(self):
        # return 500
        return min(self.source_size, self.target_size)
        
    def __getitem__(self, idx):

        idx1 = random.randint(0, self.target_size - 1)
        idx2 = random.randint(0, self.source_size - 1)

        # Get source data (corresponds to DODA's dataset2, i.e., pc2)
        points2, labels2 = self.source_dataset[idx2]
        
        # Get target data (corresponds to DODA's dataset1, i.e., pc1)
        points1, _ = self.target_dataset[idx1]
        labels1 = self.target_dataset.get_room_pseudo(idx1)
        
        # Perform DODA-style TACM mixing (room level)
        # Parameter order: pc1 (target), pc2 (source)
        mixed_points, mixed_labels, mix_info = tacm(
            self.params, self.split_sampler, self.class_names, (points1, labels1), (points2, labels2)
        )

        coord, label = data_prepare(mixed_points, mixed_labels, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

        # Add queue update information
        mix_info['tar_tail_splits'] = mix_info.get('tar_tail_splits', [])
        mix_info['tar_splits_class_ratio'] = mix_info.get('tar_splits_class_ratio', np.zeros(3))

        return coord, label, mix_info


