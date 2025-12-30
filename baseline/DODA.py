"""
Author: Benny
Date: Nov 2019
Self-training semantic segmentation script following DODA implementation
"""
import argparse
import os
import yaml
import gc
from easydict import EasyDict
from data_utils.S3DISDataLoader import S3DISDatasetTrans, ScannetDatasetWholeScene
from utils.pseudo_label_utils import generate_and_filter_pseudo_labels
from data_utils.mix_dataset import DODATACMDataset
from data_utils.data_util import collate_fn, collate_fn_mix
import torch
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import glob
from data_utils import transform as t
from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy

# Set GPU memory fraction
torch.cuda.set_per_process_memory_fraction(0.9, 0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']

class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def create_label_mapping(exclude_classes):
    """
    创建标签映射，将被排除的类别映射为-1，其他类别重新映射为连续索引
    Args:
        exclude_classes: 要排除的类别名称列表，例如 ['board', 'clutter']
    Returns:
        label_mapping: 原始标签到新标签的映射数组
        new_classes: 保留的类别列表
        new_class2label: 新类别到新标签的映射
        new_seg_label_to_cat: 新标签到类别的映射
    """
    if exclude_classes is None or len(exclude_classes) == 0:
        # 不排除任何类别，返回原始映射
        label_mapping = np.arange(len(classes))
        return label_mapping, classes, class2label, seg_label_to_cat
    
    # 获取要排除的类别索引
    exclude_indices = [class2label[cls] for cls in exclude_classes if cls in class2label]
    
    # 创建保留的类别列表
    new_classes = [cls for cls in classes if cls not in exclude_classes]
    
    # 创建新的类别到标签映射
    new_class2label = {cls: i for i, cls in enumerate(new_classes)}
    
    # 创建新标签到类别的映射
    new_seg_label_to_cat = {}
    for i, cat in enumerate(new_classes):
        new_seg_label_to_cat[i] = cat
    
    # 创建标签映射数组：原始标签 -> 新标签（-1表示被排除）
    label_mapping = np.full(len(classes), -1, dtype=np.int64)
    new_idx = 0
    for orig_idx, cls in enumerate(classes):
        if cls not in exclude_classes:
            label_mapping[orig_idx] = new_idx
            new_idx += 1
    
    return label_mapping, new_classes, new_class2label, new_seg_label_to_cat


class LabelMappingDataset:
    """包装数据集，应用标签映射并过滤被排除的类别（用于return_room=False的数据集）"""
    def __init__(self, dataset, label_mapping):
        self.dataset = dataset
        self.label_mapping = torch.from_numpy(label_mapping).long()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        coord, label = self.dataset[idx]
        # 应用标签映射
        label_mapped = self.label_mapping[label]
        # 创建掩码：保留标签不为-1的点
        mask = label_mapped >= 0
        coord_filtered = coord[mask]
        label_filtered = label_mapped[mask]
        return coord_filtered, label_filtered


class RoomLabelMappingDataset:
    """包装return_room=True的数据集，应用标签映射到房间级别的标签"""
    def __init__(self, dataset, label_mapping, new_classes=None):
        self.dataset = dataset
        # 确保label_mapping是正确的numpy数组类型
        if not isinstance(label_mapping, np.ndarray):
            label_mapping = np.array(label_mapping, dtype=np.int64)
        else:
            label_mapping = label_mapping.astype(np.int64)
        self.label_mapping = label_mapping  # numpy array
        self.new_classes = new_classes  # 保留的类别列表
        self._mapped_weights = None  # 缓存的映射权重
        
        # 为伪标签生成创建兼容的属性
        # 如果原始数据集没有这些属性，我们基于room_points和room_labels创建
        if not hasattr(self.dataset, 'scene_points_list'):
            if hasattr(self.dataset, 'room_points') and hasattr(self.dataset, 'room_labels'):
                # 创建scene_points_list和semantic_labels_list用于伪标签生成
                self._scene_points_list = []
                self._semantic_labels_list = []
                # 直接访问原始数据集的内部属性，避免递归调用
                for i in range(len(self.dataset.room_points)):
                    if self.dataset.cache:
                        points = self.dataset.room_points[i]
                        labels = self.dataset.room_labels[i]
                    else:
                        # 如果数据没有缓存，需要从文件加载
                        room_path = os.path.join(self.dataset.data_root, self.dataset.rooms_split[i])
                        room_data = np.load(room_path)
                        points = room_data[:, 0:3]
                        labels = room_data[:, 6]
                    # 确保labels是整数类型
                    if not isinstance(labels, np.ndarray):
                        labels = np.array(labels)
                    labels = labels.astype(np.int64)
                    # 应用标签映射
                    labels_mapped = self.label_mapping[labels]
                    # 保留所有点，只映射标签（用于伪标签生成时保持维度一致）
                    self._scene_points_list.append(points)
                    self._semantic_labels_list.append(labels_mapped)
            else:
                # 如果没有room_points，创建一个空列表
                self._scene_points_list = []
                self._semantic_labels_list = []
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        points_all, label_all = self.dataset[idx]
        # 确保label_all是整数类型的numpy数组
        if not isinstance(label_all, np.ndarray):
            label_all = np.array(label_all)
        label_all = label_all.astype(np.int64)
        
        # 应用标签映射到numpy数组
        # 注意：对于TACM混合，我们需要保留所有点，只映射标签
        # 被排除的类别会映射为-1，这些点在后续的loss计算中会被忽略
        label_mapped = self.label_mapping[label_all]
        
        # 不在这里过滤点，保留所有点以便TACM混合操作
        # 过滤会在后续的loss计算中通过忽略标签为-1的点来实现
        return points_all, label_mapped
    
    def get_room_pseudo(self, idx):
        """获取伪标签（伪标签已经在新的标签空间中，不需要映射）"""
        return self.dataset.get_room_pseudo(idx)
    
    def set_room_pseudo(self, pseudo_list):
        """设置伪标签（伪标签已经在新的标签空间中）"""
        return self.dataset.set_room_pseudo(pseudo_list)
    
    def get_room_points(self, idx):
        """获取房间点云"""
        return self.dataset.get_room_points(idx)
    
    def __getattr__(self, name):
        """代理所有未定义的属性到原始数据集（作为后备）"""
        return getattr(self.dataset, name)
    
    @property
    def file_list(self):
        """获取文件列表"""
        return self.dataset.file_list
    
    @property
    def rooms_split(self):
        """获取房间分割列表（用于伪标签生成）"""
        return self.dataset.rooms_split
    
    @property
    def scene_points_list(self):
        """获取场景点云列表（用于伪标签生成）"""
        if hasattr(self, '_scene_points_list'):
            return self._scene_points_list
        elif hasattr(self.dataset, 'scene_points_list'):
            # 如果原始数据集有这些属性，直接返回（但需要应用标签映射）
            return self.dataset.scene_points_list
        else:
            return []
    
    @property
    def semantic_labels_list(self):
        """获取语义标签列表（用于伪标签生成，需要应用标签映射）"""
        if hasattr(self, '_semantic_labels_list'):
            return self._semantic_labels_list
        elif hasattr(self.dataset, 'semantic_labels_list'):
            # 如果原始数据集有这些属性，需要应用标签映射
            mapped_labels = []
            for labels in self.dataset.semantic_labels_list:
                # 确保labels是整数类型
                if not isinstance(labels, np.ndarray):
                    labels = np.array(labels)
                labels = labels.astype(np.int64)
                labels_mapped = self.label_mapping[labels]
                mapped_labels.append(labels_mapped)
            return mapped_labels
        else:
            return []
    
    @property
    def labelweights(self):
        """获取标签权重（需要映射）"""
        if self._mapped_weights is None and self.new_classes is not None:
            original_weights = self.dataset.labelweights
            keep_indices = [class2label[cls] for cls in self.new_classes]
            self._mapped_weights = original_weights[keep_indices]
        elif self._mapped_weights is None:
            self._mapped_weights = self.dataset.labelweights
        return self._mapped_weights


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_att', help='model name [default: pointnet2_sem_seg_att]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer [default: Adam]')
    parser.add_argument('--npoint', type=int, default=40000, help='Point Number [default: 4096]')
    parser.add_argument('--voxel_size', type=float, default=0.04, help='Voxel size [default: 0.04]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Learning rate decay [default: 0.9]')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for learning rate decay [default: 15]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_att_baseline', help='Log path [default: pointnet2_att]')
    parser.add_argument('--n_epochs', type=int, default=60, help='Total number of epochs [default: 60]')
    parser.add_argument('--max_checkpoints', type=int, default=10, help='Maximum number of checkpoints to keep [default: 10]')
    
    # Training strategy parameters
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency')
    parser.add_argument('--loss_weight_src', type=float, default=1.0, help='Loss weight for source data')
    parser.add_argument('--loss_weight_tar', type=float, default=1.0, help='Loss weight for target data')
    
    # Checkpoint management
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--preserve_pseudo_labels', action='store_true', default=False, help='Preserve pseudo labels after training')
    
    # TACM configuration
    parser.add_argument('--data_config', type=str, default='cfgs/bim_config.yaml', help='Path to TACM YAML config file (default: cfgs/bim_config.yaml)')

    # Optimization/stability
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'], help='LR scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Warmup epochs for cosine scheduler')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--use_ema', action='store_true', default=False, help='Enable EMA for model weights')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay factor')
    parser.add_argument('--freeze_bn', action='store_true', default=False, help='Freeze BatchNorm stats/affine during training')
    
    # Data loading strategy
    parser.add_argument('--cache_data', action='store_true', default=False, help='Cache all data in memory (faster but uses more RAM). If not set, data will be loaded from disk on-the-fly.')
    parser.add_argument('--exclude_classes', type=str, nargs='+', default=['board', 'clutter'], 
                        help='Classes to exclude from training, e.g., --exclude_classes board clutter [default: board clutter]')
    
    args = parser.parse_args()
    
    return args

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


# Helper: load latest available pseudo labels up to a given epoch
def load_latest_pseudo_labels(pseudo_labels_dir: Path, upto_epoch: int):
    for e in range(upto_epoch, -1, -1):
        path = pseudo_labels_dir / f'pseudo_labels_epoch_{e}.npy'
        if path.exists():
            return np.load(path, allow_pickle=True)
    return None


def update_split_sampler(train_loader, tar_tail_splits, tar_splits_class_ratio, num_c):
    """更新立方体队列中的split sampler"""
    if not hasattr(train_loader.dataset, 'split_sampler'):
        return
    
    split_sampler = train_loader.dataset.split_sampler
    
    # 展平tail_splits
    def flatten(l):
        return [jj for ii in l for jj in ii]
    
    tar_tail_splits_total = [flatten(tar_tail_splits[i::num_c]) for i in range(num_c)]
    
    # 更新sampler
    if split_sampler.init_finish():
        split_sampler.update(tar_tail_splits_total)
        
        # 更新类别比例
        if hasattr(split_sampler, 'update_class_ratio'):
            split_sampler.update_class_ratio(tar_splits_class_ratio)


def load_config(config_path=None):
    """Load configuration from YAML file and return as EasyDict (recursive)."""

    def _to_easydict(obj):
        if isinstance(obj, dict):
            return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_easydict(v) for v in obj]
        return obj

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    config = _to_easydict(cfg_dict)
    print(f"Loaded TACM config from {config_path}")
    return config
 

def evaluate_model(model, data_loader, criterion, weights, desc="Evaluation", num_classes=13):
    """Enhanced evaluation function with detailed metrics"""
    model.eval()
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(num_classes)]
    total_correct_class = [0 for _ in range(num_classes)]
    total_iou_deno_class = [0 for _ in range(num_classes)]
    
    # Additional metrics for monitoring
    class_names = classes
    class_iou = [0.0] * num_classes
    class_acc = [0.0] * num_classes
    confidence_sum = 0.0
    confidence_count = 0
    
    with torch.no_grad():
        for i, (coord, target, offset) in enumerate(tqdm(data_loader, desc=desc)):
            coord, target, offset = coord.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)

            with autocast(enabled=True, dtype=torch.float16):
                seg_pred, trans_feat = model([coord, coord, offset])
            # In collate_fn format, seg_pred is already 2D [total_points, num_classes]
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred  # Already in correct shape

            batch_label = target.cpu().data.numpy()
            target = target  # Already in correct shape
            
            # Calculate loss using model's criterion
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss_sum += loss.item()
            
            # Get confidence scores (compute mean directly to save memory)
            probs = torch.softmax(seg_pred, dim=-1)
            conf_scores, _ = torch.max(probs, dim=-1)
            confidence_sum += conf_scores.sum().item()
            confidence_count += conf_scores.numel()
            del probs, conf_scores  # Explicit cleanup
            
            pred_val = np.argmax(pred_val, 1)
            
            # Only count non-ignore labels for evaluation
            valid_mask = (batch_label != -1)
            if valid_mask.any():
                correct = np.sum((pred_val[valid_mask] == batch_label[valid_mask]))
                total_correct += correct
                total_seen += np.sum(valid_mask)

                for l in range(num_classes):
                    # Only count valid labels
                    valid_class_mask = (batch_label == l) & valid_mask
                    total_seen_class[l] += np.sum(valid_class_mask)
                    total_correct_class[l] += np.sum((pred_val == l) & valid_class_mask)
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) & valid_mask)
            
            # Clean up tensors
            del seg_pred, trans_feat, coord, target, offset
            del pred_val, batch_label

    # Calculate detailed metrics
    for l in range(num_classes):
        if total_seen_class[l] > 0:
            class_acc[l] = total_correct_class[l] / float(total_seen_class[l])
        if total_iou_deno_class[l] > 0:
            class_iou[l] = total_correct_class[l] / float(total_iou_deno_class[l])

    mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
    accuracy = total_correct / float(total_seen) if total_seen > 0 else 0.0
    mean_loss = loss_sum / float(len(data_loader))
    avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
    
    # Clear cache after evaluation
    torch.cuda.empty_cache()
    
    return mIoU, accuracy, mean_loss, class_iou, class_acc, avg_confidence


def train_epoch(loader, src_trainDataLoader, model, optimizer, epoch, args, config, logger, criterion, weights, scaler):
    model.train()
    total_batches = 0
    sum_total_loss = 0.0
    sum_source_loss = 0.0
    sum_target_loss = 0.0
    
    # Use simple iterator instead of cycle to avoid memory accumulation
    src_iter = iter(src_trainDataLoader)
    
    loop = tqdm(enumerate(loader), total=len(loader), leave=True, smoothing=0.9)
    loop.set_description(f'Epoch {epoch+1} (Mixed + Source)')
    
    for i, (coord, labels, offset, mix_infos) in loop:
        
        # Only zero grad at the start of accumulation window
        if i % max(1, args.accum_steps) == 0:
            optimizer.zero_grad(set_to_none=True)
        
        # source forward - restart iterator if exhausted
        try:
            source_coord, source_labels, source_offset = next(src_iter)
        except StopIteration:
            src_iter = iter(src_trainDataLoader)
            source_coord, source_labels, source_offset = next(src_iter)
        source_coord, source_labels, source_offset = source_coord.cuda(non_blocking=True), source_labels.cuda(non_blocking=True), source_offset.cuda(non_blocking=True)
        with autocast(enabled=True, dtype=torch.float16):
            source_logits, source_trans_feat = model([source_coord, source_coord, source_offset])
            source_loss = criterion(source_logits, source_labels, source_trans_feat, weights)
            source_loss = source_loss * args.loss_weight_src
        # log value before backward to avoid holding graph references
        source_loss_val = float(source_loss.item())
        # scale by accumulation steps to keep effective loss magnitude
        scaler.scale(source_loss / max(1, args.accum_steps)).backward()
        # free source tensors references ASAP
        del source_logits, source_trans_feat, source_loss
        del source_coord, source_labels, source_offset

        # target forward - use same criterion as source for consistency
        coord, labels, offset = coord.cuda(non_blocking=True), labels.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        with autocast(enabled=True, dtype=torch.float16):
            logits, trans_feat = model([coord, coord, offset])
            # Use consistent criterion with class weights like source
            loss = criterion(logits, labels, trans_feat, weights)
            loss = loss * args.loss_weight_tar
        target_loss_val = float(loss.item())
        scaler.scale(loss / max(1, args.accum_steps)).backward()
        
        # free target refs
        del logits, trans_feat, loss, coord, labels, offset

        # step optimizer through scaler according to gradient accumulation
        if ((i + 1) % max(1, args.accum_steps)) == 0:
            # unscale and clip if enabled
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        # update split queue
        split_sampler_cfg = config.tacm.cuboid_queue
        if split_sampler_cfg.enabled:
            update_split_sampler(
                loader, mix_infos['tar_tail_splits'], mix_infos['tar_splits_class_ratio'],
                split_sampler_cfg.num_class
            )
        
        # Clear mix_infos to free memory
        del mix_infos
        
        # More frequent cache clearing to combat fragmentation
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
        
        # Collect metrics
        total_loss = source_loss_val + target_loss_val
        sum_source_loss += source_loss_val
        sum_target_loss += target_loss_val
        sum_total_loss += total_loss
        total_batches += 1

        loop.set_postfix(
            total=f'{total_loss:.4f}',
            src=f'{source_loss_val:.4f}',
            tar=f'{target_loss_val:.4f}',
        )
    
    avg_total = sum_total_loss / max(1, total_batches)
    avg_src = sum_source_loss / max(1, total_batches)
    avg_tar = sum_target_loss / max(1, total_batches)
    
    # Clear cache after epoch
    torch.cuda.empty_cache()
    
    return avg_total, avg_src, avg_tar, total_batches


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load full configuration for pseudo label parameters
    config = load_config(args.data_config)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    base_log_root = Path('./log/')
    base_log_root.mkdir(exist_ok=True)
    sem_seg_root = base_log_root.joinpath('sem_seg')
    sem_seg_root.mkdir(exist_ok=True)

    # Determine base name and experiment directory
    base_name = timestr if args.log_dir is None else args.log_dir
    train_stage_dir = sem_seg_root.joinpath(base_name)  # source (train) stage directory
    experiment_dir = sem_seg_root.joinpath(f"{base_name}_st")  # current (st) stage directory

    # Create directories
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    pseudo_labels_dir = experiment_dir.joinpath('pseudo_labels/')
    pseudo_labels_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    src_train_root = 'data/bim_indoor3d/'
    tar_train_root = 'data/stanford_indoor3d/'
    test_root = 'data/bim_test/'
    
    # 创建标签映射
    exclude_classes = args.exclude_classes if args.exclude_classes else []
    label_mapping, new_classes, new_class2label, new_seg_label_to_cat = create_label_mapping(exclude_classes)
    
    # 更新全局变量以使用新的类别映射
    global seg_label_to_cat
    seg_label_to_cat = new_seg_label_to_cat
    
    NUM_CLASSES = len(new_classes)
    NUM_POINT = args.npoint if args.npoint > 0 else -1  # None means unlimited points
    BATCH_SIZE = args.batch_size
    
    if exclude_classes:
        log_string(f"Excluding classes: {exclude_classes}")
        log_string(f"Training with {NUM_CLASSES} classes: {new_classes}")
    else:
        log_string(f"Training with all {NUM_CLASSES} classes")

    print("start loading training data ...")

    train_transform = t.Compose([
        # t.RandomScale([0.9, 1.1], anisotropic=True),
        t.RandomRotate(),
        # t.RandomShift([0.2, 0.2, 0.0]),
        # t.RandomFlip(p=0.5),
        # t.RandomJitter(sigma=0.01, clip=0.05)
    ])
    
    # Source/Target datasets: return whole rooms for room-level TACM
    log_string(f"Data loading mode: {'cache in memory' if args.cache_data else 'load from disk on-the-fly'}")
    src_TRAIN_DATASET_RAW = S3DISDatasetTrans(split='train', data_root=src_train_root, num_point=NUM_POINT, test_area=args.test_area, sample_rate=1.0, shuffle_index=False, transform=None, return_room=True, cache=args.cache_data)
    tar_TRAIN_DATASET_RAW = S3DISDatasetTrans(split='train', data_root=tar_train_root, num_point=NUM_POINT, test_area=args.test_area, sample_rate=1.0, shuffle_index=False, transform=None, return_room=True, cache=args.cache_data)
    # Apply label mapping to room-level datasets for TACM
    src_TRAIN_DATASET = RoomLabelMappingDataset(src_TRAIN_DATASET_RAW, label_mapping, new_classes)
    tar_TRAIN_DATASET = RoomLabelMappingDataset(tar_TRAIN_DATASET_RAW, label_mapping, new_classes)
    
    # Test dataset keeps voxelized stream for evaluation - apply label mapping
    TEST_DATASET_RAW = S3DISDatasetTrans(split='test', data_root=test_root, num_point=NUM_POINT, test_area=args.test_area, sample_rate=1.0, shuffle_index=False, transform=None, cache=args.cache_data)
    TEST_DATASET = LabelMappingDataset(TEST_DATASET_RAW, label_mapping)

    # Data loaders for evaluation only; mixing dataset provides its own collate
    def worker_init_fn(worker_id):
        np.random.seed(worker_id + int(time.time()))
    
    log_string(f"Source dataset rooms: {len(src_TRAIN_DATASET)}")
    log_string(f"Target dataset rooms: {len(tar_TRAIN_DATASET)}")
    
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,  # Reduced for stability
        pin_memory=False, 
        drop_last=False,  # Don't drop last for evaluation
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )
    # Get label weights for loss calculation
    # RoomLabelMappingDataset automatically handles weight mapping
    weights = torch.Tensor(src_TRAIN_DATASET.labelweights).cuda()

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    
    # Model loading with automatic pretrained weight detection
    start_epoch = 0
    best_mIoU = 0.0
    
    if args.resume:
        # Resume from specified checkpoint
        try:
            checkpoint = torch.load(args.resume)
            # Use strict=False to handle checkpoints with extra/missing keys
            missing_keys, unexpected_keys = classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                log_string(f'Warning: Missing keys in checkpoint: {len(missing_keys)} keys')
            if unexpected_keys:
                log_string(f'Warning: Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys')
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_mIoU = checkpoint.get('class_avg_iou', 0.0)
            log_string(f'Resumed from checkpoint: {args.resume}, epoch: {start_epoch}, best mIoU: {best_mIoU}')
        except Exception as e:
            log_string(f'Failed to resume from checkpoint ({e}). Starting from scratch...')
            classifier = classifier.apply(weights_init)
    else:
        # Try to load from train stage directory automatically
        try:
            # Look for best model in train stage directory
            src_best_path = str(train_stage_dir.joinpath('checkpoints/best_model.pth'))
            if os.path.exists(src_best_path):
                checkpoint = torch.load(src_best_path)
                # Use strict=False to handle checkpoints with extra/missing keys (e.g., from GAN training)
                missing_keys, unexpected_keys = classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing_keys:
                    log_string(f'Warning: Missing keys in checkpoint: {len(missing_keys)} keys')
                if unexpected_keys:
                    log_string(f'Warning: Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys')
                log_string(f'Loaded pretrained weights from train stage: {src_best_path}')
                log_string('Starting self-training from epoch 0 with pretrained weights.')
            else:
                # Look for any checkpoint in train stage directory
                train_checkpoints = glob.glob(str(train_stage_dir.joinpath('checkpoints/*.pth')))
                if train_checkpoints:
                    # Sort by modification time and get the latest
                    train_checkpoints.sort(key=os.path.getmtime, reverse=True)
                    latest_checkpoint = train_checkpoints[0]
                    checkpoint = torch.load(latest_checkpoint)
                    # Use strict=False to handle checkpoints with extra/missing keys
                    missing_keys, unexpected_keys = classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    if missing_keys:
                        log_string(f'Warning: Missing keys in checkpoint: {len(missing_keys)} keys')
                    if unexpected_keys:
                        log_string(f'Warning: Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys')
                    log_string(f'Loaded pretrained weights from: {latest_checkpoint}')
                    log_string('Starting self-training from epoch 0 with pretrained weights.')
                else:
                    raise FileNotFoundError("No pretrained weights found")
        except Exception as e:
            log_string(f'No existing train-stage model found or failed to load ({e}). Starting ST from scratch...')
            classifier = classifier.apply(weights_init)
    
    # Initial evaluation with enhanced metrics
    log_string('=' * 60)
    log_string('INITIAL BASELINE EVALUATION')
    log_string('=' * 60)
    
    initial_mIoU, initial_acc, initial_loss, class_iou, class_acc, avg_confidence = evaluate_model(classifier, testDataLoader, criterion, weights, "Initial Baseline", num_classes=NUM_CLASSES)
    log_string(f'Initial mIoU: {initial_mIoU:.4f}')
    log_string(f'Initial Accuracy: {initial_acc:.4f}')
    log_string(f'Initial Loss: {initial_loss:.4f}')
    log_string(f'Initial Avg Confidence: {avg_confidence:.4f}')
    
    # Log per-class performance
    log_string('Per-class IoU:')
    for i, (iou, acc) in enumerate(zip(class_iou, class_acc)):
        log_string(f'  {new_classes[i]}: IoU={iou:.4f}, Acc={acc:.4f}')
    
    best_mIoU = initial_mIoU
    log_string(f'Baseline mIoU set to: {best_mIoU:.4f}')
    log_string('=' * 60)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=args.learning_rate, 
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)

    # Optional EMA model for stabler eval
    ema_classifier = None
    if args.use_ema:
        ema_classifier = deepcopy(classifier).cuda()
        for p in ema_classifier.parameters():
            p.requires_grad_(False)

    def update_ema(ema_model, model, decay):
        if ema_model is None:
            return
        with torch.no_grad():
            msd = model.state_dict()
            for name, p_ema in ema_model.state_dict().items():
                if name in msd:
                    p = msd[name].detach()
                    p_ema.copy_(p_ema * decay + p * (1.0 - decay))

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = 10

    # Scheduler: cosine with warmup (default) or original step decay
    scheduler = None
    if args.scheduler == 'cosine':
        # T_max excludes warmup epochs; guard for small n_epochs
        t_max = max(1, args.n_epochs - max(0, args.warmup_epochs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=LEARNING_RATE_CLIP)

    MIXED_DATASET = DODATACMDataset(
        src_TRAIN_DATASET,
        tar_TRAIN_DATASET,
        voxel_size=args.voxel_size,
        voxel_max=NUM_POINT,
        class_names=new_classes,  # Use mapped classes for training
        params=config['tacm'],
        transform=train_transform,
        shuffle_index=True
    )
    
    mixed_loader = torch.utils.data.DataLoader(
        MIXED_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Reduced from 10 to prevent worker memory accumulation
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_mix,
        persistent_workers=False  # Force workers to restart each epoch to prevent memory leaks
    )

    src_TRAIN_DATASET_TRAIN_RAW = S3DISDatasetTrans(split='train', data_root=src_train_root, num_point=NUM_POINT, test_area=args.test_area, sample_rate=1.0, shuffle_index=False, transform=None, return_room=False, cache=args.cache_data)
    src_TRAIN_DATASET_TRAIN = LabelMappingDataset(src_TRAIN_DATASET_TRAIN_RAW, label_mapping)

    src_trainDataLoader = torch.utils.data.DataLoader(
        src_TRAIN_DATASET_TRAIN,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    # AMP scaler
    scaler = GradScaler(enabled=True)

    for epoch in range(start_epoch, args.n_epochs):
        log_string('**** Epoch %d (%d/%s) ****' % (epoch + 1, epoch + 1, args.n_epochs))
        # Learning rate update
        if args.scheduler == 'cosine':
            if epoch < max(0, args.warmup_epochs):
                warm_lr = args.learning_rate * float(epoch + 1) / float(max(1, args.warmup_epochs))
                for g in optimizer.param_groups:
                    g['lr'] = warm_lr
                log_string('Learning rate (warmup):%f' % warm_lr)
            else:
                # scheduler.step() is called at end of previous epoch; log current lr
                current_lr = optimizer.param_groups[0]['lr']
                log_string('Learning rate (cosine):%f' % current_lr)
        else:
            lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            log_string('Learning rate (step):%f' % lr)
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        # Optional: freeze BN for stability
        if args.freeze_bn:
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters(recurse=True):
                        p.requires_grad = False
        
        if epoch % config['pseudo_labels'].get('frequency') == 0:
            log_string('Generating pseudo labels for epoch %d (room-level voting)' % epoch)
            
            # Clear old pseudo labels explicitly before generating new ones
            # Access the underlying dataset's room_pseudo attribute
            if isinstance(tar_TRAIN_DATASET, RoomLabelMappingDataset):
                underlying_dataset = tar_TRAIN_DATASET.dataset
            else:
                underlying_dataset = tar_TRAIN_DATASET
            
            if hasattr(underlying_dataset, 'room_pseudo'):
                del underlying_dataset.room_pseudo
                underlying_dataset.room_pseudo = None
            
            # Clear cache before pseudo label generation
            torch.cuda.empty_cache()
            
            gen_pseudo_labels, class_ratio, tail_class_indices = generate_and_filter_pseudo_labels(
                model=classifier,
                target_dataset=tar_TRAIN_DATASET,
                epoch=epoch,
                args=args,
                num_classes=NUM_CLASSES,
                num_point=NUM_POINT,
                voxel_size=args.voxel_size,
                config=config
            )

            if config['tacm']['cuboid_queue']['enabled']:
                config.tacm.cuboid_queue.class_ratio = class_ratio
                config.tacm.cuboid_queue.tail_class_idx = tail_class_indices
                if not MIXED_DATASET.split_sampler.init_finish():
                    MIXED_DATASET.split_sampler.init_class_ratio(config.tacm.cuboid_queue)

            tar_TRAIN_DATASET.set_room_pseudo(gen_pseudo_labels)
            
            # Clear references to free memory
            del gen_pseudo_labels, class_ratio, tail_class_indices
            torch.cuda.empty_cache()
            
        avg_total_loss, avg_src_loss, avg_tar_loss, num_batches = train_epoch(
            mixed_loader, src_trainDataLoader, classifier, optimizer, epoch, args, config, logger, criterion, weights, scaler
        )
        # Update EMA after each epoch (already updated every step via optimizer step cadence)
        update_ema(ema_classifier, classifier, args.ema_decay)
        
        # Force garbage collection after training epoch to release memory
        gc.collect()
        
        # Periodically clear cuboid queues to prevent memory accumulation (every 5 epochs)
        if config['tacm']['cuboid_queue']['enabled'] and (epoch + 1) % 5 == 0:
            log_string('Clearing cuboid queues to free memory...')
            MIXED_DATASET.split_sampler.clear_queues()
            torch.cuda.empty_cache()

        # Advance scheduler (cosine) after warmup epochs
        if args.scheduler == 'cosine' and epoch >= max(0, args.warmup_epochs):
            scheduler.step()
        log_string('Mixed Training Results:')
        log_string(f'  Total loss: {avg_total_loss:.4f}')
        log_string(f'  Source loss: {avg_src_loss:.4f}')
        log_string(f'  Target loss: {avg_tar_loss:.4f}')
        log_string(f'  Mix Ratio: {config["tacm"]["mix_ratio"]:.4f}')

        # Save checkpoint
        if epoch % 3 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + f'/model_epoch_{epoch}.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mIoU': best_mIoU,
            }
            torch.save(state, savepath)
            log_string('Saving model....')
            
            # Clean up old checkpoints
            if epoch > 10:
                old_checkpoint = str(checkpoints_dir) + f'/model_epoch_{epoch-9}.pth'
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

        # Evaluation with enhanced metrics
        if epoch % args.eval_freq == 0 or epoch == args.n_epochs - 1:
            log_string('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            eval_model = ema_classifier if (ema_classifier is not None) else classifier
            mIoU, accuracy, mean_loss, class_iou, class_acc, avg_confidence = evaluate_model(eval_model, testDataLoader, criterion, weights, f"Epoch {epoch+1}", num_classes=NUM_CLASSES)
            
            log_string('=' * 60)
            log_string(f'EPOCH {epoch + 1} EVALUATION RESULTS')
            log_string(f'eval mean loss: {mean_loss:.4f}')
            log_string(f'eval point avg class IoU: {mIoU:.4f}')
            log_string(f'eval point accuracy: {accuracy:.4f}')
            log_string(f'eval avg confidence: {avg_confidence:.4f}')
            
            # Log per-class performance
            log_string('Per-class IoU:')
            for i, (iou, acc) in enumerate(zip(class_iou, class_acc)):
                log_string(f'  {new_classes[i]}: IoU={iou:.4f}, Acc={acc:.4f}')
            log_string('=' * 60)

            # Save best model
            if mIoU >= best_mIoU:
                best_mIoU = mIoU
                logger.info('Save best model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving best model at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': (eval_model.state_dict() if eval_model is not None else classifier.state_dict()),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'baseline_mIoU': initial_mIoU,
                    'class_iou': class_iou,
                    'class_acc': class_acc,
                }
                torch.save(state, savepath)
                log_string('Best model saved!')
                
            else:
                log_string(f'No improvement. Current mIoU: {mIoU:.4f}, Best mIoU: {best_mIoU:.4f}')
            
            log_string(f'Best mIoU so far: {best_mIoU:.4f} (Baseline: {initial_mIoU:.4f})')
            log_string(f'Total improvement: {best_mIoU - initial_mIoU:.4f} ({((best_mIoU - initial_mIoU) / (initial_mIoU + 1e-10) * 100):.2f}%)')
        else:
            log_string(f'Skipping evaluation (eval_freq={args.eval_freq})')
        
        # Clear GPU memory after each epoch
        torch.cuda.empty_cache()

    # Clean up pseudo labels if not preserving
    if not args.preserve_pseudo_labels:
        if pseudo_labels_dir.exists():
            shutil.rmtree(pseudo_labels_dir)
            log_string('Cleaned up pseudo labels directory')
    
    # Final summary
    log_string('=' * 80)
    log_string('TRAINING COMPLETED!')
    log_string(f'Initial baseline mIoU: {initial_mIoU:.4f}')
    log_string(f'Final best mIoU: {best_mIoU:.4f}')
    log_string(f'Improvement: {best_mIoU - initial_mIoU:.4f} ({((best_mIoU - initial_mIoU) / initial_mIoU * 100):.2f}%)')
    log_string(f'Total epochs: {args.n_epochs}')
    log_string(f'Best model saved at: {checkpoints_dir}/best_model.pth')
    log_string('=' * 80)


if __name__ == '__main__':
    args = parse_args()
    main(args)