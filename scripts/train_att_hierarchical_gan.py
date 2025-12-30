"""
Train attention model (point transformer) with hierarchical GAN-based domain adaptation.
Combines the strong feature extraction of attention mechanisms with multi-level adversarial 
training for improved generalization on target domain point clouds.
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDatasetTrans
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
from data_utils import transform as t
from data_utils.data_util import collate_fn
import gc
import copy

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
    """包装数据集，应用标签映射并过滤被排除的类别"""
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


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('Attention Model with Hierarchical GAN')
    parser.add_argument('--model', type=str, default='pointnet2_att_hierarchical_gan', 
                        help='model name [default: pointnet2_att_hierarchical_gan]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate [default: 0.001]')
    parser.add_argument('--d_lr', default=0.001, type=float, help='Discriminator learning rate [default: 0.0001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer [default: Adam]')
    parser.add_argument('--npoint', type=int, default=40000, help='Point Number [default: 40000]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay [default: 0.7]')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for learning rate decay [default: 10]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--log_dir', type=str, default='att_hierarchical_gan', 
                        help='Log path [default: att_hierarchical_gan]')
    parser.add_argument('--n_epochs', type=int, default=50, help='Total number of epochs [default: 50]')
    parser.add_argument('--source_data_root', type=str, default='data/bim_scan/', 
                        help='Source domain data root [default: data/bim_scan/]')
    parser.add_argument('--target_data_root', type=str, default='data/stanford_indoor3d/', 
                        help='Target domain data root [default: data/stanford_indoor3d/]')
    
    # Hierarchical GAN specific parameters
    parser.add_argument('--lambda_adv', type=float, default=0.01, 
                        help='Adversarial loss weight [default: 0.01]')
    parser.add_argument('--d_steps', type=int, default=3, 
                        help='Discriminator update steps per generator step [default: 1]')
    parser.add_argument('--gradient_penalty', type=float, default=10.0, 
                        help='Gradient penalty weight for WGAN-GP [default: 10.0]')
    parser.add_argument('--use_wgan', action='store_true', default=False, 
                        help='Use WGAN-GP instead of LSGAN')
    parser.add_argument('--discriminator_level_indices', type=int, nargs='+', default=[1, 2, 3],
                        help='Discriminator level indices to use (1=dec2, 2=dec3, 3=dec4). '
                             'Default: [1, 2, 3] uses all three levels. '
                             'Example: --discriminator_level_indices 3 to use only level 3 (dec4, global). '
                             'Example: --discriminator_level_indices 1 3 to use levels 1 and 3.')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of epochs to train without adversarial loss [default: 5]')
    parser.add_argument('--cache_data', action='store_true', default=False, 
                        help='Cache all data in memory (faster but uses more RAM). If not set, data will be loaded from disk on-the-fly.')
    
    # Memory optimization and gradient accumulation
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before updating [default: 1]')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision training to save memory')
    
    # Mean Teacher consistency regularization (DEPRECATED - not recommended for domain adaptation)
    # NOTE: Mean Teacher typically hurts performance in domain adaptation tasks
    # because teacher predictions on target domain are inaccurate
    parser.add_argument('--use_mean_teacher', action='store_true', default=True,
                        help='[DEPRECATED] Use Mean Teacher (not recommended) [default: False]')
    parser.add_argument('--lambda_consistency', type=float, default=0.01,
                        help='Consistency loss weight (if use_mean_teacher) [default: 0.5]')
    parser.add_argument('--teacher_alpha', type=float, default=0.99,
                        help='EMA coefficient for Mean Teacher [default: 0.99]')
    parser.add_argument('--consistency_confidence_threshold', type=float, default=0.7,
                        help='Confidence threshold for consistency loss (only apply to high-confidence predictions) [default: 0.7]')
    parser.add_argument('--exclude_classes', type=str, nargs='+', default=['board', 'clutter'], 
                        help='Classes to exclude from training, e.g., --exclude_classes board clutter [default: board clutter]')
    
    return parser.parse_args()


def gradient_penalty(discriminator, real_features, fake_features, device):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = real_features.shape[0]
    alpha = torch.rand(batch_size, 1).to(device)
    
    # Expand alpha to match feature dimensions
    while len(alpha.shape) < len(real_features.shape):
        alpha = alpha.unsqueeze(-1)
    alpha = alpha.expand_as(real_features)
    
    interpolated = alpha * real_features + (1 - alpha) * fake_features
    interpolated.requires_grad_(True)
    
    interpolated_pred = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=interpolated_pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    
    return gp


# Removed consistency_rampup_weight - using fixed weight instead


class MeanTeacher:
    """
    Mean Teacher模型：使用指数移动平均(EMA)的teacher模型来提供稳定的伪标签
    
    原理：
    - Student模型（主模型）通过梯度下降更新
    - Teacher模型通过Student模型的EMA更新，提供更稳定的预测
    - Student模型的预测应该与Teacher模型的预测一致
    - 通过不同的数据增强来增强模型的鲁棒性
    """
    def __init__(self, student_model, alpha=0.999):
        """
        Args:
            student_model: Student模型（主训练模型）
            alpha: EMA系数，控制teacher更新的速度（越大更新越慢，越稳定）
        """
        self.student = student_model
        self.alpha = alpha
        
        # 创建teacher模型（深拷贝student）
        self.teacher = copy.deepcopy(student_model)
        
        # Teacher模型不需要梯度更新（只通过EMA更新）
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 确保teacher在eval模式（用于推理）
        self.teacher.eval()
        
        # 初始化数据增强变换（用于一致性正则化）
        # 只使用旋转和翻转，不使用抖动
        self.consistency_augment = t.Compose([
            t.RandomRotate(),
            t.RandomFlip(p=0.5),
        ])
    
    def update_teacher(self):
        """
        更新teacher模型参数：使用指数移动平均
        teacher_param = alpha * teacher_param + (1 - alpha) * student_param
        """
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), 
                self.student.parameters()
            ):
                teacher_param.data = (
                    self.alpha * teacher_param.data + 
                    (1 - self.alpha) * student_param.data
                )
    
    def _apply_augmentation(self, coords, offsets):
        """
        对点云坐标应用数据增强
        
        Args:
            coords: 点云坐标 torch.Tensor (n, 3)
            offsets: batch偏移量 torch.Tensor (b,)
        
        Returns:
            augmented_coords: 增强后的点云坐标 torch.Tensor (n, 3)
        """
        # 将tensor转换为numpy进行处理
        coords_np = coords.detach().cpu().numpy()
        
        # 应用增强（需要dummy label，因为transform接口需要coord和label）
        dummy_label = np.zeros(coords_np.shape[0], dtype=np.int64)
        coords_aug, _ = self.consistency_augment(coords_np, dummy_label)
        
        # 转换回tensor并移到原设备
        coords_aug = torch.from_numpy(coords_aug).float().to(coords.device)
        
        return coords_aug
    
    def consistency_loss(self, coords, offsets, use_amp=False, confidence_threshold=0.7):
        """
        计算一致性损失：student预测应该与teacher预测一致
        
        实现：
        - Teacher使用原始数据（提供稳定的伪标签）
        - Student使用增强数据（旋转+翻转，学习对几何变换的鲁棒性）
        - 只对高置信度的teacher预测应用一致性约束（避免低置信度预测误导模型）
        - 用于target domain的无监督学习
        
        Args:
            coords: 点云坐标 (n, 3)
            offsets: batch偏移量 (b,)
            use_amp: 是否使用混合精度
            confidence_threshold: 置信度阈值，只对teacher预测置信度高于此值的点计算一致性损失 [default: 0.7]
        
        Returns:
            consistency_loss: 一致性损失（KL散度），只对高置信度预测计算
        """
        # Teacher使用原始数据（提供稳定的伪标签）
        # Student使用增强数据（旋转+翻转，学习对几何变换的鲁棒性）
        coords_student = self._apply_augmentation(coords, offsets)
        coords_teacher = coords  # Teacher 使用原始数据
        
        # Student预测（需要梯度）
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_student, _, _ = self.student([coords_student, coords_student, offsets], return_features=True)
        
        # Teacher预测（不需要梯度，用于生成伪标签）
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_teacher, _, _ = self.teacher([coords_teacher, coords_teacher, offsets], return_features=True)
        
        # 计算teacher预测的置信度（最大概率）
        pred_teacher_prob = F.softmax(pred_teacher, dim=1)
        teacher_confidence, _ = pred_teacher_prob.max(dim=1)  # (N,)
        
        # 创建置信度掩码：只对高置信度的预测计算一致性损失
        confidence_mask = teacher_confidence >= confidence_threshold
        
        # 如果没有高置信度的预测，返回0损失
        if confidence_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_student.device, requires_grad=True)
        
        # 只对高置信度的预测计算KL散度
        pred_student_log_prob = F.log_softmax(pred_student, dim=1)
        
        # 应用掩码：只计算高置信度点的KL散度
        # 使用reduction='none'然后手动应用掩码
        kl_div_per_point = F.kl_div(
            pred_student_log_prob,
            pred_teacher_prob,
            reduction='none'
        ).sum(dim=1)  # (N,)
        
        # 只对高置信度的点计算平均损失
        consistency_loss = kl_div_per_point[confidence_mask].mean()
        
        return consistency_loss
    
    def get_teacher_model(self):
        """返回teacher模型（用于推理）"""
        return self.teacher
    
    def state_dict(self):
        """返回teacher模型的状态字典（用于保存）"""
        return self.teacher.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载teacher模型的状态字典（用于恢复）"""
        self.teacher.load_state_dict(state_dict)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Att_Hierarchical_GAN_Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    log_string('Attention + Hierarchical GAN Configuration:')
    log_string(f'  - Adversarial loss weight: {args.lambda_adv}')
    log_string(f'  - Discriminator steps: {args.d_steps}')
    log_string(f'  - Use WGAN-GP: {args.use_wgan}')
    level_names = {1: 'dec2 (64-dim, local)', 2: 'dec3 (128-dim, local)', 3: 'dec4 (256-dim, global)'}
    level_desc = ', '.join([f'Level {idx} ({level_names[idx]})' for idx in args.discriminator_level_indices])
    log_string(f'  - Using discriminator levels: {level_desc} (total: {len(args.discriminator_level_indices)} levels)')
    log_string(f'  - Warmup epochs (no adversarial): {args.warmup_epochs}')
    log_string(f'  - Gradient accumulation steps: {args.gradient_accumulation_steps}')
    log_string(f'  - Use mixed precision (AMP): {args.use_amp}')
    if args.gradient_accumulation_steps > 1:
        log_string(f'  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps}')
    
    # Mean Teacher configuration
    if args.use_mean_teacher:
        log_string('=' * 60)
        log_string('WARNING: Mean Teacher is DEPRECATED and not recommended!')
        log_string('Reason: Typically hurts performance in domain adaptation tasks.')
        log_string('=' * 60)
        log_string('Mean Teacher Configuration:')
        log_string(f'  - Consistency loss weight: {args.lambda_consistency}')
        log_string(f'  - Teacher EMA coefficient (alpha): {args.teacher_alpha}')
        log_string(f'  - Confidence threshold: {args.consistency_confidence_threshold} (only high-confidence predictions)')
        log_string(f'  - Applied on: Target domain (unlabeled data)')

    root_source = args.source_data_root
    root_target = args.target_data_root
    
    # 创建标签映射
    exclude_classes = args.exclude_classes if args.exclude_classes else []
    label_mapping, new_classes, new_class2label, new_seg_label_to_cat = create_label_mapping(exclude_classes)
    
    # 更新全局变量以使用新的类别映射
    global seg_label_to_cat
    seg_label_to_cat = new_seg_label_to_cat
    
    NUM_CLASSES = len(new_classes)
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    
    if exclude_classes:
        log_string(f"Excluding classes: {exclude_classes}")
        log_string(f"Training with {NUM_CLASSES} classes: {new_classes}")
    else:
        log_string(f"Training with all {NUM_CLASSES} classes")

    print("start loading training data ...")
    log_string(f"Data loading mode: {'cache in memory' if args.cache_data else 'load from disk on-the-fly'}")
    
    # Define training transform
    train_transform = t.Compose([
        # t.RandomScale([0.9, 1.1], anisotropic=True),
        t.RandomRotate(),
        # t.RandomShift([0.2, 0.2, 0.0]),
        t.RandomFlip(p=0.5),
        # t.RandomJitter(sigma=0.01, clip=0.05)
    ])
    
    # Source domain (labeled data - BIM/Scan)
    SRC_TRAIN_DATASET_RAW = S3DISDatasetTrans(split='train', data_root=root_source, num_point=NUM_POINT,
                                          test_area=args.test_area, sample_rate=1.0, 
                                          shuffle_index=True, transform=train_transform, cache=args.cache_data)
    SRC_TRAIN_DATASET = LabelMappingDataset(SRC_TRAIN_DATASET_RAW, label_mapping)
    
    # Target domain (unlabeled data - Stanford Indoor3D)
    # Note: Target domain doesn't have labels, so we don't need to apply label mapping
    # But we still wrap it for consistency (it will just pass through)
    TAR_TRAIN_DATASET_RAW = S3DISDatasetTrans(split='train', data_root=root_target, num_point=NUM_POINT,
                                          test_area=args.test_area, sample_rate=1.0, 
                                          shuffle_index=True, transform=train_transform, cache=args.cache_data)
    # Target domain doesn't have labels, so we don't wrap it with LabelMappingDataset
    TAR_TRAIN_DATASET = TAR_TRAIN_DATASET_RAW
    
    # Test dataset for evaluation (on source domain)
    TEST_DATASET_RAW = S3DISDatasetTrans(split='test', data_root=root_source, num_point=NUM_POINT,
                                     test_area=args.test_area, sample_rate=1.0, 
                                     shuffle_index=False, transform=None, cache=args.cache_data)
    TEST_DATASET = LabelMappingDataset(TEST_DATASET_RAW, label_mapping)

    # Reduce num_workers to save memory
    num_workers = 2 if args.cache_data else 5
    
    src_trainDataLoader = torch.utils.data.DataLoader(SRC_TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                                                      num_workers=num_workers, pin_memory=True, drop_last=True,
                                                      worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
                                                      collate_fn=collate_fn)
    
    tar_trainDataLoader = torch.utils.data.DataLoader(TAR_TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                                                      num_workers=num_workers, pin_memory=True, drop_last=True,
                                                      worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
                                                      collate_fn=collate_fn)
    
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=num_workers, pin_memory=True,
                                                 worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
                                                 collate_fn=collate_fn)

    # 重新计算权重：只保留保留类别的权重
    if exclude_classes:
        original_weights = SRC_TRAIN_DATASET_RAW.labelweights
        # 只保留保留类别的权重
        keep_indices = [class2label[cls] for cls in new_classes]
        weights = torch.Tensor(original_weights[keep_indices]).cuda()
    else:
        weights = torch.Tensor(SRC_TRAIN_DATASET_RAW.labelweights).cuda()

    log_string("The number of source training data is: %d" % len(SRC_TRAIN_DATASET))
    log_string("The number of target training data is: %d" % len(TAR_TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    # Generator (attention-based semantic segmentation network)
    generator = MODEL.get_model(
        NUM_CLASSES, 
        discriminator_level_indices=args.discriminator_level_indices
    ).cuda()
    criterion = MODEL.get_loss().cuda()
    generator.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        
        # Support both old format (generator_state_dict) and new format (model_state_dict)
        if 'generator_state_dict' in checkpoint:
            # Load full generator state (including discriminators if present)
            state_dict_to_load = checkpoint['generator_state_dict']
            log_string('Loading from generator_state_dict (full model with discriminators)')
        elif 'model_state_dict' in checkpoint:
            # Load classifier state (without discriminators)
            state_dict_to_load = checkpoint['model_state_dict']
            log_string('Loading from model_state_dict (classifier only, discriminators will be initialized)')
        else:
            raise KeyError("No valid state_dict key found in checkpoint")
        
        # Load with strict=False to allow missing discriminator parameters
        missing_keys, unexpected_keys = generator.load_state_dict(state_dict_to_load, strict=False)
        
        if missing_keys:
            log_string(f'Missing keys (will be randomly initialized): {len(missing_keys)} keys')
            if any('discriminators' in k for k in missing_keys):
                log_string('  - Discriminator parameters missing (expected for checkpoints from non-GAN training)')
        if unexpected_keys:
            log_string(f'Unexpected keys (will be ignored): {unexpected_keys}')
            
        log_string('Use pretrain model')
        
        # Initialize discriminators if they were not loaded
        if hasattr(generator, 'discriminators') and any('discriminators' in k for k in missing_keys):
            log_string('Initializing discriminator weights...')
            generator.discriminators.apply(weights_init)
            
    except Exception as e:
        log_string(f'No existing model or error loading: {e}')
        log_string('Starting training from scratch...')
        start_epoch = 0
        # Initialize all weights including discriminators
        generator.apply(weights_init)

    # Optimizers
    if args.optimizer == 'Adam':
        g_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        if hasattr(generator, 'discriminators'):
            d_optimizer = torch.optim.Adam(
                generator.discriminators.parameters(),
                lr=args.d_lr,
                betas=(0.5, 0.999),
                eps=1e-08,
                weight_decay=args.decay_rate
            )
        else:
            d_optimizer = None
    else:
        g_optimizer = torch.optim.SGD(generator.parameters(), lr=args.learning_rate, 
                                      momentum=0.9, weight_decay=args.decay_rate)
        if hasattr(generator, 'discriminators'):
            d_optimizer = torch.optim.SGD(generator.discriminators.parameters(), 
                                         lr=args.d_lr, momentum=0.9, weight_decay=args.decay_rate)
        else:
            d_optimizer = None

    # Initialize mixed precision training
    scaler_g = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=args.use_amp) if d_optimizer is not None else None
    
    # Initialize Mean Teacher if enabled
    mean_teacher = None
    if args.use_mean_teacher:
        log_string('Initializing Mean Teacher...')
        mean_teacher = MeanTeacher(generator, alpha=args.teacher_alpha)
        log_string('Mean Teacher initialized successfully')
    
    # Try to load scaler states from checkpoint if available
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        if 'scaler_g_state_dict' in checkpoint:
            scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
            log_string('Loaded scaler_g state from checkpoint')
        if scaler_d is not None and 'scaler_d_state_dict' in checkpoint:
            scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
            log_string('Loaded scaler_d state from checkpoint')
        # Support both naming conventions for optimizer
        if 'optimizer_state_dict' in checkpoint:
            g_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            log_string('Loaded optimizer state from checkpoint')
        elif 'g_optimizer_state_dict' in checkpoint:
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            log_string('Loaded g_optimizer state from checkpoint')
        if d_optimizer is not None and 'd_optimizer_state_dict' in checkpoint:
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            log_string('Loaded d_optimizer state from checkpoint')
        # Load Mean Teacher state if available
        if mean_teacher is not None and 'teacher_state_dict' in checkpoint:
            mean_teacher.load_state_dict(checkpoint['teacher_state_dict'])
            log_string('Loaded Mean Teacher state from checkpoint')
    except Exception as e:
        log_string(f'Could not load optimizer/scaler states: {e}')

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.n_epochs):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.n_epochs))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        d_lr = max(args.d_lr * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Generator Learning rate:%f' % lr)
        log_string('Discriminator Learning rate:%f' % d_lr)

        for param_group in g_optimizer.param_groups:
            param_group['lr'] = lr
        if d_optimizer is not None:
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = d_lr

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        generator = generator.apply(lambda x: bn_momentum_adjust(x, momentum))

        # Check if we should use adversarial training
        use_adversarial = (epoch >= args.warmup_epochs) and hasattr(generator, 'discriminators')
        if epoch == args.warmup_epochs:
            log_string('===== Starting adversarial training =====')

        num_batches = min(len(src_trainDataLoader), len(tar_trainDataLoader))
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        adv_loss_sum = 0
        d_loss_sum = 0
        consistency_loss_sum = 0
        
        # Gradient accumulation tracking
        accumulation_counter = 0
        
        # Check if we should use consistency regularization
        use_consistency = (args.use_mean_teacher and mean_teacher is not None)

        generator = generator.train()

        # Create iterators
        src_iter = iter(src_trainDataLoader)
        tar_iter = iter(tar_trainDataLoader)

        loop = tqdm(range(num_batches), leave=True, smoothing=0.9)
        loop.set_description(f'Att-HierGAN (Epoch {epoch+1})')

        for i in loop:
            # Get source data (labeled)
            try:
                src_coord, src_target, src_offset = next(src_iter)
            except StopIteration:
                src_iter = iter(src_trainDataLoader)
                src_coord, src_target, src_offset = next(src_iter)
            
            # Get target data (unlabeled)
            try:
                tar_coord, _, tar_offset = next(tar_iter)
            except StopIteration:
                tar_iter = iter(tar_trainDataLoader)
                tar_coord, _, tar_offset = next(tar_iter)

            # Move data to GPU
            src_coord = src_coord.cuda(non_blocking=True)
            src_target = src_target.cuda(non_blocking=True)
            src_offset = src_offset.cuda(non_blocking=True)
            
            tar_coord = tar_coord.cuda(non_blocking=True)
            tar_offset = tar_offset.cuda(non_blocking=True)
            
            # Flatten target for loss computation
            src_target_flat = src_target.view(-1, 1)[:, 0]

            # ========== Train Discriminator ==========
            if use_adversarial and d_optimizer is not None:
                for _ in range(args.d_steps):
                    d_optimizer.zero_grad()
                    
                    # Source features (real)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=args.use_amp):
                            _, src_multi_features, _ = generator([src_coord, src_coord, src_offset], return_features=True)
                    
                    # Target features (fake)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=args.use_amp):
                            _, tar_multi_features, _ = generator([tar_coord, tar_coord, tar_offset], return_features=True)
                    
                    # Use mixed precision for discriminator forward and backward
                    with torch.cuda.amp.autocast(enabled=args.use_amp):
                        d_loss_total = 0.0
                        
                        for level_idx in range(min(len(src_multi_features), len(tar_multi_features))):
                            src_feat = src_multi_features[level_idx].detach()
                            tar_feat = tar_multi_features[level_idx].detach()
                            
                            # Get discriminator for this level
                            discriminator = generator.discriminators.discriminators[level_idx]
                            
                            # Real and fake predictions
                            d_real = discriminator(src_feat)
                            d_fake = discriminator(tar_feat)
                            
                            if args.use_wgan:
                                # WGAN-GP loss
                                d_loss_level = -torch.mean(d_real) + torch.mean(d_fake)
                                
                                # Gradient penalty
                                gp = gradient_penalty(discriminator, src_feat, tar_feat, src_feat.device)
                                d_loss_level += args.gradient_penalty * gp
                            else:
                                # LSGAN loss
                                target_real = torch.ones_like(d_real)
                                target_fake = torch.zeros_like(d_fake)
                                
                                d_loss_real = F.mse_loss(d_real, target_real)
                                d_loss_fake = F.mse_loss(d_fake, target_fake)
                                d_loss_level = (d_loss_real + d_loss_fake) * 0.5
                            
                            d_loss_total += d_loss_level
                        
                        # Average discriminator loss across levels
                        if len(src_multi_features) > 0:
                            d_loss_total = d_loss_total / len(src_multi_features)
                    
                    scaler_d.scale(d_loss_total).backward()
                    scaler_d.step(d_optimizer)
                    scaler_d.update()
                    d_loss_sum += d_loss_total.item()
                    
                    # Clear discriminator features from memory
                    del src_multi_features, tar_multi_features
                    if level_idx >= 0:  # Just to use the variable
                        torch.cuda.empty_cache()

            # ========== Train Generator ==========
            # Only zero grad at the start of accumulation
            if accumulation_counter == 0:
                g_optimizer.zero_grad()
            
            # Use mixed precision for generator forward and backward
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                # Source forward pass (supervised loss)
                if use_adversarial:
                    seg_pred_src, src_multi_features, trans_feat_src = generator([src_coord, src_coord, src_offset], 
                                                                                  return_features=True)
                else:
                    seg_pred_src, trans_feat_src = generator([src_coord, src_coord, src_offset])
                
                # Supervised segmentation loss
                seg_loss = criterion(seg_pred_src, src_target_flat, trans_feat_src, weights)
                
                # Adversarial loss
                adv_loss_total = 0.0
                if use_adversarial:
                    # Target forward pass
                    _, tar_multi_features, _ = generator([tar_coord, tar_coord, tar_offset], return_features=True)
                    
                    # Compute adversarial loss for each level
                    for level_idx in range(len(tar_multi_features)):
                        tar_feat = tar_multi_features[level_idx]
                        discriminator = generator.discriminators.discriminators[level_idx]
                        
                        tar_scores = discriminator(tar_feat)
                        
                        if args.use_wgan:
                            adv_loss_level = -torch.mean(tar_scores)
                        else:
                            # LSGAN loss - generator tries to make target look like source
                            target_real = torch.ones_like(tar_scores)
                            adv_loss_level = F.mse_loss(tar_scores, target_real)
                        
                        adv_loss_total += adv_loss_level
                    
                    # Average adversarial loss across levels
                    if len(tar_multi_features) > 0:
                        adv_loss_total = adv_loss_total / len(tar_multi_features)
                
                # Consistency regularization loss (Mean Teacher)
                consistency_loss_total = 0.0
                if use_consistency:
                    # Apply consistency loss on target domain
                    # Teacher provides pseudo labels for target domain
                    # Only apply to high-confidence predictions to avoid misleading the model
                    consistency_loss_total = mean_teacher.consistency_loss(
                        tar_coord, tar_offset, 
                        use_amp=args.use_amp,
                        confidence_threshold=args.consistency_confidence_threshold
                    )
                
                # Total generator loss (scaled by accumulation steps)
                total_g_loss = seg_loss
                if use_adversarial:
                    total_g_loss += args.lambda_adv * adv_loss_total
                if use_consistency:
                    total_g_loss += args.lambda_consistency * consistency_loss_total
                total_g_loss = total_g_loss / args.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler_g.scale(total_g_loss).backward()
            
            # Update accumulation counter
            accumulation_counter += 1
            
            # Update weights only after accumulating enough gradients
            if accumulation_counter >= args.gradient_accumulation_steps:
                scaler_g.step(g_optimizer)
                scaler_g.update()
                accumulation_counter = 0
                
                # Update Mean Teacher after optimizer step
                if use_consistency:
                    mean_teacher.update_teacher()
            
            # Clear features from memory
            if use_adversarial:
                del src_multi_features, tar_multi_features
                torch.cuda.empty_cache()

            # Calculate accuracy
            with torch.no_grad():
                pred_choice = seg_pred_src.detach().cpu().data.max(1)[1].numpy()
                batch_label = src_target_flat.detach().cpu().numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                # Handle variable point numbers
                num_points = len(batch_label)
                total_seen += num_points
                acc = correct / num_points
                # Note: seg_loss already computed, we just store the unscaled version
                loss_sum += seg_loss.item()
            
            if use_adversarial:
                adv_loss_sum += adv_loss_total.item()
            
            if use_consistency:
                consistency_loss_sum += consistency_loss_total.item()
                
            # Update progress bar (show unscaled losses for readability)
            postfix_dict = {
                'seg_loss': f'{seg_loss.item():.4f}',
                'acc': f'{acc * 100:.2f}%',
                'accum': f'{accumulation_counter}/{args.gradient_accumulation_steps}'
            }
            if use_adversarial:
                postfix_dict['adv_loss'] = f'{adv_loss_total.item():.4f}'
                postfix_dict['d_loss'] = f'{d_loss_total.item():.4f}' if 'd_loss_total' in locals() and d_optimizer is not None else 'N/A'
            if use_consistency:
                postfix_dict['cons_loss'] = f'{consistency_loss_total.item():.4f}'
            
            loop.set_postfix(postfix_dict)
            
            # Clear predictions from memory
            del seg_pred_src, pred_choice, batch_label
            del src_coord, src_target, src_offset, tar_coord, tar_offset

        log_string('Training mean seg loss: %f' % (loss_sum / num_batches))
        if use_adversarial:
            log_string('Training mean adv loss: %f' % (adv_loss_sum / num_batches))
            if d_optimizer is not None:
                log_string('Training mean d loss: %f' % (d_loss_sum / (num_batches * args.d_steps)))
        if use_consistency:
            log_string('Training mean consistency loss: %f (weight: %.4f)' % 
                       (consistency_loss_sum / num_batches, args.lambda_consistency))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            
            # Filter out discriminator parameters for st.py compatibility
            full_state_dict = generator.state_dict()
            classifier_state_dict = {k: v for k, v in full_state_dict.items() 
                                    if not k.startswith('discriminators.')}
            
            state = {
                'epoch': epoch,
                # Use model_state_dict as primary key for compatibility with st.py (without discriminators)
                'model_state_dict': classifier_state_dict,
                'optimizer_state_dict': g_optimizer.state_dict(),
                'scaler_g_state_dict': scaler_g.state_dict(),
                # Keep full generator state_dict for resuming GAN training
                'generator_state_dict': full_state_dict,
                'g_optimizer_state_dict': g_optimizer.state_dict(),
            }
            if d_optimizer is not None:
                state['d_optimizer_state_dict'] = d_optimizer.state_dict()
                state['scaler_d_state_dict'] = scaler_d.state_dict()
            if mean_teacher is not None:
                state['teacher_state_dict'] = mean_teacher.state_dict()
            torch.save(state, savepath)
            log_string('Saving model....')
        
        # Clean up memory after training epoch
        torch.cuda.empty_cache()
        gc.collect()
        '''Evaluate on test set'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            generator = generator.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (coord, target, offset) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                coord = coord.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                offset = offset.cuda(non_blocking=True)
                
                target_flat = target.view(-1, 1)[:, 0]

                # Use mixed precision for evaluation to save memory
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    seg_pred, trans_feat = generator([coord, coord, offset])
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                    
                    loss = criterion(seg_pred, target_flat, trans_feat, weights)
                
                loss_sum += loss.item()

                pred_val = seg_pred.contiguous().cpu().data.numpy()
                pred_val = np.argmax(pred_val, 1)
                batch_label = target.cpu().data.numpy()
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                # Handle variable point numbers
                total_seen += len(batch_label.flatten())
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                
                # Clear memory after each batch
                del coord, target, offset, seg_pred, trans_feat, pred_val, batch_label

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save best model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                
                # Filter out discriminator parameters for st.py compatibility
                full_state_dict = generator.state_dict()
                classifier_state_dict = {k: v for k, v in full_state_dict.items() 
                                        if not k.startswith('discriminators.')}
                
                log_string(f'Saving classifier with {len(classifier_state_dict)} params (discriminators excluded)')
                log_string(f'Full generator has {len(full_state_dict)} params')
                
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    # Use model_state_dict as primary key for compatibility with st.py (without discriminators)
                    'model_state_dict': classifier_state_dict,
                    'optimizer_state_dict': g_optimizer.state_dict(),
                    'scaler_g_state_dict': scaler_g.state_dict(),
                    # Keep full generator state_dict for resuming GAN training
                    'generator_state_dict': full_state_dict,
                    'g_optimizer_state_dict': g_optimizer.state_dict(),
                }
                if d_optimizer is not None:
                    state['d_optimizer_state_dict'] = d_optimizer.state_dict()
                    state['scaler_d_state_dict'] = scaler_d.state_dict()
                if mean_teacher is not None:
                    state['teacher_state_dict'] = mean_teacher.state_dict()
                torch.save(state, savepath)
                log_string('Best model saved!')
            log_string('Best mIoU: %f' % best_iou)
        
        # Clean up memory after evaluation
        torch.cuda.empty_cache()
        gc.collect()
        
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)

