"""
域适应训练脚本 - 基于PMAN论文框架
使用 pointnet2_sem_seg_att 作为生成器骨干网络
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDatasetTrans
import torch
import torch.nn as nn
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
    """
    if exclude_classes is None or len(exclude_classes) == 0:
        label_mapping = np.arange(len(classes))
        return label_mapping, classes, class2label, seg_label_to_cat
    
    exclude_indices = [class2label[cls] for cls in exclude_classes if cls in class2label]
    new_classes = [cls for cls in classes if cls not in exclude_classes]
    new_class2label = {cls: i for i, cls in enumerate(new_classes)}
    new_seg_label_to_cat = {}
    for i, cat in enumerate(new_classes):
        new_seg_label_to_cat[i] = cat
    
    label_mapping = np.full(len(classes), -1, dtype=np.int64)
    new_idx = 0
    for orig_idx, cls in enumerate(classes):
        if cls not in exclude_classes:
            label_mapping[orig_idx] = new_idx
            new_idx += 1
    
    return label_mapping, new_classes, new_class2label, new_seg_label_to_cat


class Generator(nn.Module):
    """生成器G：基于pointnet2_sem_seg_att，返回特征F和主分类器概率P_m"""
    def __init__(self, backbone_model, num_classes):
        super(Generator, self).__init__()
        self.backbone = backbone_model
        self.num_classes = num_classes
        # 特征维度是32（从dec1输出的特征维度）
        self.feature_dim = 32
    
    def forward(self, pxo):
        """
        前向传播
        Returns:
            features: 特征 (N, 32)
            P_m: 主分类器概率 (N, num_classes)
        """
        # 直接使用backbone的前向传播逻辑，但提取特征
        p0, x0, o0 = pxo
        x0 = p0 if self.backbone.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.backbone.enc2([p0, x0, o0])
        p2, x2, o2 = self.backbone.enc3([p1, x1, o1])
        p3, x3, o3 = self.backbone.enc4([p2, x2, o2])
        p4, x4, o4 = self.backbone.enc5([p3, x3, o3])
        x4 = self.backbone.dec5[1:]([p4, self.backbone.dec5[0]([p4, x4, o4]), o4])[1]
        x3 = self.backbone.dec4[1:]([p3, self.backbone.dec4[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.backbone.dec3[1:]([p2, self.backbone.dec3[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.backbone.dec2[1:]([p1, self.backbone.dec2[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        features = self.backbone.dec1[1:]([p0, self.backbone.dec1[0]([p0, None, o0], [p1, x1, o1]), o0])[1]
        
        # 通过主分类器得到logits，然后计算概率
        logits_m = self.backbone.cls(features)
        P_m = F.softmax(logits_m, dim=1)
        
        return features, P_m


class Discriminator(nn.Module):
    """判别器D：判断自信息图来自源域还是目标域"""
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        
        # 全卷积网络，处理自信息图
        # 输入: (N, K) 自信息图
        # 输出: (N, 1) 判别结果
        
        # 由于点云数据是稀疏的，我们使用MLP而不是卷积
        # 5层MLP，逐步降维
        self.layers = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, self_info_map):
        """
        Args:
            self_info_map: 自信息图 (N, num_classes)
        Returns:
            score: 判别分数 (N, 1)
        """
        return self.layers(self_info_map)


def compute_self_info(P_m):
    """
    计算自信息图 S = -P_m * log(P_m)
    Args:
        P_m: 主分类器概率 (N, num_classes)
    Returns:
        S: 自信息图 (N, num_classes)
    """
    eps = 1e-8
    S = -P_m * torch.log(P_m + eps)
    return S


def compute_prototypes(G, source_loader, num_classes, device):
    """
    计算源域特征原型
    Args:
        G: 生成器
        source_loader: 源域数据加载器
        num_classes: 类别数
        device: 设备
    Returns:
        prototypes: 原型 (num_classes, feature_dim)
    """
    G.eval()
    feature_dim = G.feature_dim
    
    prototypes = torch.zeros(num_classes, feature_dim).to(device)
    class_counts = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for coord, target, offset in tqdm(source_loader, desc='Computing prototypes'):
            coord = coord.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            
            # 获取特征
            F_s, _ = G([coord, coord, offset])
            target_flat = target.view(-1)
            
            # 对每个类别计算原型
            for k in range(num_classes):
                mask = (target_flat == k)
                if mask.sum() > 0:
                    prototypes[k] += F_s[mask].sum(dim=0)
                    class_counts[k] += mask.sum()
    
    # 归一化
    prototypes = prototypes / (class_counts.unsqueeze(1) + 1e-8)
    
    G.train()
    return prototypes


def compute_auxiliary_classifier(features, prototypes):
    """
    辅助分类器C_a：基于特征和原型计算概率
    Args:
        features: 特征 (N, feature_dim)
        prototypes: 原型 (num_classes, feature_dim)
    Returns:
        P_a: 辅助分类器概率 (N, num_classes)
    """
    # 计算L2距离
    # features: (N, feature_dim), prototypes: (num_classes, feature_dim)
    # distances: (N, num_classes)
    distances_sq = torch.sum((features.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2, dim=2)
    
    # 距离越小概率越大，所以取负数
    logits_a = -distances_sq
    
    # Softmax归一化
    P_a = F.softmax(logits_a, dim=1)
    
    return P_a


def compute_discrepancy_map(P_m, P_a):
    """
    计算差异图 M = 1 - cosine_similarity(P_m, P_a)
    Args:
        P_m: 主分类器概率 (N, num_classes)
        P_a: 辅助分类器概率 (N, num_classes)
    Returns:
        M: 差异图 (N,)
    """
    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(P_m, P_a, dim=1)
    M = 1 - cosine_sim
    return M


def compute_pseudo_labels(P_m, th_p=0.95, th_e=0.01):
    """
    生成伪标签
    Args:
        P_m: 主分类器概率 (N, num_classes)
        th_p: 概率阈值
        th_e: 熵阈值（归一化后）
    Returns:
        pseudo_labels: 伪标签 (N,)，-1表示不满足条件
    """
    # 计算最大概率和熵
    max_prob, pred = torch.max(P_m, dim=1)
    
    # 计算归一化熵: -sum(p * log(p)) / log(num_classes)
    num_classes = P_m.shape[1]
    entropy = -torch.sum(P_m * torch.log(P_m + 1e-8), dim=1) / np.log(num_classes)
    
    # 生成伪标签
    mask = (max_prob > th_p) & (entropy < th_e)
    pseudo_labels = torch.full_like(pred, -1)
    pseudo_labels[mask] = pred[mask]
    
    return pseudo_labels


def compute_loss_seg(P_s_m, Y_s, weights):
    """
    源域分割损失 L_s_seg
    Args:
        P_s_m: 源域主分类器概率 (N, num_classes)
        Y_s: 源域标签 (N,)
        weights: 类别权重
    Returns:
        loss: 交叉熵损失
    """
    # 转换为log概率
    log_P_s_m = torch.log(P_s_m + 1e-8)
    
    # 计算交叉熵损失
    loss = F.nll_loss(log_P_s_m, Y_s, weight=weights, ignore_index=-1)
    return loss


def compute_loss_adv_adaptive(P_t_m, P_t_a, D, gamma_local=80.0, delta=0.4):
    """
    自适应对抗损失 L_t,ada_adv
    Args:
        P_t_m: 目标域主分类器概率 (N, num_classes)
        P_t_a: 目标域辅助分类器概率 (N, num_classes)
        D: 判别器
        gamma_local: 局部权重系数
        delta: 偏移量
    Returns:
        loss: 自适应对抗损失
    """
    # 计算差异图
    M = compute_discrepancy_map(P_t_m, P_t_a)
    
    # 计算自信息图
    S_t = compute_self_info(P_t_m)
    
    # 判别器输出
    D_on_target = D(S_t).squeeze(-1)  # (N,)
    
    # 计算权重
    weights = gamma_local * M + delta
    
    # 源域标签为0
    source_label = torch.zeros_like(D_on_target)
    
    # 加权MSE损失
    loss = torch.mean(weights * (D_on_target - source_label) ** 2)
    
    return loss


def compute_loss_ppd_source(F_s, Y_s, prototypes):
    """
    源域点到原型损失 L_s_ppd
    Args:
        F_s: 源域特征 (N, feature_dim)
        Y_s: 源域标签 (N,)
        prototypes: 原型 (num_classes, feature_dim)
    Returns:
        loss: L1距离损失
    """
    # 获取每个点对应的原型
    prototypes_selected = prototypes[Y_s]  # (N, feature_dim)
    
    # 计算L1距离
    loss = torch.mean(torch.abs(F_s - prototypes_selected))
    
    return loss


def compute_loss_ppd_target(F_t, P_t_m, prototypes, th_p=0.95, th_e=0.01):
    """
    目标域点到原型损失 L_t_ppd
    Args:
        F_t: 目标域特征 (N, feature_dim)
        P_t_m: 目标域主分类器概率 (N, num_classes)
        prototypes: 原型 (num_classes, feature_dim)
        th_p: 概率阈值
        th_e: 熵阈值
    Returns:
        loss: 加权L1距离损失
    """
    # 生成伪标签
    pseudo_labels = compute_pseudo_labels(P_t_m, th_p, th_e)
    
    # 只计算有伪标签的点
    mask = (pseudo_labels >= 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=F_t.device, requires_grad=True)
    
    F_t_masked = F_t[mask]
    pseudo_labels_masked = pseudo_labels[mask]
    
    # 获取对应的原型
    prototypes_selected = prototypes[pseudo_labels_masked]  # (M, feature_dim)
    
    # 计算余弦相似度作为渐进权重
    cosine_sim = F.cosine_similarity(F_t_masked, prototypes_selected, dim=1)
    d_t_cos = 0.5 * cosine_sim + 0.5  # 归一化到[0, 1]
    
    # 计算加权L1距离
    l1_dist = torch.abs(F_t_masked - prototypes_selected).sum(dim=1)
    loss = torch.mean(d_t_cos * l1_dist)
    
    return loss


def compute_loss_discriminator(S_s, S_t, D):
    """
    判别器损失 L_D
    Args:
        S_s: 源域自信息图 (N_s, num_classes)
        S_t: 目标域自信息图 (N_t, num_classes)
        D: 判别器
    Returns:
        loss: MSE损失
    """
    # 源域标签为0，目标域标签为1
    D_on_source = D(S_s).squeeze(-1)
    D_on_target = D(S_t.detach()).squeeze(-1)  # detach很重要
    
    loss_source = F.mse_loss(D_on_source, torch.zeros_like(D_on_source))
    loss_target = F.mse_loss(D_on_target, torch.ones_like(D_on_target))
    
    loss = 0.5 * (loss_source + loss_target)
    return loss


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power=0.9):
    """Poly学习率衰减策略"""
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def parse_args():
    parser = argparse.ArgumentParser('Domain Adaptation Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_att', help='model name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--epoch', default=50, type=int, help='Epoch to run')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--log_dir', type=str, default='PMAN', help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=20000, help='Point Number')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test')
    parser.add_argument('--data_root_source', type=str, default='data/bim_indoor3d', help='Source domain data root')
    parser.add_argument('--data_root_target', type=str, default='data/stanford_indoor3d', help='Target domain data root')
    parser.add_argument('--cache_data', action='store_true', default=False, help='Cache data in memory')
    parser.add_argument('--exclude_classes', type=str, nargs='+', default=['board', 'clutter'], help='Classes to exclude')
    
    # 域适应超参数
    parser.add_argument('--lr_G', type=float, default=2.5e-4, help='Generator learning rate')
    parser.add_argument('--lr_D', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--gamma_2', type=float, default=0.001, help='Adversarial loss weight')
    parser.add_argument('--gamma_3', type=float, default=1.0, help='Point-to-prototype loss weight')
    parser.add_argument('--gamma_local', type=float, default=80.0, help='Local weight coefficient')
    parser.add_argument('--delta', type=float, default=0.4, help='Offset for adaptive adversarial loss')
    parser.add_argument('--th_p', type=float, default=0.95, help='Probability threshold for pseudo labels')
    parser.add_argument('--th_e', type=float, default=0.01, help='Entropy threshold for pseudo labels')
    parser.add_argument('--update_prototypes_every', type=int, default=5, help='Update prototypes every N epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs (only train on source domain) [default: 5]')
    
    return parser.parse_args()


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
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    if args.warmup_epochs > 0:
        log_string(f'Warmup phase: {args.warmup_epochs} epochs (only source domain, no adversarial training)')
    else:
        log_string('No warmup phase: starting domain adaptation training immediately')

    root_source = args.data_root_source
    root_target = args.data_root_target
    
    # 创建标签映射
    exclude_classes = args.exclude_classes if args.exclude_classes else []
    label_mapping, new_classes, new_class2label, new_seg_label_to_cat = create_label_mapping(exclude_classes)
    
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

    print("start loading source domain data ...")
    log_string(f"Data loading mode: {'cache in memory' if args.cache_data else 'load from disk on-the-fly'}")
    train_transform = t.Compose([
        t.RandomRotate(),
        t.RandomFlip(p=0.5),
        t.RandomJitter(sigma=0.01, clip=0.05)
    ])
    
    # 源域数据集（有标签）
    SRC_TRAIN_DATASET_RAW = S3DISDatasetTrans(split='train', data_root=root_source, num_point=NUM_POINT, 
                                               test_area=args.test_area, sample_rate=1.0, shuffle_index=True, 
                                               transform=train_transform, cache=args.cache_data)
    SRC_TRAIN_DATASET = LabelMappingDataset(SRC_TRAIN_DATASET_RAW, label_mapping)
    
    # 目标域数据集（无标签）
    TAR_TRAIN_DATASET_RAW = S3DISDatasetTrans(split='train', data_root=root_target, num_point=NUM_POINT,
                                               test_area=args.test_area, sample_rate=1.0, shuffle_index=True,
                                               transform=train_transform, cache=args.cache_data)
    # 目标域没有标签，但为了兼容性，我们创建一个包装器
    class TargetDataset:
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            coord, _ = self.dataset[idx]
            return coord, None  # 目标域没有标签
    
    TAR_TRAIN_DATASET = TargetDataset(TAR_TRAIN_DATASET_RAW)
    
    # 目标域的collate函数（没有标签）
    def collate_fn_target(batch):
        coord, _ = list(zip(*batch))
        offset, count = [], 0
        for item in coord:
            count += item.shape[0]
            offset.append(count)
        return torch.cat(coord), torch.IntTensor(offset)
    
    # 测试数据集（源域）
    TEST_DATASET_RAW = S3DISDatasetTrans(split='test', data_root=root_source, num_point=NUM_POINT,
                                          test_area=args.test_area, sample_rate=1.0, shuffle_index=False,
                                          transform=None, cache=args.cache_data)
    TEST_DATASET = LabelMappingDataset(TEST_DATASET_RAW, label_mapping)

    src_trainDataLoader = torch.utils.data.DataLoader(SRC_TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                                                      num_workers=5, pin_memory=False, drop_last=True,
                                                      worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
                                                      collate_fn=collate_fn)
    
    tar_trainDataLoader = torch.utils.data.DataLoader(TAR_TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                                                     num_workers=5, pin_memory=False, drop_last=True,
                                                     worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
                                                     collate_fn=collate_fn_target)
    
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=5, pin_memory=False, collate_fn=collate_fn)
    
    # 计算类别权重
    if exclude_classes:
        original_weights = SRC_TRAIN_DATASET_RAW.labelweights
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

    # 创建生成器和判别器
    backbone = MODEL.get_model(NUM_CLASSES).cuda()
    G = Generator(backbone, NUM_CLASSES).cuda()
    D = Discriminator(NUM_CLASSES).cuda()

    # 优化器
    optimizer_G = torch.optim.Adam(
        filter(lambda p: p.requires_grad, G.parameters()),
        lr=args.lr_G,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    
    optimizer_D = torch.optim.Adam(
        filter(lambda p: p.requires_grad, D.parameters()),
        lr=args.lr_D,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    # 尝试加载检查点
    start_epoch = 0
    prototypes = None
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'prototypes' in checkpoint:
            prototypes = checkpoint['prototypes']
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')

    # 计算初始原型
    if prototypes is None:
        log_string('Computing initial prototypes...')
        prototypes = compute_prototypes(G, src_trainDataLoader, NUM_CLASSES, 'cuda')
        log_string('Prototypes computed.')

    best_iou = 0
    max_iter = args.epoch * len(src_trainDataLoader)
    is_warmup = args.warmup_epochs > 0

    for epoch in range(start_epoch, args.epoch):
        # 判断是否在预热阶段
        in_warmup = is_warmup and epoch < args.warmup_epochs
        
        if in_warmup:
            log_string('**** WARMUP Epoch %d (%d/%d) ****' % (epoch + 1, epoch + 1, args.warmup_epochs))
        else:
            # 如果是预热阶段结束后的第一个epoch，添加提示
            if is_warmup and epoch == args.warmup_epochs:
                log_string('=' * 50)
                log_string('WARMUP PHASE COMPLETED! Starting domain adaptation training...')
                log_string('=' * 50)
            log_string('**** Epoch %d (%d/%s) ****' % (epoch + 1, epoch + 1, args.epoch))
        
        # Poly学习率衰减
        current_iter = epoch * len(src_trainDataLoader)
        lr_G = poly_lr_scheduler(optimizer_G, args.lr_G, current_iter, max_iter, power=0.9)
        if not in_warmup:
            lr_D = poly_lr_scheduler(optimizer_D, args.lr_D, current_iter, max_iter, power=0.9)
            log_string(f'Learning rate G: {lr_G:.6f}, D: {lr_D:.6f}')
        else:
            log_string(f'Learning rate G: {lr_G:.6f} (WARMUP: D not trained)')
        
        # 每隔N个epoch更新原型
        if epoch % args.update_prototypes_every == 0 and epoch > 0:
            log_string('Updating prototypes...')
            prototypes = compute_prototypes(G, src_trainDataLoader, NUM_CLASSES, 'cuda')
            log_string('Prototypes updated.')

        G.train()
        if not in_warmup:
            D.train()
        
        # 训练循环
        src_iter = iter(src_trainDataLoader)
        if not in_warmup:
            tar_iter = iter(tar_trainDataLoader)
            num_batches = min(len(src_trainDataLoader), len(tar_trainDataLoader))
        else:
            num_batches = len(src_trainDataLoader)
        
        total_loss_G = 0
        total_loss_D = 0
        
        loop = tqdm(range(num_batches), desc='train' + (' (warmup)' if in_warmup else ''))
        
        for i in loop:
            try:
                coord_s, target_s, offset_s = next(src_iter)
            except StopIteration:
                src_iter = iter(src_trainDataLoader)
                coord_s, target_s, offset_s = next(src_iter)
            
            if not in_warmup:
                try:
                    coord_t, offset_t = next(tar_iter)
                except StopIteration:
                    tar_iter = iter(tar_trainDataLoader)
                    coord_t, offset_t = next(tar_iter)
            
            # 移动到GPU
            coord_s = coord_s.cuda(non_blocking=True)
            target_s = target_s.cuda(non_blocking=True)
            offset_s = offset_s.cuda(non_blocking=True)
            
            # ------------------
            #  训练生成器 G
            # ------------------
            optimizer_G.zero_grad()
            
            # 源域前向传播
            F_s, P_s_m = G([coord_s, coord_s, offset_s])
            target_s_flat = target_s.view(-1)
            
            if in_warmup:
                # 预热阶段：只使用源域数据，只计算分割损失和源域点到原型损失
                loss_seg = compute_loss_seg(P_s_m, target_s_flat, weights)
                loss_ppd_s = compute_loss_ppd_source(F_s, target_s_flat, prototypes)
                loss_G = loss_seg + args.gamma_3 * loss_ppd_s
                
                loss_G.backward()
                optimizer_G.step()
                
                total_loss_G += loss_G.item()
                
                loop.set_postfix(
                    loss_G=f'{loss_G.item():.4f}',
                    loss_seg=f'{loss_seg.item():.4f}',
                    loss_ppd_s=f'{loss_ppd_s.item():.4f}'
                )
            else:
                # 正常训练阶段：使用源域和目标域数据
                coord_t = coord_t.cuda(non_blocking=True)
                offset_t = offset_t.cuda(non_blocking=True)
                
                # 目标域前向传播
                F_t, P_t_m = G([coord_t, coord_t, offset_t])
                
                # 计算辅助分类器概率
                P_t_a = compute_auxiliary_classifier(F_t, prototypes)
                
                # 计算G的所有损失
                loss_seg = compute_loss_seg(P_s_m, target_s_flat, weights)
                loss_adv = compute_loss_adv_adaptive(P_t_m, P_t_a, D, args.gamma_local, args.delta)
                loss_ppd_s = compute_loss_ppd_source(F_s, target_s_flat, prototypes)
                loss_ppd_t = compute_loss_ppd_target(F_t, P_t_m, prototypes, args.th_p, args.th_e)
                
                loss_G = loss_seg + args.gamma_2 * loss_adv + args.gamma_3 * (loss_ppd_s + loss_ppd_t)
                
                loss_G.backward()
                optimizer_G.step()
                
                # ------------------
                #  训练判别器 D
                # ------------------
                optimizer_D.zero_grad()
                
                # 使用之前计算的概率，但需要detach以防止梯度传回G
                S_s = compute_self_info(P_s_m.detach())
                S_t = compute_self_info(P_t_m.detach())
                
                loss_D = compute_loss_discriminator(S_s, S_t, D)
                
                loss_D.backward()
                optimizer_D.step()
                
                total_loss_G += loss_G.item()
                total_loss_D += loss_D.item()
                
                loop.set_postfix(
                    loss_G=f'{loss_G.item():.4f}',
                    loss_D=f'{loss_D.item():.4f}',
                    loss_seg=f'{loss_seg.item():.4f}',
                    loss_adv=f'{loss_adv.item():.4f}'
                )
        
        log_string(f'Training mean loss G: {total_loss_G / num_batches:.4f}')
        if not in_warmup:
            log_string(f'Training mean loss D: {total_loss_D / num_batches:.4f}')
        
        # 保存模型
        if epoch % 5 == 0:
            log_string('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            # 确保 model_state_dict 包含所有backbone参数，供测试脚本使用
            backbone_state_dict = G.backbone.state_dict()
            state = {
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'model_state_dict': backbone_state_dict,  # 保存backbone权重供测试脚本使用
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'prototypes': prototypes,
                'num_classes': NUM_CLASSES,  # 保存类别数，便于测试脚本验证
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        
        torch.cuda.empty_cache()
        
        # 评估（每个epoch都评估）
        with torch.no_grad():
            G.eval()
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

            log_string('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            for i, (coord, target, offset) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                coord = coord.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                offset = offset.cuda(non_blocking=True)

                _, P_m = G([coord, coord, offset])
                pred_val = P_m.contiguous().cpu().data.numpy()
                pred_val = np.argmax(pred_val, 1)

                batch_label = target.view(-1).cpu().data.numpy()
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                
                if NUM_POINT < 0:
                    total_seen += len(batch_label)
                else:
                    total_seen += (BATCH_SIZE * NUM_POINT)
                
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (13 + 1 - len(seg_label_to_cat[l])), labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)

            if mIoU >= best_iou:
                best_iou = mIoU
                log_string('Save best model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                # 确保 model_state_dict 包含所有backbone参数，供测试脚本使用
                backbone_state_dict = G.backbone.state_dict()
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'G_state_dict': G.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'model_state_dict': backbone_state_dict,  # 保存backbone权重供测试脚本使用
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'prototypes': prototypes,
                    'num_classes': NUM_CLASSES,  # 保存类别数，便于测试脚本验证
                }
                torch.save(state, savepath)
                log_string('Saving best model....')
            log_string('Best mIoU: %f' % best_iou)
        
        torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    main(args)

