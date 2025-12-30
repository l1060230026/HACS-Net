"""
t-SNE Feature Distribution Visualization for Domain Adaptation
对比 Source-only 模型和 HACS-Net 模型的特征分布
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from tqdm import tqdm
import importlib
import importlib.util

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

from data_utils.S3DISDataLoader import S3DISDatasetTrans
from data_utils.data_util import collate_fn
from train_att import LabelMappingDataset, create_label_mapping


def load_model(model_path, model_name, num_classes, device, model_file_path=None):
    """
    加载模型权重
    Args:
        model_path: checkpoint 路径
        model_name: 模型名称 ('pointnet2_sem_seg_att' 或 'pointnet2_att_hierarchical_gan')
        num_classes: 类别数
        device: 设备
        model_file_path: 模型文件路径（可选，如果不在 models 目录）
    Returns:
        model: 加载好的模型
    """
    # 动态导入模型
    if model_file_path and os.path.exists(model_file_path):
        # 从指定路径导入
        spec = importlib.util.spec_from_file_location(model_name, model_file_path)
        MODEL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MODEL)
    else:
        # 从 models 目录导入
        try:
            MODEL = importlib.import_module(f'models.{model_name}')
        except:
            # 尝试直接导入
            MODEL = importlib.import_module(model_name)
    
    # 构建模型 - 尝试不同的参数组合
    # 先尝试只使用 num_classes（更通用）
    try:
        model = MODEL.get_model(num_classes).to(device)
    except TypeError:
        # 如果失败，尝试带 n_discriminator_levels 参数
        try:
            model = MODEL.get_model(num_classes, n_discriminator_levels=3).to(device)
        except Exception as e:
            print(f"Error building model with both parameter combinations: {e}")
            raise
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同的 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'generator_state_dict' in checkpoint:
        state_dict = checkpoint['generator_state_dict']
        # 移除 discriminator 相关的键
        state_dict = {k: v for k, v in state_dict.items() if 'discriminators' not in k}
    else:
        state_dict = checkpoint
    
    # 加载权重（允许部分缺失）
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"  - {key}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f"  - {key}")
    
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, dataloader, device, max_samples=2000, layer='high_level', pool='mean'):
    """
    从数据中提取特征
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        max_samples: 最大样本数
        layer: 要提取的特征层 ('high_level' 表示高层特征)
        pool: 聚合方式 ('mean' 或 'max')
    Returns:
        features: (M, D) 特征矩阵
        labels: (M,) 标签（如果有）
        domain_labels: (M,) 域标签（'source' 或 'target'）
    """
    features_list = []
    labels_list = []
    domain_labels_list = []
    
    sample_count = 0
    
    for batch_idx, (coord, label, offset) in enumerate(tqdm(dataloader, desc="Extracting features")):
        if sample_count >= max_samples:
            break
        
        coord = coord.to(device)
        offset = offset.to(device)
        
        # 检查输入数据是否包含 NaN/Inf（诊断用）
        if batch_idx == 0 and sample_count == 0:
            if torch.isnan(coord).any() or torch.isinf(coord).any():
                print(f"Warning: Input coordinates contain NaN/Inf!")
                print(f"  - NaN count: {torch.isnan(coord).sum().item()}")
                print(f"  - Inf count: {torch.isinf(coord).sum().item()}")
        
        # 模型前向传播
        # 输入格式: [coord, coord, offset]
        try:
            # 检查模型是否支持 return_features 参数
            import inspect
            sig = inspect.signature(model.forward)
            if 'return_features' in sig.parameters:
                # HACS-Net 模型，支持 return_features
                output = model([coord, coord, offset], return_features=True)
                if isinstance(output, tuple) and len(output) == 3:
                    # HACS-Net 返回 (logits, multi_features, trans_feat)
                    _, multi_features, trans_feat = output
                    # 使用最高层的特征（通常是最后一个，256-dim）

                    feat = trans_feat  # 备用：使用 trans_feat (N, 512)
                elif isinstance(output, tuple) and len(output) == 2:
                    # 可能返回 (logits, features)
                    _, feat = output
                else:
                    feat = output
            else:
                # Source-only 模型，标准输出
                output = model([coord, coord, offset])
                if isinstance(output, tuple):
                    _, feat = output  # (logits, trans_feat)
                else:
                    feat = output
        except Exception as e:
            print(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # feat 形状: (N, D)，其中 N 是所有点的总数，D 是特征维度
        # 需要按 batch 分割并聚合
        
        # 检查模型输出的特征是否包含 NaN/Inf（诊断用）
        if torch.isnan(feat).any() or torch.isinf(feat).any():
            nan_count = torch.isnan(feat).sum().item()
            inf_count = torch.isinf(feat).sum().item()
            print(f"Warning [Batch {batch_idx}]: Model output contains NaN/Inf!")
            print(f"  - NaN count: {nan_count}/{feat.numel()} ({100*nan_count/feat.numel():.2f}%)")
            print(f"  - Inf count: {inf_count}/{feat.numel()} ({100*inf_count/feat.numel():.2f}%)")
            print(f"  - Feature shape: {feat.shape}")
            print(f"  - Feature range: [{feat[~torch.isnan(feat) & ~torch.isinf(feat)].min():.4f}, "
                  f"{feat[~torch.isnan(feat) & ~torch.isinf(feat)].max():.4f}]")
            # 用 0 填充 NaN/Inf（临时解决方案，用于继续运行）
            feat = torch.where(torch.isnan(feat) | torch.isinf(feat), 
                              torch.zeros_like(feat), feat)
            print(f"  - Replaced NaN/Inf with zeros. Continuing...")
        
        # 将点级特征聚合成样本级特征
        batch_features = []
        batch_labels = []
        
        for i in range(offset.shape[0]):
            if i == 0:
                start_idx = 0
            else:
                start_idx = offset[i-1].item()
            end_idx = offset[i].item()
            
            # 提取当前样本的特征
            sample_feat = feat[start_idx:end_idx]  # (N_i, D)
            
            # 跳过空样本
            if sample_feat.shape[0] == 0:
                print(f"Warning: Empty sample at batch {batch_idx}, sample {i}. Skipping...")
                continue
            
            # 聚合点级特征到样本级
            if pool == 'mean':
                sample_feat_vec = sample_feat.mean(dim=0)  # (D,)
            elif pool == 'max':
                sample_feat_vec = sample_feat.max(dim=0)[0]  # (D,)
            else:
                raise ValueError(f"Unknown pool method: {pool}")
            
            # 检查是否包含 NaN 或 Inf
            if torch.isnan(sample_feat_vec).any() or torch.isinf(sample_feat_vec).any():
                print(f"Warning: NaN/Inf in sample at batch {batch_idx}, sample {i}. Skipping...")
                continue
            
            batch_features.append(sample_feat_vec.cpu().numpy())
            
            # 提取标签（如果有）
            sample_label = label[start_idx:end_idx]  # (N_i,)
            # 使用最常见的标签作为样本标签
            unique_labels, counts = torch.unique(sample_label, return_counts=True)
            most_common_label = unique_labels[counts.argmax()].item()
            batch_labels.append(most_common_label)
        
        features_list.extend(batch_features)
        labels_list.extend(batch_labels)
        
        sample_count += len(batch_features)
    
    features = np.array(features_list)  # (M, D)
    labels = np.array(labels_list)  # (M,)
    
    return features, labels


def plot_tsne_comparison(features_before, features_after, domain_labels, output_path, 
                        title_before="Before Adaptation (Source-only)", 
                        title_after="After Adaptation (HACS-Net)"):
    """
    绘制对比 t-SNE 图
    Args:
        features_before: (M, 2) Before adaptation 的 t-SNE 结果
        features_after: (M, 2) After adaptation 的 t-SNE 结果
        domain_labels: (M,) 域标签数组 ('source' 或 'target')
        output_path: 输出路径
        title_before: 左图标题
        title_after: 右图标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 准备颜色
    source_color = '#1f77b4'  # 蓝色
    target_color = '#d62728'  # 红色
    
    # 转换为 numpy 数组
    domain_labels = np.array(domain_labels)
    source_mask = domain_labels == 'source'
    target_mask = domain_labels == 'target'
    
    # 左图：Before Adaptation
    ax1 = axes[0]
    ax1.scatter(features_before[source_mask, 0], features_before[source_mask, 1], 
               c=source_color, label='Source', alpha=0.6, s=20, edgecolors='none')
    ax1.scatter(features_before[target_mask, 0], features_before[target_mask, 1], 
               c=target_color, label='Target', alpha=0.6, s=20, edgecolors='none')
    ax1.set_title(title_before, fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 右图：After Adaptation
    ax2 = axes[1]
    ax2.scatter(features_after[source_mask, 0], features_after[source_mask, 1], 
               c=source_color, label='Source', alpha=0.6, s=20, edgecolors='none')
    ax2.scatter(features_after[target_mask, 0], features_after[target_mask, 1], 
               c=target_color, label='Target', alpha=0.6, s=20, edgecolors='none')
    ax2.set_title(title_after, fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 添加整体标题
    fig.suptitle('t-SNE Visualization of Feature Distributions', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser('t-SNE Feature Visualization')
    parser.add_argument('--source_data_root', type=str, default='data/bim_scan/', 
                       help='Source domain data root [default: data/bim_scan/]')
    parser.add_argument('--target_data_root', type=str, default='data/stanford_indoor3d/', 
                       help='Target domain data root [default: data/stanford_indoor3d/]')
    parser.add_argument('--model_before', type=str, 
                       default='log/sem_seg/pointnet2_att/checkpoints/best_model.pth',
                       help='Path to source-only model checkpoint')
    parser.add_argument('--model_after', type=str,
                       default='log/sem_seg/att_hierarchical_gan/checkpoints/best_model.pth',
                       help='Path to HACS-Net model checkpoint')
    parser.add_argument('--model_name_before', type=str, default='pointnet2_sem_seg_att',
                       help='Model name for source-only [default: pointnet2_sem_seg_att]')
    parser.add_argument('--model_name_after', type=str, default='pointnet2_att_hierarchical_gan',
                       help='Model name for HACS-Net [default: pointnet2_att_hierarchical_gan]')
    parser.add_argument('--model_file_before', type=str, default=None,
                       help='Path to source-only model file (optional)')
    parser.add_argument('--model_file_after', type=str, default=None,
                       help='Path to HACS-Net model file (optional)')
    parser.add_argument('--output_dir', type=str, default='outputs/tsne',
                       help='Output directory [default: outputs/tsne]')
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='Maximum number of samples per domain [default: 2000]')
    parser.add_argument('--num_point', type=int, default=40000,
                       help='Number of points per sample [default: 40000]')
    parser.add_argument('--test_area', type=int, default=5,
                       help='Test area [default: 5]')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size [default: 4]')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU to use [default: 0]')
    parser.add_argument('--exclude_classes', type=str, nargs='+', default=['board', 'clutter'],
                       help='Classes to exclude [default: board clutter]')
    parser.add_argument('--perplexity', type=float, default=30.0,
                       help='t-SNE perplexity [default: 30.0]')
    parser.add_argument('--pool', type=str, default='mean', choices=['mean', 'max'],
                       help='Feature pooling method [default: mean]')
    
    args = parser.parse_args()
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建标签映射
    exclude_classes = args.exclude_classes if args.exclude_classes else []
    label_mapping, new_classes, new_class2label, new_seg_label_to_cat = create_label_mapping(exclude_classes)
    NUM_CLASSES = len(new_classes)
    
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {new_classes}")
    
    # 加载数据
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    # Source domain
    print("Loading source domain data...")
    SOURCE_DATASET_RAW = S3DISDatasetTrans(
        split='train', 
        data_root=args.source_data_root, 
        num_point=args.num_point,
        test_area=args.test_area, 
        sample_rate=1.0, 
        shuffle_index=True, 
        transform=None,  # 不使用数据增强
        cache=False
    )
    SOURCE_DATASET = LabelMappingDataset(SOURCE_DATASET_RAW, label_mapping)
    source_loader = torch.utils.data.DataLoader(
        SOURCE_DATASET, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False, 
        collate_fn=collate_fn
    )
    
    # Target domain
    print("Loading target domain data...")
    TARGET_DATASET_RAW = S3DISDatasetTrans(
        split='train', 
        data_root=args.target_data_root, 
        num_point=args.num_point,
        test_area=args.test_area, 
        sample_rate=1.0, 
        shuffle_index=True, 
        transform=None,  # 不使用数据增强
        cache=False
    )
    # Target domain 可能没有标签，但我们仍然包装它
    try:
        TARGET_DATASET = LabelMappingDataset(TARGET_DATASET_RAW, label_mapping)
    except:
        TARGET_DATASET = TARGET_DATASET_RAW
    
    target_loader = torch.utils.data.DataLoader(
        TARGET_DATASET, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False, 
        collate_fn=collate_fn
    )
    
    # 合并 source 和 target 数据加载器
    def combined_loader(source_loader, target_loader, max_samples):
        """合并 source 和 target 数据加载器"""
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        domain_labels = []
        
        count = 0
        while count < max_samples:
            try:
                # 交替获取 source 和 target
                if count % 2 == 0:
                    batch = next(source_iter)
                    domain = 'source'
                else:
                    batch = next(target_iter)
                    domain = 'target'
                
                batch_size = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else len(batch[0])
                domain_labels.extend([domain] * batch_size)
                count += batch_size
                
                yield batch, domain
            except StopIteration:
                break
    
    # 加载模型
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)
    
    # Before Adaptation (Source-only)
    print(f"Loading source-only model from: {args.model_before}")
    model_before = load_model(
        args.model_before, 
        args.model_name_before, 
        NUM_CLASSES, 
        device,
        model_file_path=args.model_file_before
    )
    
    # After Adaptation (HACS-Net)
    print(f"Loading HACS-Net model from: {args.model_after}")
    model_after = load_model(
        args.model_after, 
        args.model_name_after, 
        NUM_CLASSES, 
        device,
        model_file_path=args.model_file_after
    )
    
    # 提取特征
    print("\n" + "="*60)
    print("Extracting features...")
    print("="*60)
    
    # Before Adaptation
    print("\nExtracting features from source-only model...")
    features_before_list = []
    domain_labels_list = []
    
    # Source domain features
    print("  - Source domain...")
    feat_src, _ = extract_features(model_before, source_loader, device, 
                                   max_samples=args.max_samples // 2, pool=args.pool)
    features_before_list.append(feat_src)
    domain_labels_list.extend(['source'] * feat_src.shape[0])
    
    # Target domain features
    print("  - Target domain...")
    feat_tgt, _ = extract_features(model_before, target_loader, device, 
                                   max_samples=args.max_samples // 2, pool=args.pool)
    features_before_list.append(feat_tgt)
    domain_labels_list.extend(['target'] * feat_tgt.shape[0])
    
    features_before = np.concatenate(features_before_list, axis=0)
    print(f"  Total features shape: {features_before.shape}")
    
    # After Adaptation
    print("\nExtracting features from HACS-Net model...")
    features_after_list = []
    
    # Source domain features
    print("  - Source domain...")
    feat_src, _ = extract_features(model_after, source_loader, device, 
                                   max_samples=args.max_samples // 2, pool=args.pool)
    features_after_list.append(feat_src)
    
    # Target domain features
    print("  - Target domain...")
    feat_tgt, _ = extract_features(model_after, target_loader, device, 
                                   max_samples=args.max_samples // 2, pool=args.pool)
    features_after_list.append(feat_tgt)
    
    features_after = np.concatenate(features_after_list, axis=0)
    print(f"  Total features shape: {features_after.shape}")
    
    # 确保两个特征矩阵的样本数一致
    min_samples = min(features_before.shape[0], features_after.shape[0])
    features_before = features_before[:min_samples]
    features_after = features_after[:min_samples]
    domain_labels_list = domain_labels_list[:min_samples]
    
    print(f"\nFinal feature shapes: Before={features_before.shape}, After={features_after.shape}")
    print(f"Domain distribution: Source={domain_labels_list.count('source')}, Target={domain_labels_list.count('target')}")
    
    # 检查并处理 NaN 值
    print("\n" + "="*60)
    print("Checking for NaN values...")
    print("="*60)
    
    # 找出包含 NaN 的行（在任一特征矩阵中）
    nan_mask_before = np.isnan(features_before).any(axis=1)
    nan_mask_after = np.isnan(features_after).any(axis=1)
    nan_mask = nan_mask_before | nan_mask_after
    
    if nan_mask.any():
        n_nan = nan_mask.sum()
        print(f"Warning: Found {n_nan} samples with NaN values. Removing them...")
        print(f"  - NaN in features_before: {nan_mask_before.sum()}")
        print(f"  - NaN in features_after: {nan_mask_after.sum()}")
        
        # 移除包含 NaN 的行
        features_before = features_before[~nan_mask]
        features_after = features_after[~nan_mask]
        domain_labels_list = [label for i, label in enumerate(domain_labels_list) if not nan_mask[i]]
        
        print(f"After removal: Before={features_before.shape}, After={features_after.shape}")
        print(f"Remaining samples: {len(domain_labels_list)}")
    else:
        print("No NaN values found. All samples are valid.")
    
    # 再次确保两个特征矩阵的样本数一致
    min_samples = min(features_before.shape[0], features_after.shape[0])
    if features_before.shape[0] != features_after.shape[0]:
        print(f"Warning: Feature matrices have different sizes. Truncating to {min_samples} samples.")
        features_before = features_before[:min_samples]
        features_after = features_after[:min_samples]
        domain_labels_list = domain_labels_list[:min_samples]
    
    # t-SNE 降维
    print("\n" + "="*60)
    print("Running t-SNE...")
    print("="*60)
    
    print("Computing t-SNE for Before Adaptation...")
    tsne_before = TSNE(
        n_components=2, 
        perplexity=args.perplexity, 
        learning_rate='auto',
        init='pca', 
        random_state=42,
        n_iter=1000,
        verbose=1
    )
    features_before_2d = tsne_before.fit_transform(features_before)
    
    print("Computing t-SNE for After Adaptation...")
    tsne_after = TSNE(
        n_components=2, 
        perplexity=args.perplexity, 
        learning_rate='auto',
        init='pca', 
        random_state=42,
        n_iter=1000,
        verbose=1
    )
    features_after_2d = tsne_after.fit_transform(features_after)
    
    # 绘制对比图
    print("\n" + "="*60)
    print("Generating visualization...")
    print("="*60)
    
    output_path = output_dir / 'tsne_comparison.png'
    plot_tsne_comparison(
        features_before_2d, 
        features_after_2d, 
        domain_labels_list,
        output_path,
        title_before="(a) Before Adaptation (Source-only)",
        title_after="(b) After Adaptation (HACS-Net)"
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print(f"Visualization saved to: {output_path}")


if __name__ == '__main__':
    main()

