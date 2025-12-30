"""
DGT-ST 主训练脚本
两阶段训练：Stage 1 (PCAN) + Stage 2 (SAC_LM)
"""
import argparse
import os
import sys
import importlib
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import logging
import numpy as np
import time
from tqdm import tqdm

from data_utils.S3DISDataLoader import S3DISDatasetTrans
from data_utils import transform as t
from data_utils.data_util import collate_fn
from models.pointnet2_sem_seg_att import get_model, get_loss
from network.generator import Generator
from network.discriminator_out import DiscriminatorOut
from trainer_PCAN import TrainerPCAN
from trainer_SAC_LM import TrainerSACLM


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


class LabelMappingDataset:
    """包装数据集，应用标签映射并过滤被排除的类别"""
    def __init__(self, dataset, label_mapping, num_classes=None):
        self.dataset = dataset
        self.label_mapping = torch.from_numpy(label_mapping).long()
        # 计算有效类别数（映射后不为-1的类别数）
        if num_classes is None:
            valid_mapping = self.label_mapping[self.label_mapping >= 0]
            if len(valid_mapping) > 0:
                self.num_classes = valid_mapping.max().item() + 1
            else:
                self.num_classes = 0
        else:
            self.num_classes = num_classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        coord, label = self.dataset[idx]
        # 应用标签映射
        label_mapped = self.label_mapping[label]
        # 确保映射后的标签在有效范围内（-1表示无效，>=0表示有效类别）
        # label_mapping已经确保值在[-1, num_classes-1]范围内
        # 创建掩码：保留标签不为-1的点
        mask = label_mapped >= 0
        coord_filtered = coord[mask]
        label_filtered = label_mapped[mask]
        # 最终安全检查：确保标签值在有效范围内
        if len(label_filtered) > 0:
            label_min, label_max = label_filtered.min().item(), label_filtered.max().item()
            if label_min < 0 or (self.num_classes > 0 and label_max >= self.num_classes):
                print(f"[WARNING] LabelMappingDataset: Filtered labels out of range! "
                      f"min={label_min}, max={label_max}, num_classes={self.num_classes}")
                # 强制clamp到有效范围
                if self.num_classes > 0:
                    label_filtered = torch.clamp(label_filtered, 0, self.num_classes - 1)
        return coord_filtered, label_filtered


def create_label_mapping(exclude_classes):
    """创建标签映射"""
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


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def evaluate_model(model, data_loader, criterion, weights, num_classes, log_string):
    """评估模型性能"""
    model.eval()
    num_batches = len(data_loader)
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(num_classes)]
    total_correct_class = [0 for _ in range(num_classes)]
    total_iou_deno_class = [0 for _ in range(num_classes)]
    
    with torch.no_grad():
        for i, (coord, target, offset) in enumerate(tqdm(data_loader, desc='Evaluating', leave=False)):
            coord, target, offset = coord.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
            
            seg_pred, trans_feat = model([coord, coord, offset])
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            
            batch_label = target.cpu().data.numpy()
            target_flat = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target_flat, trans_feat, weights)
            loss_sum += loss.item()
            
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += len(batch_label.flatten())
            
            for l in range(num_classes):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
    
    mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
    accuracy = total_correct / float(total_seen) if total_seen > 0 else 0.0
    mean_loss = loss_sum / float(num_batches) if num_batches > 0 else 0.0
    
    model.train()
    return mIoU, accuracy, mean_loss


def parse_args():
    parser = argparse.ArgumentParser('DGT-ST Training')
    parser.add_argument('--stage', type=str, default='stage_2', choices=['stage_1', 'stage_2', 'both'],
                       help='Training stage: stage_1, stage_2, or both')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_att', help='Model name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--log_dir', type=str, default='DGT_ST', help='Log directory')
    parser.add_argument('--npoint', type=int, default=40000, help='Point number')
    parser.add_argument('--test_area', type=int, default=5, help='Test area')
    parser.add_argument('--src_data_root', type=str, default='data/bim_indoor3d', help='Source data root')
    parser.add_argument('--tgt_data_root', type=str, default='data/stanford_indoor3d', help='Target data root')
    parser.add_argument('--exclude_classes', type=str, nargs='+', default=['board', 'clutter'],
                       help='Classes to exclude')
    parser.add_argument('--cache_data', action='store_true', default=False, help='Cache data in memory')
    parser.add_argument('--stage1_checkpoint', type=str, default=None, help='Stage 1 checkpoint path')
    parser.add_argument('--max_iters_stage1', type=int, default=10000, help='Max iterations for stage 1')
    parser.add_argument('--max_iters_stage2', type=int, default=10000, help='Max iterations for stage 2')
    parser.add_argument('--eval_freq', type=int, default=10, help='Evaluation frequency (epochs)')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision training (AMP) to save memory')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false', help='Disable mixed precision training')
    parser.add_argument('--accum_steps', type=int, default=2, help='Gradient accumulation steps (increase to save more memory)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    def log_string(str):
        logger.info(str)
        print(str)
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 创建目录
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
    
    # 日志
    logger = logging.getLogger("DGT-ST")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    # 创建标签映射
    exclude_classes = args.exclude_classes if args.exclude_classes else []
    label_mapping, new_classes, new_class2label, new_seg_label_to_cat = create_label_mapping(exclude_classes)
    
    global seg_label_to_cat
    seg_label_to_cat = new_seg_label_to_cat
    
    NUM_CLASSES = len(new_classes)
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    
    log_string(f"Training with {NUM_CLASSES} classes: {new_classes}")
    
    # 数据加载
    log_string("Loading data...")
    train_transform = t.Compose([
        t.RandomRotate(),
        t.RandomFlip(p=0.5),
        t.RandomJitter(sigma=0.01, clip=0.05)
    ])
    
    # 源域数据
    SRC_DATASET_RAW = S3DISDatasetTrans(
        split='train', data_root=args.src_data_root, num_point=NUM_POINT,
        test_area=args.test_area, sample_rate=1.0, shuffle_index=True,
        transform=train_transform, cache=args.cache_data
    )
    SRC_DATASET = LabelMappingDataset(SRC_DATASET_RAW, label_mapping, num_classes=NUM_CLASSES)
    
    # 目标域数据
    TGT_DATASET_RAW = S3DISDatasetTrans(
        split='train', data_root=args.tgt_data_root, num_point=NUM_POINT,
        test_area=args.test_area, sample_rate=1.0, shuffle_index=True,
        transform=train_transform, cache=args.cache_data
    )
    TGT_DATASET = LabelMappingDataset(TGT_DATASET_RAW, label_mapping, num_classes=NUM_CLASSES)
    
    src_loader = torch.utils.data.DataLoader(
        SRC_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=False, drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
        collate_fn=collate_fn
    )
    tgt_loader = torch.utils.data.DataLoader(
        TGT_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=False, drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
        collate_fn=collate_fn
    )
    
    # 测试数据（用于验证）
    TEST_DATASET_RAW = S3DISDatasetTrans(
        split='test', data_root=args.src_data_root, num_point=NUM_POINT,
        test_area=args.test_area, sample_rate=1.0, shuffle_index=False,
        transform=None, cache=args.cache_data
    )
    TEST_DATASET = LabelMappingDataset(TEST_DATASET_RAW, label_mapping, num_classes=NUM_CLASSES)
    test_loader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=False, collate_fn=collate_fn
    )
    
    log_string(f"Source dataset size: {len(SRC_DATASET)}")
    log_string(f"Target dataset size: {len(TGT_DATASET)}")
    log_string(f"Test dataset size: {len(TEST_DATASET)}")
    
    # 计算类别权重
    if exclude_classes:
        original_weights = SRC_DATASET_RAW.labelweights
        keep_indices = [class2label[cls] for cls in new_classes]
        weights = torch.Tensor(original_weights[keep_indices]).cuda()
    else:
        weights = torch.Tensor(SRC_DATASET_RAW.labelweights).cuda()
    
    # 模型加载
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
    
    backbone = MODEL.get_model(NUM_CLASSES).cuda()
    backbone.apply(inplace_relu)
    criterion = MODEL.get_loss().cuda()
    
    # 创建Generator
    G = Generator(backbone, NUM_CLASSES, return_feature=True).cuda()
    
    # Stage 1 配置
    stage1_config = {
        'lambda_adv': 0.001,
        'lambda_cal_adv': 1.0,
        'pseudo_start_iter': 5000,
        'cal_start_iter': 5000,
        'category_adv': True,
        'use_mt': True,
        'alpha_ema': 0.9999,
        'proto_update_period': 20000,
        'proto_update_domain': 'src',
        'proto_update_mode': 'mean',
        'ent_threshold': 0.05,
        'gan_mode': 'ls_gan',
        'weights': weights
    }
    
    # Stage 2 配置
    stage2_config = {
        'lambda_sac': 0.1,
        'use_mt': True,
        'alpha_ema': 0.99,
        'update_every': 100,
        'pseudo_threshold': 0.9,
        'weights': weights,
        'use_amp': args.use_amp,  # 启用混合精度训练以节省显存
        'accum_steps': args.accum_steps,  # 梯度累积步数，可以增加以进一步节省显存
    }
    
    # ========== Stage 1: PCAN ==========
    if args.stage in ['stage_1', 'both']:
        log_string("=" * 50)
        log_string("Starting Stage 1: PCAN Training")
        log_string("=" * 50)
        
        # 创建Discriminator
        D = DiscriminatorOut(NUM_CLASSES).cuda()
        
        # 创建Teacher模型（需要独立的backbone副本，不能共享同一个backbone对象）
        teacher_backbone = MODEL.get_model(NUM_CLASSES).cuda()
        teacher_backbone.apply(inplace_relu)
        # 复制G的backbone权重到Teacher的backbone
        teacher_backbone.load_state_dict(backbone.state_dict())
        Teacher_G = Generator(teacher_backbone, NUM_CLASSES, return_feature=True).cuda()
        
        # 优化器
        G_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        D_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate * 0.1, betas=(0.9, 0.999))
        
        # 训练器
        trainer = TrainerPCAN(
            G, D, Teacher_G, src_loader, tgt_loader,
            G_optimizer, D_optimizer, criterion, stage1_config
        )
        
        # 训练循环
        max_iters = args.max_iters_stage1
        iters_per_epoch = min(len(src_loader), len(tgt_loader))
        num_epochs = max_iters // iters_per_epoch + 1
        best_iou = 0.0
        
        for epoch in range(num_epochs):
            if epoch * iters_per_epoch >= max_iters:
                break
            
            current_iters = min(iters_per_epoch, max_iters - epoch * iters_per_epoch)
            losses = trainer.train_epoch(epoch, current_iters)
            
            log_string(f'Epoch {epoch}: {losses}')
            
            # 定期评估并保存模型
            if (epoch + 1) % args.eval_freq == 0 or epoch == num_epochs - 1:
                log_string('---- EPOCH %03d EVALUATION (Stage 1) ----' % (epoch + 1))
                mIoU, accuracy, mean_loss = evaluate_model(G.backbone, test_loader, criterion, weights, NUM_CLASSES, log_string)
                log_string('eval mean loss: %f' % mean_loss)
                log_string('eval point avg class IoU: %f' % mIoU)
                log_string('eval point accuracy: %f' % accuracy)
                
                # 保存checkpoint
                checkpoint_path = checkpoints_dir / f'stage1_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'G_state_dict': G.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'Teacher_G_state_dict': Teacher_G.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                    'model_state_dict': G.backbone.state_dict(),  # 用于测试脚本
                }, checkpoint_path)
                log_string(f'Saved checkpoint: {checkpoint_path}')
                
                # 保存best_model
                if mIoU >= best_iou:
                    best_iou = mIoU
                    best_checkpoint = checkpoints_dir / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'class_avg_iou': mIoU,
                        'G_state_dict': G.state_dict(),
                        'D_state_dict': D.state_dict(),
                        'Teacher_G_state_dict': Teacher_G.state_dict(),
                        'G_optimizer_state_dict': G_optimizer.state_dict(),
                        'D_optimizer_state_dict': D_optimizer.state_dict(),
                        'model_state_dict': G.backbone.state_dict(),  # 用于测试脚本
                    }, best_checkpoint)
                    log_string(f'Saved best model (mIoU: {mIoU:.4f}): {best_checkpoint}')
                log_string('Best mIoU: %f' % best_iou)
        
        # 保存最终模型
        final_checkpoint = checkpoints_dir / 'stage1_final.pth'
        torch.save({
            'epoch': num_epochs - 1,
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'Teacher_G_state_dict': Teacher_G.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
            'model_state_dict': G.backbone.state_dict(),  # 用于测试脚本
        }, final_checkpoint)
        log_string(f'Saved final Stage 1 checkpoint: {final_checkpoint}')
        
        # 如果只运行 Stage 1，确保保存 best_model
        if args.stage == 'stage_1':
            if not (checkpoints_dir / 'best_model.pth').exists():
                log_string('Copying final model as best_model.pth')
                shutil.copy(final_checkpoint, checkpoints_dir / 'best_model.pth')
        
        # 加载Stage 1的checkpoint到Stage 2
        stage1_checkpoint_path = final_checkpoint
    
    # ========== Stage 2: SAC_LM ==========
    if args.stage in ['stage_2', 'both']:
        log_string("=" * 50)
        log_string("Starting Stage 2: SAC_LM Training")
        log_string("=" * 50)
        
        # 加载Stage 1的checkpoint
        if args.stage == 'stage_2':
            if args.stage1_checkpoint:
                stage1_checkpoint_path = args.stage1_checkpoint
            else:
                stage1_checkpoint_path = checkpoints_dir / 'stage1_final.pth'
        
        if stage1_checkpoint_path and os.path.exists(stage1_checkpoint_path):
            log_string(f'Loading Stage 1 checkpoint: {stage1_checkpoint_path}')
            checkpoint = torch.load(stage1_checkpoint_path)
            if 'G_state_dict' in checkpoint:
                G.load_state_dict(checkpoint['G_state_dict'])
            elif 'model_state_dict' in checkpoint:
                G.backbone.load_state_dict(checkpoint['model_state_dict'])
            log_string('Stage 1 checkpoint loaded')
        else:
            log_string('Warning: Stage 1 checkpoint not found, starting from scratch')
        
        # 创建Teacher模型（需要独立的backbone副本，不能共享同一个backbone对象）
        teacher_backbone = MODEL.get_model(NUM_CLASSES).cuda()
        teacher_backbone.apply(inplace_relu)
        # 复制G的backbone权重到Teacher的backbone
        teacher_backbone.load_state_dict(G.backbone.state_dict())
        Teacher_G = Generator(teacher_backbone, NUM_CLASSES, return_feature=True).cuda()
        if stage1_checkpoint_path and os.path.exists(stage1_checkpoint_path):
            checkpoint = torch.load(stage1_checkpoint_path)
            if 'Teacher_G_state_dict' in checkpoint:
                Teacher_G.load_state_dict(checkpoint['Teacher_G_state_dict'])
        
        # 优化器
        G_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate * 0.1, betas=(0.9, 0.999))
        
        # 训练器
        trainer = TrainerSACLM(
            G, Teacher_G, src_loader, tgt_loader,
            G_optimizer, criterion, stage2_config
        )
        
        # 训练循环
        max_iters = args.max_iters_stage2
        iters_per_epoch = min(len(src_loader), len(tgt_loader))
        num_epochs = max_iters // iters_per_epoch + 1
        best_iou = 0.0
        
        # 如果已有 best_model，加载其 mIoU
        best_model_path = checkpoints_dir / 'best_model.pth'
        if best_model_path.exists():
            try:
                best_checkpoint = torch.load(best_model_path)
                if 'class_avg_iou' in best_checkpoint:
                    best_iou = best_checkpoint['class_avg_iou']
                    log_string(f'Loaded previous best mIoU: {best_iou:.4f}')
            except:
                pass
        
        for epoch in range(num_epochs):
            if epoch * iters_per_epoch >= max_iters:
                break
            
            current_iters = min(iters_per_epoch, max_iters - epoch * iters_per_epoch)
            losses = trainer.train_epoch(epoch, current_iters)
            
            log_string(f'Epoch {epoch}: {losses}')
            
            # 定期评估并保存模型
            if (epoch + 1) % args.eval_freq == 0 or epoch == num_epochs - 1:
                log_string('---- EPOCH %03d EVALUATION (Stage 2) ----' % (epoch + 1))
                mIoU, accuracy, mean_loss = evaluate_model(G.backbone, test_loader, criterion, weights, NUM_CLASSES, log_string)
                log_string('eval mean loss: %f' % mean_loss)
                log_string('eval point avg class IoU: %f' % mIoU)
                log_string('eval point accuracy: %f' % accuracy)
                
                # 保存checkpoint
                checkpoint_path = checkpoints_dir / f'stage2_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'G_state_dict': G.state_dict(),
                    'Teacher_G_state_dict': Teacher_G.state_dict(),
                    'G_optimizer_state_dict': G_optimizer.state_dict(),
                    'model_state_dict': G.backbone.state_dict(),  # 用于测试脚本
                }, checkpoint_path)
                log_string(f'Saved checkpoint: {checkpoint_path}')
                
                # 保存best_model
                if mIoU >= best_iou:
                    best_iou = mIoU
                    best_checkpoint = checkpoints_dir / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'class_avg_iou': mIoU,
                        'G_state_dict': G.state_dict(),
                        'Teacher_G_state_dict': Teacher_G.state_dict(),
                        'G_optimizer_state_dict': G_optimizer.state_dict(),
                        'model_state_dict': G.backbone.state_dict(),  # 用于测试脚本
                    }, best_checkpoint)
                    log_string(f'Saved best model (mIoU: {mIoU:.4f}): {best_checkpoint}')
                else:
                    log_string(f'No improvement. Current mIoU: {mIoU:.4f}, Best mIoU: {best_iou:.4f}')
                log_string('Best mIoU: %f' % best_iou)
        
        # 保存最终模型（如果还没有 best_model）
        final_checkpoint = checkpoints_dir / 'stage2_final.pth'
        torch.save({
            'epoch': num_epochs - 1,
            'G_state_dict': G.state_dict(),
            'Teacher_G_state_dict': Teacher_G.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'model_state_dict': G.backbone.state_dict(),  # 用于测试脚本
        }, final_checkpoint)
        log_string(f'Saved final Stage 2 checkpoint: {final_checkpoint}')
        
        # 确保 best_model 存在
        if not best_model_path.exists():
            log_string('Copying final model as best_model.pth')
            shutil.copy(final_checkpoint, best_model_path)
    
    log_string("Training completed!")


if __name__ == '__main__':
    main()

