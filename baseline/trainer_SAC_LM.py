"""
Stage 2: SAC_LM (Self-Adaptive Consistency + LaserMix) 训练器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from network.generator import Generator
from network.domain_mix import laserMix_batch
from utils.pseudo_label_generator import generate_pseudo_labels
from utils.ema_model import update_ema_variables
from data_utils import transform as t


class TrainerSACLM:
    """Stage 2 SAC_LM 训练器"""
    def __init__(self, G, Teacher_G, src_loader, tgt_loader,
                 G_optimizer, criterion, config, device='cuda'):
        """
        Args:
            G: Generator模型
            Teacher_G: Teacher模型（EMA）
            src_loader: 源域数据加载器
            tgt_loader: 目标域数据加载器
            G_optimizer: Generator优化器
            criterion: 分割损失函数
            config: 配置字典
            device: 设备
        """
        self.G = G
        self.Teacher_G = Teacher_G
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.G_optimizer = G_optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        
        # 配置参数
        self.lambda_sac = config.get('lambda_sac', 0.1)
        self.use_mean_teacher = config.get('use_mt', True)
        self.alpha_ema = config.get('alpha_ema', 0.99)
        self.update_every = config.get('update_every', 100)
        self.pseudo_threshold = config.get('pseudo_threshold', 0.9)
        
        # 显存优化参数
        self.use_amp = config.get('use_amp', True)  # 混合精度训练
        self.accum_steps = config.get('accum_steps', 1)  # 梯度累积步数
        self.scaler = GradScaler() if self.use_amp else None
        
        # SAC损失使用的数据增强（用于目标域增强）
        self.sac_augmentation = t.Compose([
            t.RandomRotate(),
            t.RandomFlip(p=0.5),
            t.RandomJitter(sigma=0.01, clip=0.05)
        ])
    
    def train_iteration(self, iteration, src_data, tgt_data, is_accum_start=False, is_accum_end=False):
        """
        执行一次训练迭代
        
        Args:
            iteration: 当前迭代次数
            src_data: 源域数据 (coord, label, offset)
            tgt_data: 目标域数据 (coord, label, offset)
            is_accum_start: 是否为梯度累积的开始
            is_accum_end: 是否为梯度累积的结束
        
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # ========== 1. 生成伪标签 ==========
        tgt_coord, tgt_label, tgt_offset = tgt_data
        tgt_coord = tgt_coord.to(self.device)
        tgt_offset = tgt_offset.to(self.device)
        
        # 生成伪标签，同时保存teacher logits用于后续SAC损失计算
        # Teacher在原始目标域上的预测（用于SAC损失）
        with torch.no_grad():
            tgt_teacher_logits, _ = self.Teacher_G([tgt_coord, tgt_coord, tgt_offset], is_train=False)
            tgt_teacher_logits_flat = tgt_teacher_logits.view(-1, self.G.num_classes)
            tgt_pseudo_labels = generate_pseudo_labels(
                tgt_teacher_logits_flat,
                method='confidence',
                threshold=self.pseudo_threshold
            )
            # 调试信息：检查伪标签范围
            if iteration % 100 == 0:
                pseudo_min = tgt_pseudo_labels.min().item()
                pseudo_max = tgt_pseudo_labels.max().item()
                valid_count = (tgt_pseudo_labels >= 0).sum().item()
                invalid_count = (tgt_pseudo_labels < 0).sum().item()
                out_of_range = ((tgt_pseudo_labels >= 0) & (tgt_pseudo_labels >= self.G.num_classes)).sum().item()
                if out_of_range > 0 or pseudo_min < -1 or pseudo_max >= self.G.num_classes:
                    print(f"[DEBUG] Iter {iteration}: Pseudo labels - min={pseudo_min}, max={pseudo_max}, "
                          f"valid={valid_count}, invalid={invalid_count}, out_of_range={out_of_range}, "
                          f"num_classes={self.G.num_classes}")
            # 确保伪标签在有效范围内
            tgt_pseudo_labels = torch.clamp(tgt_pseudo_labels, -1, self.G.num_classes - 1)
            # 保留teacher logits用于SAC损失
            tgt_teacher_logits_raw_flat = tgt_teacher_logits_flat.clone()
            del tgt_teacher_logits, tgt_teacher_logits_flat  # 及时释放
        
        # 为SAC损失准备增强后的目标域（Student在增强目标域上预测）
        # 先将目标域坐标移到CPU进行numpy操作（用于后续LaserMix和增强）
        tgt_coord_cpu = tgt_coord.cpu().numpy()
        
        # 生成增强后的目标域用于SAC损失
        tgt_coord_augmented_list = []
        tgt_start = 0
        for i in range(len(tgt_offset)):
            tgt_end = tgt_offset[i].item()
            tgt_coord_batch = tgt_coord_cpu[tgt_start:tgt_end]
            # 应用数据增强（旋转、翻转、抖动）
            tgt_coord_augmented_batch, _ = self.sac_augmentation(tgt_coord_batch, np.zeros(len(tgt_coord_batch), dtype=np.int64))
            tgt_coord_augmented_list.append(torch.from_numpy(tgt_coord_augmented_batch).float())
            tgt_start = tgt_end
        tgt_coord_augmented = torch.cat(tgt_coord_augmented_list, dim=0).to(self.device)
        del tgt_coord_augmented_list  # 释放内存
        
        # ========== 2. LaserMix域混合 ==========
        src_coord, src_label, src_offset = src_data
        # 将数据移到CPU进行numpy操作以节省GPU显存
        src_coord_np = src_coord.numpy()
        src_label_np = src_label.numpy()
        # 使用之前保存的tgt_coord_cpu用于LaserMix（原始版本）
        tgt_coord_np = tgt_coord_cpu
        tgt_pseudo_labels_np = tgt_pseudo_labels.cpu().numpy()
        
        # 释放GPU上的tgt_coord（已经转换为numpy用于LaserMix，增强版本已保存在tgt_coord_augmented）
        del tgt_coord  # 释放GPU显存
        
        # 按batch处理LaserMix
        mixed_coords = []
        mixed_labels = []
        src_masks = []
        tgt_masks = []
        batch_sizes = []  # 记录每个batch的大小，用于计算offset
        
        # 处理每个batch
        src_start = 0
        tgt_start = 0
        for i in range(len(src_offset)):
                src_end = src_offset[i].item()
                tgt_end = tgt_offset[i].item()
                
                src_coord_batch = src_coord_np[src_start:src_end]
                src_label_batch = src_label_np[src_start:src_end]
                tgt_coord_batch = tgt_coord_np[tgt_start:tgt_end]
                tgt_pseudo_batch = tgt_pseudo_labels_np[tgt_start:tgt_end]
                
                # 调试信息：检查输入标签范围
                if iteration % 100 == 0 and i == 0:
                    src_label_min, src_label_max = src_label_batch.min(), src_label_batch.max()
                    tgt_pseudo_min, tgt_pseudo_max = tgt_pseudo_batch.min(), tgt_pseudo_batch.max()
                    print(f"[DEBUG] Iter {iteration} Batch {i}: Input labels - "
                          f"src: min={src_label_min}, max={src_label_max}, "
                          f"tgt_pseudo: min={tgt_pseudo_min}, max={tgt_pseudo_max}, "
                          f"num_classes={self.G.num_classes}")
                
                # 确保输入标签在有效范围内
                src_label_batch = np.clip(src_label_batch.astype(np.int64), -1, self.G.num_classes - 1)
                tgt_pseudo_batch = np.clip(tgt_pseudo_batch.astype(np.int64), -1, self.G.num_classes - 1)
                
                # LaserMix
                mc, ml, sm, tm = laserMix_batch(
                    [src_coord_batch], [src_label_batch],
                    [tgt_coord_batch], [tgt_pseudo_batch]
                )
                
                # 确保标签值在有效范围内，防止索引越界
                ml_array = ml[0].copy()  # 复制以避免修改原始数据
                # 将超出范围的标签设置为-1（无效标签）
                # 使用clip确保值在[-1, num_classes-1]范围内，并转换为int64
                ml_array = np.clip(ml_array.astype(np.int64), -1, self.G.num_classes - 1)
                
                # 调试信息：检查LaserMix后的标签范围
                if iteration % 100 == 0 and i == 0:
                    ml_min, ml_max = ml_array.min(), ml_array.max()
                    out_of_range = ((ml_array >= 0) & (ml_array >= self.G.num_classes)).sum()
                    if out_of_range > 0 or ml_min < -1 or ml_max >= self.G.num_classes:
                        print(f"[WARNING] Iter {iteration} Batch {i}: LaserMix output labels out of range! "
                              f"min={ml_min}, max={ml_max}, out_of_range={out_of_range}, num_classes={self.G.num_classes}")
                
                # 记录batch大小
                batch_size = len(mc[0])
                batch_sizes.append(batch_size)
                
                mixed_coords.append(torch.from_numpy(mc[0]).float())
                mixed_labels.append(torch.from_numpy(ml_array).long())
                src_masks.append(torch.from_numpy(sm[0]).bool())
                tgt_masks.append(torch.from_numpy(tm[0]).bool())
                
                src_start = src_end
                tgt_start = tgt_end
        
        # 释放原始numpy数组以节省内存
        del src_coord_np, src_label_np, tgt_coord_np, tgt_pseudo_labels_np, tgt_coord_cpu
        
        # 合并所有batch
        mixed_coord = torch.cat(mixed_coords, dim=0).to(self.device)
        mixed_label = torch.cat(mixed_labels, dim=0).to(self.device)
        src_mask = torch.cat(src_masks, dim=0).to(self.device)
        tgt_mask = torch.cat(tgt_masks, dim=0).to(self.device)
        
        # 释放列表中的tensor以节省显存
        del mixed_coords, mixed_labels, src_masks, tgt_masks
        
        # 确保标签值在有效范围内，将超出范围的标签设置为-1（无效标签）
        # 只处理超出范围的标签，保留正常范围内的标签
        invalid_mask = (mixed_label < -1) | (mixed_label >= self.G.num_classes)
        if invalid_mask.sum() > 0:
            if iteration % 100 == 0:
                print(f"[WARNING] Iter {iteration}: Found {invalid_mask.sum().item()} invalid labels after merge! "
                      f"min={mixed_label.min().item()}, max={mixed_label.max().item()}, "
                      f"num_classes={self.G.num_classes}")
        mixed_label[invalid_mask] = -1
        
        # 最终安全检查：强制clamp所有标签
        mixed_label = torch.clamp(mixed_label, -1, self.G.num_classes - 1)
        
        # 计算offset - 使用IntTensor确保类型为int32（不是int64）
        # batch_sizes是每个batch的点数，需要累积求和得到offset
        mixed_offset_list = []
        count = 0
        for size in batch_sizes:
            count += size
            mixed_offset_list.append(count)
        mixed_offset = torch.IntTensor(mixed_offset_list).to(self.device)
        
        # 验证offset是否正确
        if len(mixed_offset) > 0 and mixed_offset[-1].item() != mixed_coord.shape[0]:
            print(f"[ERROR] Iter {iteration}: Offset mismatch! mixed_offset[-1]={mixed_offset[-1].item()}, mixed_coord.shape[0]={mixed_coord.shape[0]}")
            # 修复offset
            mixed_offset = torch.IntTensor([mixed_coord.shape[0]]).to(self.device)
        
        del batch_sizes, mixed_offset_list  # 释放内存
        
        # ========== 3. 训练Generator ==========
        # 梯度累积：只在累积开始时清零梯度
        if is_accum_start:
            self.G_optimizer.zero_grad(set_to_none=True)
        
        # 获取权重（从config中获取）
        weights = self.config.get('weights', None)
        
        # 使用混合精度训练
        with autocast(enabled=self.use_amp, dtype=torch.float16):
            # 3.1 源域损失 (使用混合后的源域部分)
            # 在模型调用前同步CUDA，确保之前的错误被捕获
            torch.cuda.synchronize()
            
            # 检查标签范围（在模型调用前）
            label_min = mixed_label.min().item()
            label_max = mixed_label.max().item()
            if label_min < -1 or label_max >= self.G.num_classes:
                print(f"[ERROR] Iter {iteration}: Labels out of range before model call!")
                print(f"  - label_min: {label_min}, label_max: {label_max}, num_classes: {self.G.num_classes}")
                print(f"  - invalid labels: {torch.unique(mixed_label[(mixed_label < -1) | (mixed_label >= self.G.num_classes)]).cpu().numpy()}")
                # 强制修复
                mixed_label = torch.clamp(mixed_label, -1, self.G.num_classes - 1)
            
            # 验证offset和coord的匹配性
            coord_total = mixed_coord.shape[0]
            if len(mixed_offset) > 0:
                offset_total = mixed_offset[-1].item()
                if offset_total != coord_total:
                    print(f"[ERROR] Iter {iteration}: Offset mismatch detected!")
                    print(f"  - mixed_coord.shape[0]: {coord_total}")
                    print(f"  - mixed_offset[-1]: {offset_total}")
                    print(f"  - mixed_offset: {mixed_offset.cpu().numpy()}")
                    # 修复offset
                    mixed_offset = torch.IntTensor([coord_total]).to(self.device)
                    print(f"  - Fixed offset to: {mixed_offset.cpu().numpy()}")
            
            # 验证offset的单调性和有效性
            if len(mixed_offset) > 1:
                offset_diff = torch.diff(mixed_offset)
                if (offset_diff <= 0).any():
                    print(f"[ERROR] Iter {iteration}: Invalid offset (not monotonic)!")
                    print(f"  - mixed_offset: {mixed_offset.cpu().numpy()}")
                    print(f"  - offset_diff: {offset_diff.cpu().numpy()}")
                    # 修复：确保offset单调递增
                    fixed_offsets = [mixed_offset[0].item()]
                    for i in range(1, len(mixed_offset)):
                        if mixed_offset[i].item() <= fixed_offsets[-1]:
                            fixed_offsets.append(fixed_offsets[-1] + 1)
                        else:
                            fixed_offsets.append(mixed_offset[i].item())
                    mixed_offset = torch.IntTensor(fixed_offsets).to(self.device)
                    print(f"  - Fixed offset to: {mixed_offset.cpu().numpy()}")
            
            try:
                mixed_logits, _ = self.G([mixed_coord, mixed_coord, mixed_offset], is_train=True)
                # 立即同步CUDA，捕获模型内部的错误
                torch.cuda.synchronize()
            except RuntimeError as model_e:
                # 捕获到错误，打印详细信息
                print(f"[ERROR] Iter {iteration}: Model forward failed!")
                print(f"  - mixed_coord shape: {mixed_coord.shape}, dtype: {mixed_coord.dtype}")
                print(f"  - mixed_coord range: [{mixed_coord.min().item():.3f}, {mixed_coord.max().item():.3f}]")
                print(f"  - mixed_offset: {mixed_offset.cpu().numpy()}")
                print(f"  - mixed_offset[-1]: {mixed_offset[-1].item() if len(mixed_offset) > 0 else 'N/A'}")
                print(f"  - mixed_coord.shape[0]: {mixed_coord.shape[0]}")
                print(f"  - offset matches coord: {mixed_offset[-1].item() == mixed_coord.shape[0] if len(mixed_offset) > 0 else 'N/A'}")
                print(f"  - mixed_label shape: {mixed_label.shape}")
                print(f"  - mixed_label range: [{label_min}, {label_max}], num_classes: {self.G.num_classes}")
                invalid_mask = (mixed_label < -1) | (mixed_label >= self.G.num_classes)
                if invalid_mask.sum() > 0:
                    print(f"  - Invalid labels count: {invalid_mask.sum().item()}")
                    print(f"  - Invalid label values: {torch.unique(mixed_label[invalid_mask]).cpu().numpy()}")
                raise
            
            # 检查输出是否包含NaN
            if torch.isnan(mixed_logits).any():
                # 如果输出包含NaN，跳过这个batch
                return {'src_loss': 0.0, 'tgt_pse_loss': 0.0, 'sac_loss': 0.0, 'total_loss': 0.0}
            
            mixed_logits_flat = mixed_logits.view(-1, self.G.num_classes)
            mixed_label_flat = mixed_label.view(-1)
            
            src_mask_flat = src_mask.view(-1)
            tgt_mask_flat = tgt_mask.view(-1)
            # 只使用有效的标签（标签值 >= 0 且在有效范围内）
            valid_label_mask = (mixed_label_flat >= 0) & (mixed_label_flat < self.G.num_classes)
            src_loss_mask = src_mask_flat & valid_label_mask
            tgt_loss_mask = tgt_mask_flat & valid_label_mask
            
            # 检测标签值范围（调试用）
            try:
                # 最终安全检查：再次确保标签在有效范围内
                mixed_label_flat = torch.clamp(mixed_label_flat, -1, self.G.num_classes - 1)
                
                if src_loss_mask.sum() > 0:
                    src_labels = mixed_label_flat[src_loss_mask]
                    # 只使用有效标签（>=0且<num_classes）
                    src_valid_mask = (src_labels >= 0) & (src_labels < self.G.num_classes)
                    if src_valid_mask.sum() == 0:
                        src_loss = torch.tensor(0.0, device=self.device)
                    else:
                        if (src_labels < 0).any() or (src_labels >= self.G.num_classes).any():
                            print(f"[ERROR] Iter {iteration}: Source labels out of range! "
                                  f"min={src_labels.min().item()}, max={src_labels.max().item()}, "
                                  f"num_classes={self.G.num_classes}, valid_count={src_valid_mask.sum().item()}")
                            # 只使用有效标签
                            src_labels_valid = src_labels[src_valid_mask]
                            src_logits_valid = mixed_logits_flat[src_loss_mask][src_valid_mask]
                            src_loss = self.criterion(
                                src_logits_valid,
                                src_labels_valid,
                                None, weights
                            )
                        else:
                            src_loss = self.criterion(
                                mixed_logits_flat[src_loss_mask],
                                mixed_label_flat[src_loss_mask],
                                None, weights
                            )
                else:
                    src_loss = torch.tensor(0.0, device=self.device)
                
                losses['src_loss'] = src_loss.item()
                
                # 3.2 目标域伪标签损失 (使用混合后的目标域部分)
                
                if tgt_loss_mask.sum() > 0:
                    tgt_labels = mixed_label_flat[tgt_loss_mask]
                    # 只使用有效标签（>=0且<num_classes）
                    tgt_valid_mask = (tgt_labels >= 0) & (tgt_labels < self.G.num_classes)
                    if tgt_valid_mask.sum() == 0:
                        tgt_pse_loss = torch.tensor(0.0, device=self.device)
                    else:
                        if (tgt_labels < 0).any() or (tgt_labels >= self.G.num_classes).any():
                            print(f"[ERROR] Iter {iteration}: Target labels out of range! "
                                  f"min={tgt_labels.min().item()}, max={tgt_labels.max().item()}, "
                                  f"num_classes={self.G.num_classes}, valid_count={tgt_valid_mask.sum().item()}")
                            # 只使用有效标签
                            tgt_labels_valid = tgt_labels[tgt_valid_mask]
                            tgt_logits_valid = mixed_logits_flat[tgt_loss_mask][tgt_valid_mask]
                            tgt_pse_loss = self.criterion(
                                tgt_logits_valid,
                                tgt_labels_valid,
                                None, weights
                            )
                        else:
                            tgt_pse_loss = self.criterion(
                                mixed_logits_flat[tgt_loss_mask],
                                mixed_label_flat[tgt_loss_mask],
                                None, weights
                            )
                else:
                    tgt_pse_loss = torch.tensor(0.0, device=self.device)
            except RuntimeError as e:
                if "index" in str(e).lower() or "out of bounds" in str(e).lower():
                    print(f"[ERROR] Iter {iteration}: Index out of bounds in loss calculation!")
                    print(f"[ERROR] mixed_label_flat shape: {mixed_label_flat.shape}")
                    print(f"[ERROR] mixed_label_flat min: {mixed_label_flat.min().item()}, max: {mixed_label_flat.max().item()}")
                    print(f"[ERROR] num_classes: {self.G.num_classes}")
                    print(f"[ERROR] src_loss_mask sum: {src_loss_mask.sum().item()}, tgt_loss_mask sum: {tgt_loss_mask.sum().item()}")
                    if src_loss_mask.sum() > 0:
                        src_labels = mixed_label_flat[src_loss_mask]
                        print(f"[ERROR] src_labels min: {src_labels.min().item()}, max: {src_labels.max().item()}")
                        print(f"[ERROR] src_labels unique: {torch.unique(src_labels).cpu().numpy()}")
                    if tgt_loss_mask.sum() > 0:
                        tgt_labels = mixed_label_flat[tgt_loss_mask]
                        print(f"[ERROR] tgt_labels min: {tgt_labels.min().item()}, max: {tgt_labels.max().item()}")
                        print(f"[ERROR] tgt_labels unique: {torch.unique(tgt_labels).cpu().numpy()}")
                    print(f"[ERROR] Full error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                raise
            
            losses['tgt_pse_loss'] = tgt_pse_loss.item()
            
            # 立即释放混合损失相关的tensor以节省显存
            del mixed_logits, mixed_logits_flat, mixed_label_flat
            del src_loss_mask, tgt_loss_mask, valid_label_mask, src_mask_flat, tgt_mask_flat
            
            # 3.3 SAC损失 (Self-Adaptive Consistency)
            # 按照文档要求：Teacher在原始目标域上的预测 vs Student在增强目标域上的预测
            sac_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_sac > 0 and tgt_teacher_logits_raw_flat is not None:
                # Student在增强目标域上的预测
                tgt_student_logits_aug, _ = self.G([tgt_coord_augmented, tgt_coord_augmented, tgt_offset], is_train=True)
                tgt_student_logits_aug_flat = tgt_student_logits_aug.view(-1, self.G.num_classes)
                
                if tgt_student_logits_aug_flat.shape[0] > 0 and tgt_teacher_logits_raw_flat.shape[0] > 0:
                    # 对齐长度（取较小的）
                    min_len = min(tgt_student_logits_aug_flat.shape[0], tgt_teacher_logits_raw_flat.shape[0])
                    tgt_student_logits_aug_aligned = tgt_student_logits_aug_flat[:min_len]
                    tgt_teacher_logits_raw_aligned = tgt_teacher_logits_raw_flat[:min_len]
                    
                    # KL散度损失（使用detach避免保留teacher的计算图）
                    sac_loss = F.kl_div(
                        F.log_softmax(tgt_student_logits_aug_aligned, dim=1),
                        F.softmax(tgt_teacher_logits_raw_aligned.detach(), dim=1),
                        reduction='mean'
                    )
                    # 立即释放对齐后的tensor
                    del tgt_student_logits_aug_aligned, tgt_teacher_logits_raw_aligned
                # 立即释放student logits和teacher logits
                del tgt_student_logits_aug, tgt_student_logits_aug_flat
                # 释放teacher logits（不再需要）
                del tgt_teacher_logits_raw_flat
            
            losses['sac_loss'] = sac_loss.item()
            
            # 总损失（除以累积步数以保持有效损失大小）
            total_loss = (src_loss + tgt_pse_loss + self.lambda_sac * sac_loss) / self.accum_steps
        
            # 检查损失是否为NaN
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                return {'src_loss': 0.0, 'tgt_pse_loss': 0.0, 'sac_loss': 0.0, 'total_loss': 0.0}
            
            # 使用scaler进行反向传播（如果启用AMP）
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # 梯度累积：只在累积结束时更新参数
            if is_accum_end:
                # 梯度裁剪防止梯度爆炸
                if self.use_amp:
                    self.scaler.unscale_(self.G_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
                    self.scaler.step(self.G_optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
                    self.G_optimizer.step()
            
            losses['total_loss'] = total_loss.item() * self.accum_steps  # 恢复原始损失值用于记录
            
            # 立即释放不需要的变量以节省显存
            del mixed_coord, mixed_label, mixed_offset, src_mask, tgt_mask
            if 'tgt_coord_augmented' in locals():
                del tgt_coord_augmented
            del total_loss, src_loss, tgt_pse_loss, sac_loss
            
        # ========== 4. 更新Teacher模型 ==========
        if self.use_mean_teacher and iteration % self.update_every == 0:
            update_ema_variables(self.Teacher_G, self.G, self.alpha_ema, iteration)
        
        return losses
    
    def train_epoch(self, epoch, max_iters):
        """训练一个epoch"""
        self.G.train()
        if self.use_mean_teacher:
            self.Teacher_G.eval()
        
        src_iter = iter(self.src_loader)
        tgt_iter = iter(self.tgt_loader)
        
        total_losses = {}
        pbar = tqdm(range(max_iters), desc=f'Epoch {epoch}')
        
        for iteration in pbar:
            try:
                src_data = next(src_iter)
            except StopIteration:
                src_iter = iter(self.src_loader)
                src_data = next(src_iter)
            
            try:
                tgt_data = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(self.tgt_loader)
                tgt_data = next(tgt_iter)
            
            # 梯度累积逻辑
            is_accum_start = (iteration % self.accum_steps == 0)
            is_accum_end = ((iteration + 1) % self.accum_steps == 0) or (iteration == max_iters - 1)
            
            try:
                losses = self.train_iteration(iteration, src_data, tgt_data, is_accum_start, is_accum_end)
                
                # 累计损失
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value
                
                # 释放数据引用
                del src_data, tgt_data, losses
            except Exception as e:
                # 捕获异常并打印关键信息
                print(f"\n[ERROR] Iter {iteration} (Epoch {epoch}) failed: {type(e).__name__}: {str(e)}")
                # 只在CUDA错误时打印详细信息
                if "CUDA" in str(e) or "assert" in str(e).lower():
                    try:
                        src_coord, src_label, src_offset = src_data
                        tgt_coord, tgt_label, tgt_offset = tgt_data
                        print(f"  - num_classes: {self.G.num_classes}")
                        print(f"  - src_label range: [{src_label.min().item()}, {src_label.max().item()}]")
                        print(f"  - tgt_label range: [{tgt_label.min().item()}, {tgt_label.max().item()}]")
                    except:
                        pass
                import traceback
                traceback.print_exc()
                raise
            
            # 更频繁地清理显存缓存
            if iteration % 5 == 0:
                torch.cuda.empty_cache()
                pbar.set_postfix({k: f'{v/(iteration+1):.4f}' for k, v in total_losses.items()})
        
        # 平均损失
        avg_losses = {k: v / max_iters for k, v in total_losses.items()}
        return avg_losses

