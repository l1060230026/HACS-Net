"""
Stage 1: PCAN (Prototype-based Category Adversarial Network) 训练器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from network.generator import Generator
from network.discriminator_out import DiscriminatorOut, GANLoss
from utils.prototype_estimator import PrototypeEstimator
from utils.pseudo_label_generator import generate_pseudo_labels, compute_category_adv_loss
from utils.ema_model import create_ema_model, update_ema_variables


class TrainerPCAN:
    """Stage 1 PCAN 训练器"""
    def __init__(self, G, D, Teacher_G, src_loader, tgt_loader, 
                 G_optimizer, D_optimizer, criterion, config, device='cuda'):
        """
        Args:
            G: Generator模型
            D: Discriminator模型
            Teacher_G: Teacher模型（EMA）
            src_loader: 源域数据加载器
            tgt_loader: 目标域数据加载器
            G_optimizer: Generator优化器
            D_optimizer: Discriminator优化器
            criterion: 分割损失函数
            config: 配置字典
            device: 设备
        """
        self.G = G
        self.D = D
        self.Teacher_G = Teacher_G
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        
        # 配置参数
        self.lambda_adv = config.get('lambda_adv', 0.001)
        self.lambda_cal = config.get('lambda_cal_adv', 1.0)
        self.pseudo_start_iter = config.get('pseudo_start_iter', 5000)
        self.cal_start_iter = config.get('cal_start_iter', 5000)
        self.use_category_adv = config.get('category_adv', True)
        self.use_mean_teacher = config.get('use_mt', True)
        self.alpha_ema = config.get('alpha_ema', 0.9999)
        self.proto_update_period = config.get('proto_update_period', 20000)
        self.proto_update_domain = config.get('proto_update_domain', 'src')
        
        # GAN损失
        self.gan_loss = GANLoss(gan_mode=config.get('gan_mode', 'ls_gan'))
        
        # 原型估计器
        self.prototypes = PrototypeEstimator(
            num_classes=G.num_classes,
            feature_dim=G.feature_dim,
            update_mode=config.get('proto_update_mode', 'mean'),
            device=device
        )
        
        # 初始化Teacher模型
        if self.use_mean_teacher:
            create_ema_model(self.Teacher_G, self.G)
    
    def train_iteration(self, iteration, src_data, tgt_data):
        """
        执行一次训练迭代
        
        Args:
            iteration: 当前迭代次数
            src_data: 源域数据 (coord, label, offset)
            tgt_data: 目标域数据 (coord, label, offset)
        
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # ========== 1. 训练Generator ==========
        self.G_optimizer.zero_grad()
        
        # 1.1 源域监督损失
        src_coord, src_label, src_offset = src_data
        src_coord = src_coord.to(self.device)
        src_label = src_label.to(self.device)
        src_offset = src_offset.to(self.device)
        
        src_logits, src_features = self.G([src_coord, src_coord, src_offset], is_train=True)
        src_logits_flat = src_logits.view(-1, self.G.num_classes)
        src_label_flat = src_label.view(-1)
        
        # 获取权重（从config中获取）
        weights = self.config.get('weights', None)
        src_loss = self.criterion(src_logits_flat, src_label_flat, None, weights)
        losses['src_loss'] = src_loss.item()
        
        # 1.2 目标域对抗损失
        tgt_coord, tgt_label, tgt_offset = tgt_data
        tgt_coord = tgt_coord.to(self.device)
        tgt_offset = tgt_offset.to(self.device)
        
        tgt_logits, tgt_features = self.G([tgt_coord, tgt_coord, tgt_offset], is_train=True)
        tgt_logits_flat = tgt_logits.view(-1, self.G.num_classes)
        
        # 生成伪标签
        tgt_pseudo_labels = None
        if iteration > self.pseudo_start_iter and self.use_mean_teacher:
            with torch.no_grad():
                tgt_teacher_logits, _ = self.Teacher_G([tgt_coord, tgt_coord, tgt_offset], is_train=False)
                tgt_teacher_logits_flat = tgt_teacher_logits.view(-1, self.G.num_classes)
                tgt_pseudo_labels = generate_pseudo_labels(
                    tgt_teacher_logits_flat,
                    method='entropy',
                    threshold=self.config.get('ent_threshold', 0.05)
                )
        
        # 对抗损失
        tgt_D_out = self.D(tgt_logits_flat)
        adv_loss = self.gan_loss(tgt_D_out, target_is_real=True)  # 让G欺骗D，认为目标域是源域
        
        # 类别对抗损失 (CAL)
        if iteration > self.cal_start_iter and self.use_category_adv and tgt_pseudo_labels is not None:
            cal_adv_loss = compute_category_adv_loss(
                tgt_D_out,
                tgt_pseudo_labels,
                tgt_features.view(-1, self.G.feature_dim),
                self.prototypes.get_prototypes()
            )
            if isinstance(adv_loss, torch.Tensor):
                adv_loss = self.lambda_cal * cal_adv_loss + (1 - self.lambda_cal) * adv_loss.mean()
            else:
                adv_loss = self.lambda_cal * cal_adv_loss + (1 - self.lambda_cal) * adv_loss
        
        losses['adv_loss'] = adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss
        
        # 总损失
        if isinstance(adv_loss, torch.Tensor):
            G_loss = src_loss + self.lambda_adv * adv_loss
        else:
            G_loss = src_loss + self.lambda_adv * torch.tensor(adv_loss, device=self.device)
        G_loss.backward()
        self.G_optimizer.step()
        losses['G_loss'] = G_loss.item()
        
        # ========== 2. 训练Discriminator ==========
        self.D_optimizer.zero_grad()
        
        # 源域：标签为0 (源域)
        src_D_out = self.D(src_logits_flat.detach())
        src_D_loss = self.gan_loss(src_D_out, target_is_real=False)
        
        # 目标域：标签为1 (目标域)
        tgt_D_out_detached = self.D(tgt_logits_flat.detach())
        tgt_D_loss = self.gan_loss(tgt_D_out_detached, target_is_real=True)
        
        D_loss = (src_D_loss + tgt_D_loss) * 0.5
        D_loss.backward()
        self.D_optimizer.step()
        losses['D_loss'] = D_loss.item()
        
        # ========== 3. 更新原型 ==========
        if iteration % self.proto_update_period == 0:
            if self.proto_update_domain == 'src':
                self.prototypes.update(src_features.view(-1, self.G.feature_dim), src_label_flat)
            else:
                if tgt_pseudo_labels is not None:
                    self.prototypes.update(tgt_features.view(-1, self.G.feature_dim), tgt_pseudo_labels)
        
        # ========== 4. 更新Teacher模型 ==========
        if self.use_mean_teacher:
            update_ema_variables(self.Teacher_G, self.G, self.alpha_ema, iteration)
        
        return losses
    
    def train_epoch(self, epoch, max_iters):
        """训练一个epoch"""
        self.G.train()
        self.D.train()
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
            
            losses = self.train_iteration(iteration, src_data, tgt_data)
            
            # 累计损失
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value
            
            # 更新进度条
            if iteration % 10 == 0:
                pbar.set_postfix({k: f'{v/(iteration+1):.4f}' for k, v in total_losses.items()})
        
        # 平均损失
        avg_losses = {k: v / max_iters for k, v in total_losses.items()}
        return avg_losses

