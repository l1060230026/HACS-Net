"""
DawNet: Domain Adaptation Network with Hybrid Attention and Whitening
Based on the paper: "Automated BIM-to-scan point cloud semantic segmentation using 
a domain adaptation network with hybrid attention and whitening"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import farthest_point_sample, index_points
import models.pointops as pointops
from torch.autograd import Function


# ==================== ZCA Whitening Module ====================
class ZCAWhitening(nn.Module):
    """
    ZCA (Zero-phase Component Analysis) Whitening operation
    Used for feature decorrelation and reducing feature redundancy between domains
    Improved version: Uses Newton-Schulz iteration method for enhanced numerical stability
    """
    def __init__(self, momentum=0.1, eps=1e-3, num_iters=5):
        super(ZCAWhitening, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.num_iters = num_iters  # Number of Newton-Schulz iterations
        self.register_buffer('running_mean', None)
        self.register_buffer('running_whitening_matrix', None)
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def _newton_schulz_sqrt_inv(self, cov, num_iters=5):
        """
        Compute inverse square root of covariance matrix using Newton-Schulz iteration
        This is a numerically stable method that does not require eigenvalue decomposition
        """
        C = cov.shape[0]
        
        # Normalize covariance matrix so its spectral norm is close to 1
        trace = torch.trace(cov)
        cov_normalized = cov / (trace + self.eps)
        
        # Initialize
        I = torch.eye(C, device=cov.device, dtype=cov.dtype)
        Y = cov_normalized
        Z = I.clone()
        
        # Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3I - Z_k * Y_k)
        #                    Z_{k+1} = 0.5 * (3I - Z_k * Y_k) * Z_k
        for _ in range(num_iters):
            T = torch.mm(Z, Y)
            Y_new = 0.5 * torch.mm(Y, 3 * I - T)
            Z_new = 0.5 * torch.mm(3 * I - T, Z)
            Y = Y_new
            Z = Z_new
        
        # Z is the inverse square root of cov_normalized, need to scale back to original scale
        whitening_matrix = Z / torch.sqrt(trace + self.eps)
        
        return whitening_matrix
    
    def _safe_whitening(self, x, mean, whitening_matrix):
        """Safely apply whitening transformation with nan detection"""
        x_centered = x - mean
        x_whitened = torch.mm(x_centered, whitening_matrix)
        
        # Detect nan/inf
        if torch.isnan(x_whitened).any() or torch.isinf(x_whitened).any():
            return x  # Return original features
        
        return x_whitened
        
    def forward(self, x):
        """
        Args:
            x: (N, C) 特征张量
        Returns:
            whitened features: (N, C)
        """
        # Check if input has nan/inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            return x
        
        N, C = x.shape
        
        # Skip whitening if sample count is too small
        if N < C * 2:
            return x
        
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=0, keepdim=True)  # (1, C)
            x_centered = x - mean
            
            # Compute covariance matrix with diagonal regularization
            cov = torch.mm(x_centered.t(), x_centered) / (N - 1)  # Use unbiased estimator
            cov = cov + self.eps * torch.eye(C, device=cov.device, dtype=cov.dtype)
            
            try:
                # Compute whitening matrix using Newton-Schulz iteration
                whitening_matrix = self._newton_schulz_sqrt_inv(cov, self.num_iters)
                
                # Check if whitening matrix is valid
                if torch.isnan(whitening_matrix).any() or torch.isinf(whitening_matrix).any():
                    return x
                
                # Update running statistics
                if self.running_mean is None:
                    self.running_mean = mean.squeeze(0).detach()
                    self.running_whitening_matrix = whitening_matrix.detach()
                else:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze(0).detach()
                    self.running_whitening_matrix = (1 - self.momentum) * self.running_whitening_matrix + self.momentum * whitening_matrix.detach()
                
                self.num_batches_tracked += 1
                
                # Apply whitening
                x_whitened = self._safe_whitening(x, mean, whitening_matrix)
                return x_whitened
                
            except Exception as e:
                # If whitening fails, return original features
                # print(f"Warning: ZCA whitening failed ({str(e)}), returning original features")
                return x
        else:
            # Use running statistics during testing
            if self.running_mean is None or self.running_whitening_matrix is None:
                return x
            
            try:
                mean = self.running_mean.unsqueeze(0)
                x_whitened = self._safe_whitening(x, mean, self.running_whitening_matrix)
                return x_whitened
            except Exception as e:
                # print(f"Warning: ZCA whitening failed in eval ({str(e)}), returning original features")
                return x


# ==================== Hybrid Attention Module ====================
class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (N, C) 特征张量
        Returns:
            attention weighted features: (N, C)
        """
        # Convert to (C, N) to use AdaptiveAvgPool1d
        x_t = x.t().unsqueeze(0)  # (1, C, N)
        
        avg_out = self.fc(self.avg_pool(x_t).squeeze(-1).squeeze(0))  # (C,)
        max_out = self.fc(self.max_pool(x_t).squeeze(-1).squeeze(0))  # (C,)
        
        attention = self.sigmoid(avg_out + max_out)  # (C,)
        return x * attention.unsqueeze(0)  # (N, C)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (N, C) 特征张量
        Returns:
            attention weighted features: (N, C)
        """
        # Compute statistics along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (N, 1)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (N, 1)
        
        # Concatenate and apply convolution
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (N, 2)
        x_cat = x_cat.t().unsqueeze(0)  # (1, 2, N)
        
        attention = self.sigmoid(self.conv(x_cat))  # (1, 1, N)
        attention = attention.squeeze(0).t()  # (N, 1)
        
        return x * attention  # (N, C)


class HybridAttention(nn.Module):
    """Hybrid attention module: combines channel attention and spatial attention"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(HybridAttention, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: (N, C) 特征张量
        Returns:
            attention weighted features: (N, C)
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ==================== Residual Block with Whitening ====================
class WhiteningResidualBlock(nn.Module):
    """Residual block with whitening operation"""
    def __init__(self, in_channels, out_channels, use_whitening=True):
        super(WhiteningResidualBlock, self).__init__()
        self.use_whitening = use_whitening
        
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        if use_whitening:
            self.whitening = ZCAWhitening()
        
        self.linear2 = nn.Linear(out_channels, out_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # If input and output dimensions differ, use 1x1 convolution for dimension matching
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: (N, C_in) 特征张量
        Returns:
            output: (N, C_out)
        """
        identity = self.shortcut(x)
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.use_whitening:
            out = self.whitening(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


# ==================== Gradient Reversal Layer ====================
class GradientReversalFunction(Function):
    """Gradient reversal layer for domain adversarial training"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = 1.0
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ==================== Domain Discriminator ====================
class DomainDiscriminator(nn.Module):
    """Domain discriminator network"""
    def __init__(self, input_dim, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim // 2, 2)  # Binary classification: source domain vs target domain
        )
    
    def forward(self, x):
        """
        Args:
            x: (N, C) 特征张量
        Returns:
            domain_output: (N, 2) 域分类logits
        """
        return self.discriminator(x)


# ==================== Point Transformer Components ====================
class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            if x1 is None:
                x = pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
            else:
                x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


# ==================== DawNet Main Model ====================
class DawNet(nn.Module):
    """
    DawNet: Domain Adaptation Network with Hybrid Attention and Whitening
    """
    def __init__(self, num_classes, use_whitening=True, use_attention=True):
        super(DawNet, self).__init__()
        self.in_planes = 3
        self.c = 3
        self.use_whitening = use_whitening
        self.use_attention = use_attention
        
        # Encoder part (based on PointTransformer)
        self.enc2 = self._make_enc(PointTransformerBlock, 64, 3, share_planes=8, stride=4, nsample=16)  # N/4
        self.enc3 = self._make_enc(PointTransformerBlock, 128, 3, share_planes=8, stride=4, nsample=16) # N/16
        self.enc4 = self._make_enc(PointTransformerBlock, 256, 3, share_planes=8, stride=4, nsample=16) # N/64
        self.enc5 = self._make_enc(PointTransformerBlock, 512, 3, share_planes=8, stride=4, nsample=16) # N/256

        # Decoder part with Whitening and Attention
        self.dec5 = self._make_dec(PointTransformerBlock, 512, 1, share_planes=8, nsample=16, is_head=True)
        self.dec4 = self._make_dec(PointTransformerBlock, 256, 1, share_planes=8, nsample=16)
        self.dec3 = self._make_dec(PointTransformerBlock, 128, 1, share_planes=8, nsample=16)
        self.dec2 = self._make_dec(PointTransformerBlock, 64, 1, share_planes=8, nsample=16)
        self.dec1 = self._make_dec(PointTransformerBlock, 32, 1, share_planes=8, nsample=8)

        # Whitening residual block
        if use_whitening:
            self.whitening_block = WhiteningResidualBlock(512, 512, use_whitening=True)
        
        # Hybrid attention module
        if use_attention:
            self.hybrid_attention = HybridAttention(512, reduction=16)
        
        # Classification head
        self.cls = nn.Sequential(
            nn.Linear(32, 32), 
            nn.BatchNorm1d(32), 
            nn.ReLU(inplace=True), 
            nn.Linear(32, num_classes)
        )
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer()
        
        # Domain discriminator
        self.domain_classifier = DomainDiscriminator(512, hidden_dim=256)

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
    
    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo, alpha=1.0):
        """
        Args:
            pxo: [p, x, o] where p is coordinates, x is features, o is offsets
            alpha: Lambda parameter for domain adversarial training
        Returns:
            seg_pred: Segmentation prediction (N, num_classes)
            domain_pred: Domain classification prediction (N, 2)
            feature: Feature vector (N, 512)
        """
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        
        # Encoder
        p1, x1, o1 = self.enc2([p0, x0, o0])
        p2, x2, o2 = self.enc3([p1, x1, o1])
        p3, x3, o3 = self.enc4([p2, x2, o2])
        p4, x4, o4 = self.enc5([p3, x3, o3])
        
        # Apply whitening residual block
        if self.use_whitening:
            x4 = self.whitening_block(x4)
        
        # Apply hybrid attention
        if self.use_attention:
            x4 = self.hybrid_attention(x4)
        
        # Save features for domain discrimination
        feature = x4
        
        # Decoder
        x4_dec = self.dec5[1:]([p4, self.dec5[0]([p4, x4, o4]), o4])[1]
        x3 = self.dec4[1:]([p3, self.dec4[0]([p3, x3, o3], [p4, x4_dec, o4]), o3])[1]
        x2 = self.dec3[1:]([p2, self.dec3[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec2[1:]([p1, self.dec2[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x0 = self.dec1[1:]([p0, self.dec1[0]([p0, None, o0], [p1, x1, o1]), o0])[1]
        
        # Semantic segmentation prediction
        seg_pred = self.cls(x0)
        seg_pred = F.log_softmax(seg_pred, dim=1)
        
        # Domain discrimination prediction (through gradient reversal layer)
        self.grl.set_lambda(alpha)
        reversed_feature = self.grl(feature)
        domain_pred = self.domain_classifier(reversed_feature)
        
        return seg_pred, domain_pred, feature


# ==================== Loss Function ====================
class DawNetLoss(nn.Module):
    """DawNet combined loss function"""
    def __init__(self, lambda_adv=1.0):
        super(DawNetLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.seg_criterion = nn.NLLLoss(ignore_index=-1)
        self.domain_criterion = nn.CrossEntropyLoss()
    
    def forward(self, seg_pred, seg_target, domain_pred, domain_target, weight=None):
        """
        Args:
            seg_pred: Segmentation prediction (N, num_classes) - log_softmax output
            seg_target: Segmentation labels (N,)
            domain_pred: Domain prediction (N, 2)
            domain_target: Domain labels (N,) - 0 for source domain, 1 for target domain
            weight: Class weights
        Returns:
            total_loss: Total loss
            seg_loss: Segmentation loss
            domain_loss: Domain discrimination loss
        """
        # Segmentation loss (only computed on labeled data)
        valid_mask = seg_target >= 0
        if valid_mask.sum() > 0:
            if weight is not None:
                seg_loss = F.nll_loss(seg_pred[valid_mask], seg_target[valid_mask], weight=weight)
            else:
                seg_loss = self.seg_criterion(seg_pred[valid_mask], seg_target[valid_mask])
        else:
            seg_loss = torch.tensor(0.0, device=seg_pred.device)
        
        # Domain discrimination loss
        domain_loss = self.domain_criterion(domain_pred, domain_target)
        
        # Total loss
        total_loss = seg_loss + self.lambda_adv * domain_loss
        
        return total_loss, seg_loss, domain_loss


def get_model(num_classes, use_whitening=True, use_attention=True):
    """Get DawNet model"""
    return DawNet(num_classes, use_whitening=use_whitening, use_attention=use_attention)


def get_loss(lambda_adv=1.0):
    """Get DawNet loss function"""
    return DawNetLoss(lambda_adv=lambda_adv)


if __name__ == '__main__':
    # Test code
    model = get_model(13).cuda()
    
    # 模拟输入
    batch_size = 2
    num_points = 4096
    p = torch.randn(batch_size * num_points, 3).cuda()
    x = p  # 使用坐标作为特征
    o = torch.cuda.IntTensor([num_points, num_points * 2])
    
    # 前向传播
    seg_pred, domain_pred, feature = model([p, x, o], alpha=1.0)
    
    print(f"Segmentation prediction shape: {seg_pred.shape}")
    print(f"Domain prediction shape: {domain_pred.shape}")
    print(f"Feature shape: {feature.shape}")
    print("DawNet model test passed!")

