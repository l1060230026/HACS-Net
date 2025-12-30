"""
Point Transformer with Hierarchical GAN for Domain Adaptation
Combines attention-based feature extraction with multi-level adversarial training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import farthest_point_sample, index_points
import torch.autograd.profiler as profiler
from torch_points_kernels import knn
import models.pointops as pointops


class FeatureDiscriminator(nn.Module):
    """
    Discriminator for point cloud features
    Handles variable-length point features with offset indexing
    """
    def __init__(self, input_dim, hidden_dim=64, discriminator_type='local'):
        super(FeatureDiscriminator, self).__init__()
        self.discriminator_type = discriminator_type
        
        if discriminator_type == 'local':
            # Local/patch discriminator - point-wise discrimination
            self.layers = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim * 2)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))
            )
        else:
            # Global discriminator - batch-level discrimination
            self.layers = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim * 2)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features, offset=None):
        """
        Args:
            features: (n, c) point features
            offset: (b,) cumulative point counts per batch (optional)
        Returns:
            scores: discrimination scores
        """
        if self.discriminator_type == 'local':
            # Point-wise discrimination
            scores = self.layers(features)  # (n, 1)
            return scores.squeeze(-1)  # (n,)
        else:
            # Global discrimination with batch pooling
            if offset is not None:
                # Pool features per batch
                batch_features = []
                for i in range(offset.shape[0]):
                    if i == 0:
                        s_i, e_i = 0, offset[0].item()
                    else:
                        s_i, e_i = offset[i-1].item(), offset[i].item()
                    
                    # Max pooling per batch
                    batch_feat = features[s_i:e_i, :].max(dim=0, keepdim=True)[0]
                    batch_features.append(batch_feat)
                
                batch_features = torch.cat(batch_features, dim=0)  # (b, c)
                scores = self.layers(batch_features)  # (b, 1)
                return scores
            else:
                # No offset provided, treat as single batch
                global_feat = features.max(dim=0, keepdim=True)[0]  # (1, c)
                scores = self.layers(global_feat)  # (1, 1)
                return scores


class HierarchicalDiscriminators(nn.Module):
    """
    Hierarchical discriminators for multi-level feature alignment in point transformer
    """
    def __init__(self, feature_dims, discriminator_types=None):
        super(HierarchicalDiscriminators, self).__init__()
        
        # Default: low-level features use local discriminators, high-level use global
        if discriminator_types is None:
            discriminator_types = ['local', 'local', 'global']
        
        self.discriminators = nn.ModuleList()
        
        for i, (feat_dim, disc_type) in enumerate(zip(feature_dims, discriminator_types)):
            self.discriminators.append(
                FeatureDiscriminator(feat_dim, hidden_dim=64, discriminator_type=disc_type)
            )
    
    def forward(self, features_list, offsets_list):
        """
        Args:
            features_list: List of feature tensors [(n1, c1), (n2, c2), ...]
            offsets_list: List of offset tensors [(b,), (b,), ...]
        Returns:
            scores_list: List of discriminator scores
        """
        scores_list = []
        for i, (features, offset) in enumerate(zip(features_list, offsets_list)):
            scores = self.discriminators[i](features, offset)
            scores_list.append(scores)
        
        return scores_list


class get_model(nn.Module):
    """
    Point Transformer with Hierarchical Discriminators
    """
    def __init__(self, num_classes, discriminator_level_indices=[1, 2, 3]):
        """
        Args:
            num_classes: Number of segmentation classes
            discriminator_level_indices: List of level indices to use (1, 2, 3 for dec2, dec3, dec4)
                                       Default: [1, 2, 3] uses all three levels
                                       Example: [3] to use only level 3 (dec4), [1, 3] to use levels 1 and 3
        """
        super(get_model, self).__init__()
        self.in_planes = 3
        self.c = 3
        self.discriminator_level_indices = discriminator_level_indices
        self.n_discriminator_levels = len(discriminator_level_indices)
        
        # Point Transformer Encoder
        self.enc2 = self._make_enc(PointTransformerBlock, 64, 3, share_planes=8, stride=4, nsample=16)  # N/4
        self.enc3 = self._make_enc(PointTransformerBlock, 128, 3, share_planes=8, stride=4, nsample=16) # N/16
        self.enc4 = self._make_enc(PointTransformerBlock, 256, 3, share_planes=8, stride=4, nsample=16) # N/64
        self.enc5 = self._make_enc(PointTransformerBlock, 512, 3, share_planes=8, stride=4, nsample=16) # N/256

        # Point Transformer Decoder
        self.dec5 = self._make_dec(PointTransformerBlock, 512, 1, share_planes=8, nsample=16, is_head=True)
        self.dec4 = self._make_dec(PointTransformerBlock, 256, 1, share_planes=8, nsample=16)
        self.dec3 = self._make_dec(PointTransformerBlock, 128, 1, share_planes=8, nsample=16)
        self.dec2 = self._make_dec(PointTransformerBlock, 64, 1, share_planes=8, nsample=16)
        self.dec1 = self._make_dec(PointTransformerBlock, 32, 1, share_planes=8, nsample=8)

        # Segmentation classifier
        self.cls = nn.Sequential(
            nn.Linear(32, 32), 
            nn.BatchNorm1d(32), 
            nn.ReLU(inplace=True), 
            nn.Linear(32, num_classes)
        )
        
        # Determine which levels to use
        # Level mapping: 1->dec2(64), 2->dec3(128), 3->dec4(256)
        all_level_info = [
            (1, 64, 'local'),   # dec2
            (2, 128, 'local'),  # dec3
            (3, 256, 'global')  # dec4
        ]
        
        # Build feature dims and types for selected levels
        feature_dims = []
        discriminator_types = []
        for level_idx in self.discriminator_level_indices:
            if 1 <= level_idx <= 3:
                level_info = all_level_info[level_idx - 1]
                feature_dims.append(level_info[1])
                discriminator_types.append(level_info[2])
            else:
                raise ValueError(f"Invalid discriminator level index: {level_idx}. Must be 1, 2, or 3.")
        
        self.discriminators = HierarchicalDiscriminators(feature_dims, discriminator_types)

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

    def forward(self, pxo, return_features=False):
        """
        Args:
            pxo: [p, x, o] where p: (n, 3), x: (n, c), o: (b)
            return_features: If True, return multi-level features for discriminators
        Returns:
            x: (n, num_classes) segmentation logits
            trans_feat: (n, 512) high-level features for regularization
            If return_features:
                multi_level_features: List of features from decoder levels
        """
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        
        # Encoder
        p1, x1, o1 = self.enc2([p0, x0, o0])
        p2, x2, o2 = self.enc3([p1, x1, o1])
        p3, x3, o3 = self.enc4([p2, x2, o2])
        p4, x4, o4 = self.enc5([p3, x3, o3])
        
        # Decoder with feature extraction
        x4_dec = self.dec5[1:]([p4, self.dec5[0]([p4, x4, o4]), o4])[1]
        x3_dec = self.dec4[1:]([p3, self.dec4[0]([p3, x3, o3], [p4, x4_dec, o4]), o3])[1]
        x2_dec = self.dec3[1:]([p2, self.dec3[0]([p2, x2, o2], [p3, x3_dec, o3]), o2])[1]
        x1_dec = self.dec2[1:]([p1, self.dec2[0]([p1, x1, o1], [p2, x2_dec, o2]), o1])[1]
        x0_dec = self.dec1[1:]([p0, self.dec1[0]([p0, None, o0], [p1, x1_dec, o1]), o0])[1]
        
        # Segmentation prediction
        x = self.cls(x0_dec)
        x = F.log_softmax(x, dim=1)
        
        if return_features:
            # Return multi-level features for discriminators
            # Use decoder features at different resolutions
            # Level mapping: 1->dec2(x1_dec, 64-dim), 2->dec3(x2_dec, 128-dim), 3->dec4(x3_dec, 256-dim)
            multi_level_features = []
            offsets = []
            
            level_feature_map = {
                1: (x1_dec, o1),  # dec2 output (64-dim)
                2: (x2_dec, o2),  # dec3 output (128-dim)
                3: (x3_dec, o3)   # dec4 output (256-dim)
            }
            
            for level_idx in self.discriminator_level_indices:
                if level_idx in level_feature_map:
                    feat, offset = level_feature_map[level_idx]
                    multi_level_features.append(feat)
                    offsets.append(offset)
            
            return x, multi_level_features, x0_dec  # x4 as trans_feat for regularization
        
        return x, x4  # Standard forward pass


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
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3), 
            nn.BatchNorm1d(3), 
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes)
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes), 
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.BatchNorm1d(mid_planes // share_planes), 
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)
        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


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
            self.linear1 = nn.Sequential(
                nn.Linear(2*in_planes, in_planes), 
                nn.BatchNorm1d(in_planes), 
                nn.ReLU(inplace=True)
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), 
                nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True)
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True)
            )
        
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
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            if x1 is None:
                x = pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
            else:
                x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class get_loss(nn.Module):
    """
    Loss function for attention-based segmentation
    Compatible with point transformer output format
    """
    def __init__(self):
        super(get_loss, self).__init__()
    
    def forward(self, pred, target, trans_feat, weight):
        """
        Args:
            pred: (n, num_classes) prediction logits
            target: (n,) ground truth labels
            trans_feat: (n, c) transformation features (for compatibility)
            weight: (num_classes,) class weights
        Returns:
            total_loss: scalar loss value
        """
        # Filter out invalid labels
        total_loss = F.nll_loss(pred[target != -1], target[target != -1], weight=weight)
        return total_loss


if __name__ == '__main__':
    import torch
    
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=13, discriminator_level_indices=[1, 2, 3]).to(device)
    
    # Simulate batch of points
    B = 2
    N = 4096
    C = 3
    
    # Create dummy data
    p = torch.randn(B * N, 3).to(device)  # coordinates
    x = p  # use coordinates as features
    o = torch.tensor([N, N*2], dtype=torch.int32).to(device)  # offsets
    
    print("Testing forward pass...")
    seg_pred, trans_feat = model([p, x, o])
    print(f"Segmentation prediction shape: {seg_pred.shape}")
    print(f"Trans features shape: {trans_feat.shape}")
    
    print("\nTesting forward pass with feature extraction...")
    seg_pred, multi_features, trans_feat = model([p, x, o], return_features=True)
    print(f"Segmentation prediction shape: {seg_pred.shape}")
    print(f"Number of multi-level features: {len(multi_features)}")
    for i, feat in enumerate(multi_features):
        print(f"Level {i+1} features shape: {feat.shape}")
    
    print("\nTesting discriminators...")
    offsets_list = [o, o, o]  # Dummy offsets for each level
    disc_scores = model.discriminators(multi_features, offsets_list)
    print(f"Number of discriminator scores: {len(disc_scores)}")
    for i, scores in enumerate(disc_scores):
        print(f"Discriminator {i+1} scores shape: {scores.shape}")
    
    print("\nModel test completed successfully!")

