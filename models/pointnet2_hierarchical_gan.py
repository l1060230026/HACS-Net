import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class PatchDiscriminator(nn.Module):
    """
    Patch-based Discriminator for low-level geometric features
    Input: [B, C, N] where N is large (high resolution)
    Output: [B, N] patch-wise predictions
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(PatchDiscriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(input_dim, hidden_dim, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(hidden_dim, 1, kernel_size=1))
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: [B, C, N] feature map
        Returns:
            patch_scores: [B, N] patch-wise domain prediction scores
        """
        patch_scores = self.conv_layers(features)  # [B, 1, N]
        return patch_scores.squeeze(1)  # [B, N]


class GlobalDiscriminator(nn.Module):
    """
    Global Discriminator for high-level semantic features
    Input: [B, C, N] where N is small (low resolution)
    Output: [B, 1] global domain prediction
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(GlobalDiscriminator, self).__init__()
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim // 2, 1))
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: [B, C, N] feature map
        Returns:
            global_score: [B, 1] global domain prediction score
        """
        # Global pooling: [B, C, N] -> [B, C, 1] -> [B, C]
        global_feat = self.global_pool(features).squeeze(-1)
        
        # Classification: [B, C] -> [B, 1]
        global_score = self.classifier(global_feat)
        
        return global_score


class HierarchicalDiscriminators(nn.Module):
    """
    Hierarchical discriminators for multi-level feature alignment
    """
    def __init__(self, feature_dims):
        super(HierarchicalDiscriminators, self).__init__()
        
        # feature_dims: [fp1_dim, fp2_dim, fp3_dim, fp4_dim]
        self.discriminators = nn.ModuleList()
        
        # Low-level features (fp1, fp2): use patch-based discriminators
        for i in range(2):
            self.discriminators.append(
                PatchDiscriminator(feature_dims[i])
            )
        
        # High-level features (fp3, fp4): use global discriminators
        for i in range(2, 4):
            self.discriminators.append(
                GlobalDiscriminator(feature_dims[i])
            )
    
    def forward(self, features_list):
        """
        Args:
            features_list: List of feature maps from FP modules [fp1, fp2, fp3, fp4]
        Returns:
            scores_list: List of discriminator scores for each level
        """
        scores_list = []
        for i, features in enumerate(features_list):
            scores = self.discriminators[i](features)
            scores_list.append(scores)
        
        return scores_list


class get_model(nn.Module):
    """
    PointNet++ with hierarchical discriminators for multi-level feature alignment
    """
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        
        # PointNet++ Encoder
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        
        # PointNet++ Decoder (FP modules)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])  # 512 + 256 = 768
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])  # 256 + 128 = 384
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])  # 128 + 64 + 128 = 320
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])  # 128 + 64 = 192, but output is 128
        
        # Segmentation head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        
        # Hierarchical discriminators
        feature_dims = [128, 128, 256, 256]  # Output dimensions of fp1, fp2, fp3, fp4
        self.discriminators = HierarchicalDiscriminators(feature_dims)

    def forward(self, xyz):
        """
        Args:
            xyz: [B, 9, N] input point cloud (xyz + rgb + normal)
        Returns:
            seg_pred: [B, N, C] segmentation prediction
            multi_level_features: List of feature maps from FP modules
            discriminator_scores: List of discriminator scores for each level
        """
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder (FP modules) - extract multi-level features
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [B, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)       # [B, 128, N]

        # Segmentation head
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        seg_pred = F.log_softmax(x, dim=1)
        seg_pred = seg_pred.permute(0, 2, 1)  # [B, N, C]

        # Multi-level features for discriminators
        multi_level_features = [l0_points, l1_points, l2_points, l3_points]
        
        # Discriminator scores
        discriminator_scores = self.discriminators(multi_level_features)

        return seg_pred, multi_level_features, discriminator_scores


class get_loss(nn.Module):
    """
    Combined loss for segmentation and hierarchical adversarial learning
    """
    def __init__(self, lambda_adv=0.01, use_wgan=False):
        super(get_loss, self).__init__()
        self.lambda_adv = lambda_adv
        self.use_wgan = use_wgan
        
        if use_wgan:
            # WGAN-GP loss (no sigmoid)
            self.adv_loss_fn = lambda pred, target: -torch.mean(pred * target)
        else:
            # LSGAN loss
            self.adv_loss_fn = nn.MSELoss()

    def forward(self, seg_pred, target, discriminator_scores_list, weight, is_training_generator=True):
        """
        Args:
            seg_pred: [B, N, C] segmentation prediction
            target: [B, N] ground truth labels
            discriminator_scores_list: List of discriminator scores for each level
            weight: Class weights for segmentation loss
            is_training_generator: Whether training generator or discriminator
        Returns:
            total_loss: Combined loss
        """
        # Segmentation loss
        seg_loss = F.nll_loss(seg_pred, target, weight=weight)
        
        if not is_training_generator:
            # Training discriminator: no adversarial loss
            return seg_loss
        
        # Adversarial loss for generator
        adv_loss = 0.0
        for scores in discriminator_scores_list:
            if scores.dim() == 2:  # Patch discriminator output [B, N]
                # For patch discriminators, we want to fool them at every patch
                target_real = torch.ones_like(scores)  # [B, N]
                adv_loss += self.adv_loss_fn(scores, target_real)
            else:  # Global discriminator output [B, 1]
                # For global discriminators, we want to fool them globally
                target_real = torch.ones_like(scores)  # [B, 1]
                adv_loss += self.adv_loss_fn(scores, target_real)
        
        # Normalize by number of discriminators
        adv_loss = adv_loss / len(discriminator_scores_list)
        
        # Total loss
        total_loss = seg_loss + self.lambda_adv * adv_loss
        
        return total_loss


def gradient_penalty(discriminator, real_features, fake_features, device):
    """
    Calculate gradient penalty for WGAN-GP
    """
    batch_size = real_features.shape[0]
    alpha = torch.rand(batch_size, 1, 1).to(device)
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
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


if __name__ == '__main__':
    import torch
    
    # Test the model
    model = get_model(num_classes=13)
    xyz = torch.randn(2, 9, 4096)  # [B, 9, N]
    
    seg_pred, multi_level_features, discriminator_scores = model(xyz)
    
    print(f"Segmentation prediction shape: {seg_pred.shape}")
    print(f"Number of multi-level features: {len(multi_level_features)}")
    for i, feat in enumerate(multi_level_features):
        print(f"FP{i+1} features shape: {feat.shape}")
    print(f"Number of discriminator scores: {len(discriminator_scores)}")
    for i, scores in enumerate(discriminator_scores):
        print(f"Discriminator {i+1} scores shape: {scores.shape}")

