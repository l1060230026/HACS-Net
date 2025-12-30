import torch
import torch.nn as nn
import torch.nn.functional as F

class get_model(nn.Module):
    """
    Output space discriminator (Output Space Discriminator)
    Input: Segmentation network's Softmax probability map [B, C, N] (C is number of classes)
    Output: Scene-level domain discrimination score [B, 1]
    """
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        
        # 1. Shared MLP, processes C-dimensional probability vector for each point
        # Use spectral normalization to increase training stability
        self.shared_mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(num_classes, 64, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(64, 128, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(128, 256, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. Global feature classifier, processes aggregated global vector
        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(128, 1))
            # Note: No Sigmoid at the end here, to facilitate using more stable GAN losses like LSGAN or WGAN-GP
            # If you insist on using BCE Loss, you can add torch.sigmoid(score) at the end of forward
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pred_softmax):
        """
        Args:
            pred_softmax: Segmentation network's Softmax probability map [B, C, N]
        Returns:
            score: Domain discrimination score [B, 1]
        """
        # 1. Extract per-point features through shared MLP
        # Input: [B, C, N] -> Output: [B, 256, N]
        point_features = self.shared_mlp(pred_softmax)
        
        # 2. Symmetric function aggregation: max pooling, get global features
        # Input: [B, 256, N] -> Output: [B, 256]
        global_feature = torch.max(point_features, dim=2)[0]
        
        # 3. Global feature classification, get final score
        # Input: [B, 256] -> Output: [B, 1]
        score = self.classifier(global_feature)

        score = torch.sigmoid(score)
        
        return score

# Loss function can remain unchanged, but it is recommended to use more stable LSGAN Loss
class get_loss(nn.Module):
    def __init__(self, loss_type='lsgan'):
        super(get_loss, self).__init__()
        self.loss_type = loss_type
        if self.loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.loss_type == 'lsgan':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred, target_is_real):
        """
        pred: Discriminator output score [B, 1]
        target_is_real: Boolean indicating whether target is real (True for real/source, False for fake/target)
        """
        target_val = 1.0 if target_is_real else 0.0
        target = torch.full_like(pred, target_val)
        return self.loss_fn(pred, target)

# --- Usage example in main function ---

if __name__ == '__main__':
    NUM_CLASSES = 13
    BATCH_SIZE = 8
    NUM_POINTS = 4096

    # 1. Initialize discriminator
    discriminator = get_model(num_classes=NUM_CLASSES).cuda()
    print("Discriminator initialized.")
    
    # 2. Initialize loss function (recommended to use LSGAN)
    criterion_adv = get_loss(loss_type='lsgan').cuda()
    
    # --- Simulated training loop ---
    
    # Assume this is your segmentation network (classifier)
    segmentation_net = ... # Your PointNet++ model

    # Segmentation logits from source domain (synthetic data)
    # Shape of seg_pred_source_logits is [B, N, C]
    seg_pred_source_logits = torch.randn(BATCH_SIZE, NUM_POINTS, NUM_CLASSES).cuda()
    
    # Segmentation logits from target domain (real data)
    seg_pred_target_logits = torch.randn(BATCH_SIZE, NUM_POINTS, NUM_CLASSES).cuda()
    
    # Convert to format required by discriminator [B, C, N]
    seg_pred_source = seg_pred_source_logits.permute(0, 2, 1)
    seg_pred_target = seg_pred_target_logits.permute(0, 2, 1)

    # Compute Softmax probabilities
    prob_source = F.softmax(seg_pred_source, dim=1)
    prob_target = F.softmax(seg_pred_target, dim=1)

    # --- Discriminator training step ---
    # optimizer_D.zero_grad()
    
    # Discriminate source domain (real domain)
    d_out_source = discriminator(prob_source.detach()) # detach to prevent gradient from flowing back to segmentation network
    loss_d_source = criterion_adv(d_out_source, True)
    
    # Discriminate target domain (fake domain)
    d_out_target = discriminator(prob_target.detach())
    loss_d_target = criterion_adv(d_out_target, False)
    
    loss_d = (loss_d_source + loss_d_target) * 0.5
    # loss_d.backward()
    # optimizer_D.step()
    
    print(f"Discriminator Loss: {loss_d.item():.4f}")

    # --- Segmentation network (generator) training step ---
    # optimizer_G.zero_grad()

    # 1. Segmentation loss (only computed on source domain)
    # loss_seg = ...
    
    # 2. Adversarial loss (goal is to fool discriminator, make discriminator think target domain output is from source domain)
    d_out_target_for_g = discriminator(prob_target) # Cannot detach here
    loss_g_adv = criterion_adv(d_out_target_for_g, True) # Goal is to make discriminator think it's True (real/source)
    
    # Total loss
    lambda_adv = 0.01 # Weight for adversarial loss, needs careful tuning
    # total_loss_g = loss_seg + lambda_adv * loss_g_adv
    # total_loss_g.backward()
    # optimizer_G.step()

    print(f"Generator Adversarial Loss: {loss_g_adv.item():.4f}")
