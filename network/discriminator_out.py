"""
Discriminator (D) - Output space discriminator
For Stage 1 PCAN, distinguishes source and target domain prediction outputs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiscriminatorOut(nn.Module):
    """Output space discriminator, performs domain discrimination based on logits"""
    def __init__(self, num_classes, feature_channels=[32, 64, 64, 128, 1], 
                 down_sample_times=[2, 2, 2, 2, 2], dis_kernel_size=4,
                 is_advent=True, gan_mode='ls_gan'):
        """
        Args:
            num_classes: Number of classes
            feature_channels: Feature channel number list
            down_sample_times: Downsampling factor list
            dis_kernel_size: Discriminator convolution kernel size
            is_advent: Whether to use ADVENT-style entropy normalization
            gan_mode: GAN loss type ('ls_gan' or 'vanilla_gan')
        """
        super(DiscriminatorOut, self).__init__()
        self.num_classes = num_classes
        self.is_advent = is_advent
        self.gan_mode = gan_mode
        
        # If using ADVENT mode, input channels are num_classes + 1
        input_channels = num_classes + 1 if is_advent else num_classes
        
        # Build downsampling-upsampling network
        layers = []
        in_channels = input_channels
        
        # Downsampling part
        for i, (out_ch, down_times) in enumerate(zip(feature_channels[:-1], down_sample_times[:-1])):
            # Downsampling
            for _ in range(down_times):
                layers.append(nn.Conv1d(in_channels, out_ch, kernel_size=dis_kernel_size, stride=2, padding=1))
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                in_channels = out_ch
        
        # Last downsampling layer
        for _ in range(down_sample_times[-1]):
            layers.append(nn.Conv1d(in_channels, feature_channels[-1], kernel_size=dis_kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(feature_channels[-1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = feature_channels[-1]
        
        # Upsampling part (restore to original size)
        # Simplified: use global average pooling + fully connected layer
        self.downsample_net = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_channels[-1], 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, logits):
        """
        Args:
            logits: [N, num_classes] Generator output (may be log probabilities or raw logits)
        
        Returns:
            output: [N, 1] Domain discrimination result
        """
        # ADVENT mode: apply entropy normalization to logits
        if self.is_advent:
            # If input is log probabilities (usually in negative range), first exp to get probabilities
            # Check if log probabilities: if max value < 0, likely log probabilities
            if logits.max() < 0:
                probs = F.softmax(logits, dim=1)  # softmax automatically handles log probabilities
            else:
                probs = F.softmax(logits, dim=1)
            num_classes = logits.shape[1]
            # entropy normalization: softmax(logits) * log2(softmax(logits)) / log2(C)
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1, keepdim=True) / np.log2(num_classes)
            # Use entropy as additional channel
            features = torch.cat([probs, entropy], dim=1)  # [N, num_classes+1]
        else:
            # Directly use logits (whether log probabilities or raw logits, Discriminator can learn)
            features = logits  # [N, num_classes]
        
        # Convert to [batch, channels, length] format for Conv1d
        # features: [N, C] -> [1, C, N]
        if features.dim() == 2:
            features = features.transpose(0, 1).unsqueeze(0)  # [1, C, N]
        
        # Downsampling
        x = self.downsample_net(features)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # [1, C]
        
        # Fully connected layer
        output = self.fc(x)  # [1, 1]
        
        # Expand to each point
        N = logits.shape[0]
        output = output.expand(N, 1)  # [N, 1]
        
        return output


class GANLoss(nn.Module):
    """GAN loss function"""
    def __init__(self, gan_mode='ls_gan', target_real_label=1.0, target_fake_label=0.0):
        """
        Args:
            gan_mode: 'ls_gan' or 'vanilla_gan'
            target_real_label: Real label value
            target_fake_label: Fake label value
        """
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        if gan_mode == 'ls_gan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla_gan':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported GAN mode: {gan_mode}")
    
    def __call__(self, prediction, target_is_real):
        """
        Args:
            prediction: Discriminator output
            target_is_real: Whether it is a real sample (True/False)
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        # Ensure target_tensor is on same device as prediction
        target_tensor = target_tensor.expand_as(prediction).to(prediction.device)
        loss = self.loss(prediction, target_tensor)
        return loss

