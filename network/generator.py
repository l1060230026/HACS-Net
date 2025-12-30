"""
Generator wrapper, adapted for pointnet2_sem_seg_att
Supports returning features for prototype computation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Generator (G) - Segmentation network wrapper"""
    def __init__(self, backbone_model, num_classes, return_feature=False):
        """
        Args:
            backbone_model: pointnet2_sem_seg_att model instance
            num_classes: Number of classes
            return_feature: Whether to return intermediate features (for prototype computation)
        """
        super(Generator, self).__init__()
        self.backbone = backbone_model
        self.num_classes = num_classes
        self.return_feature = return_feature
        # Feature dimension is 32 (feature dimension from dec1 output)
        self.feature_dim = 32
    
    def forward(self, pxo, is_train=True):
        """
        Args:
            pxo: [coord, coord, offset] Input point cloud
            is_train: Training/inference mode
        
        Returns:
            logits: [N, num_classes] Classification logits
            features: [N, feature_dim] Feature vectors (if return_feature=True)
        """
        p0, x0, o0 = pxo
        x0 = p0 if self.backbone.c == 3 else torch.cat((p0, x0), 1)
        
        # Encoder
        p1, x1, o1 = self.backbone.enc2([p0, x0, o0])
        p2, x2, o2 = self.backbone.enc3([p1, x1, o1])
        p3, x3, o3 = self.backbone.enc4([p2, x2, o2])
        p4, x4, o4 = self.backbone.enc5([p3, x3, o3])
        
        # Decoder
        x4 = self.backbone.dec5[1:]([p4, self.backbone.dec5[0]([p4, x4, o4]), o4])[1]
        x3 = self.backbone.dec4[1:]([p3, self.backbone.dec4[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.backbone.dec3[1:]([p2, self.backbone.dec3[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.backbone.dec2[1:]([p1, self.backbone.dec2[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        features = self.backbone.dec1[1:]([p0, self.backbone.dec1[0]([p0, None, o0], [p1, x1, o1]), o0])[1]
        
        # Classifier
        logits = self.backbone.cls(features)
        # Apply log_softmax to match F.nll_loss requirement
        log_probs = F.log_softmax(logits, dim=1)
        
        if self.return_feature:
            return log_probs, features
        else:
            return log_probs

