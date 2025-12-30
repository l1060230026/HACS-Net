"""
Prototype estimator - for class prototype computation and updating
"""
import torch
import torch.nn.functional as F


class PrototypeEstimator:
    """Class prototype estimator"""
    def __init__(self, num_classes, feature_dim, update_mode='mean', momentum=0.9999, device='cuda'):
        """
        Args:
            num_classes: Number of classes
            feature_dim: Feature dimension
            update_mode: 'mean' or 'moving_average'
            momentum: Momentum coefficient (for moving_average mode)
            device: Device
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.update_mode = update_mode
        self.momentum = momentum
        self.device = device
        
        # Initialize prototypes and counts
        self.Proto = torch.zeros(num_classes, feature_dim).to(device)
        self.Amount = torch.zeros(num_classes).to(device)
    
    def update(self, features, labels):
        """
        Update prototypes
        
        Args:
            features: [N, feature_dim] Feature vectors
            labels: [N] Labels
        """
        # Ignore background (labels 0 or -1)
        mask = (labels > 0) & (labels < self.num_classes)
        if mask.sum() == 0:
            return
        
        features = features[mask]
        labels = labels[mask]
        
        if self.update_mode == 'mean':
            # Weighted average update
            unique_labels = torch.unique(labels)
            for label in unique_labels:
                label = label.item()
                if label >= self.num_classes:
                    continue
                
                label_mask = (labels == label)
                label_features = features[label_mask]
                
                if label_features.shape[0] == 0:
                    continue
                
                mean_feat = label_features.mean(dim=0)
                
                # Compute weight
                n_samples = len(label_features)
                weight = n_samples / (n_samples + self.Amount[label] + 1e-8)
                
                # Update prototype
                self.Proto[label] = (1 - weight) * self.Proto[label] + weight * mean_feat
                self.Amount[label] += n_samples
                self.Amount[label] = min(self.Amount[label], 100000)
        
        elif self.update_mode == 'moving_average':
            # Momentum update
            unique_labels = torch.unique(labels)
            for label in unique_labels:
                label = label.item()
                if label >= self.num_classes:
                    continue
                
                label_mask = (labels == label)
                label_features = features[label_mask]
                
                if label_features.shape[0] == 0:
                    continue
                
                mean_feat = label_features.mean(dim=0)
                
                # Momentum update
                self.Proto[label] = (1 - self.momentum) * mean_feat + \
                                    self.momentum * self.Proto[label]
                self.Amount[label] += len(label_features)
    
    def get_prototypes(self):
        """Get current prototypes"""
        return self.Proto.clone()
    
    def reset(self):
        """Reset prototypes"""
        self.Proto.zero_()
        self.Amount.zero_()

