"""
Pseudo label generation utilities
"""
import torch
import torch.nn.functional as F
import numpy as np


def generate_pseudo_labels(logits, method='entropy', threshold=0.05):
    """
    Generate pseudo labels
    
    Args:
        logits: [N, num_classes] Output logits from Teacher model
        method: 'entropy' or 'confidence'
        threshold: Threshold value
    
    Returns:
        pseudo_labels: [N] Pseudo labels (invalid positions are -1, valid labels are in range [0, num_classes-1])
    """
    probs = F.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    
    if method == 'entropy':
        # Entropy threshold method
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        mask = entropy < threshold
        pseudo_labels = probs.argmax(dim=1)
        # Ensure label values are in valid range
        pseudo_labels = torch.clamp(pseudo_labels, 0, num_classes - 1)
        # Set invalid labels to -1
        pseudo_labels[~mask] = -1
    
    elif method == 'confidence':
        # Confidence threshold method
        conf, pseudo_labels = probs.max(dim=1)
        mask = conf > threshold
        # Ensure label values are in valid range
        pseudo_labels = torch.clamp(pseudo_labels, 0, num_classes - 1)
        # Set invalid labels to -1
        pseudo_labels[~mask] = -1
    
    # Final safety check: ensure all label values are in valid range
    if (pseudo_labels >= 0).any():
        valid_labels = pseudo_labels[pseudo_labels >= 0]
        if (valid_labels < 0).any() or (valid_labels >= num_classes).any():
            print(f"[WARNING] Pseudo labels out of range after generation! "
                  f"min={valid_labels.min().item()}, max={valid_labels.max().item()}, num_classes={num_classes}")
            # Force clamp
            pseudo_labels = torch.clamp(pseudo_labels, -1, num_classes - 1)
    
    return pseudo_labels


def compute_category_adv_loss(adv_loss, pseudo_labels, features, prototypes):
    """
    Compute category-aware adversarial loss using prototypes for reweighting
    
    Args:
        adv_loss: [N, 1] Original adversarial loss
        pseudo_labels: [N] Pseudo labels
        features: [N, feature_dim] Feature vectors
        prototypes: [num_classes, feature_dim] Class prototypes
    
    Returns:
        cal_adv_loss: Reweighted adversarial loss
    """
    unique_labels = torch.unique(pseudo_labels)
    # Initialize as zero tensor on same device as adv_loss
    cal_adv_loss = torch.tensor(0.0, device=adv_loss.device, dtype=adv_loss.dtype)
    
    for label in unique_labels:
        label = label.item()
        if label == 0:  # Ignore background
            continue
        
        mask = (pseudo_labels == label)
        if mask.sum() < 30:
            # Too few samples, directly average
            cal_adv_loss = cal_adv_loss + adv_loss[mask].mean()
        else:
            # Use prototype similarity for reweighting
            label_features = features[mask]
            label_proto = prototypes[label]
            
            # Ensure prototype is on correct device
            if label_proto.device != features.device:
                label_proto = label_proto.to(features.device)
            
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(
                label_proto.unsqueeze(0).expand_as(label_features),
                label_features,
                dim=1
            )
            # Convert to distance weights (farther from prototype, larger weight)
            weights = 1.0 - cos_sim
            
            # Weighted average
            weighted_loss = (adv_loss[mask].squeeze() * weights).sum() / (weights.sum() + 1e-8)
            cal_adv_loss = cal_adv_loss + weighted_loss
    
    return cal_adv_loss

