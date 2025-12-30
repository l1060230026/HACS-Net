"""
EMA (Exponential Moving Average) model utilities
For creating and updating Teacher model
"""
import torch
import torch.nn as nn


def create_ema_model(ema_net, net):
    """Initialize EMA model"""
    for param_q, param_k in zip(net.parameters(), ema_net.parameters()):
        param_k.data = param_q.data.clone()
    for buffer_q, buffer_k in zip(net.buffers(), ema_net.buffers()):
        buffer_k.data = buffer_q.data.clone()
    ema_net.eval()
    for param in ema_net.parameters():
        param.requires_grad_(False)


def update_ema_variables(ema_net, net, alpha_ema, iteration):
    """Update EMA model parameters"""
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_ema)
    for ema_param, param in zip(ema_net.parameters(), net.parameters()):
        ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)
    for t, s in zip(ema_net.buffers(), net.buffers()):
        if not t.dtype == torch.int64:
            t.data.mul_(alpha_teacher).add_(s.data, alpha=1 - alpha_teacher)

