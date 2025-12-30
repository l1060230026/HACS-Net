"""
DawNet Training Configuration File
"""

class DawNetConfig:
    """DawNet Default Configuration"""
    
    # Model configuration
    model_name = 'pointnet2_dawnet'
    num_classes = 11  # Number of classes after excluding board and clutter
    use_whitening = True
    use_attention = True
    
    # Training configuration
    batch_size = 4
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-4
    optimizer = 'Adam'
    
    # Learning rate scheduling
    step_size = 10
    lr_decay = 0.5
    lr_clip = 1e-5
    
    # BN momentum scheduling
    bn_momentum = 0.1
    bn_momentum_decay = 0.5
    bn_momentum_decay_step = 10
    
    # Domain adaptation configuration
    lambda_adv = 1.0  # Weight for adversarial loss
    
    # Data configuration
    source_data_root = 'data/bim_indoor3d'
    target_data_root = 'data/stanford_indoor3d'
    test_area = 5
    num_point = 40000
    cache_data = False
    
    # Excluded classes
    exclude_classes = ['board', 'clutter']
    
    # Class definitions
    all_classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 
                   'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    
    # GPU configuration
    gpu_id = '0'
    num_workers = 5
    
    # Logging configuration
    log_dir = 'dawnet_bim2scan'
    save_freq = 5  # Save every 5 epochs
    
    # Data augmentation configuration
    use_random_rotate = True
    use_random_flip = True
    random_flip_prob = 0.5
    use_random_jitter = True
    jitter_sigma = 0.01
    jitter_clip = 0.05
    

class DawNetConfigNoWhitening(DawNetConfig):
    """Ablation experiment configuration without whitening"""
    use_whitening = False
    log_dir = 'dawnet_no_whitening'


class DawNetConfigNoAttention(DawNetConfig):
    """Ablation experiment configuration without attention"""
    use_attention = False
    log_dir = 'dawnet_no_attention'


class DawNetConfigBaseline(DawNetConfig):
    """Baseline configuration (without domain adaptation)"""
    lambda_adv = 0.0
    log_dir = 'dawnet_baseline'


class DawNetConfigFull(DawNetConfig):
    """Full feature configuration"""
    use_whitening = True
    use_attention = True
    lambda_adv = 1.0
    cache_data = True  # If sufficient memory is available
    log_dir = 'dawnet_full'


# Configuration dictionary
CONFIGS = {
    'default': DawNetConfig,
    'no_whitening': DawNetConfigNoWhitening,
    'no_attention': DawNetConfigNoAttention,
    'baseline': DawNetConfigBaseline,
    'full': DawNetConfigFull,
}


def get_config(config_name='default'):
    """
    Get specified configuration
    
    Args:
        config_name: Configuration name, options: 'default', 'no_whitening', 'no_attention', 'baseline', 'full'
    
    Returns:
        Configuration class instance
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available configs: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]()


if __name__ == '__main__':
    # Test configuration
    config = get_config('default')
    print("DawNet Default Configuration:")
    print("-" * 50)
    for key, value in vars(config).items():
        if not key.startswith('_'):
            print(f"{key:30s}: {value}")

