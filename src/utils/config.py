import yaml
import torch
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for G-hash experiments"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = "configs/config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self.config.update(updates)
    
    @property
    def device(self):
    
        """Get device from config or auto-detect"""
        if 'device' in self.config:
            return torch.device(self.config['device'])
            
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def save(self, path: str):
        """Save configuration to file"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
