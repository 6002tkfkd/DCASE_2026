"""Utility to load loss functions dynamically from config."""

import importlib
import torch.nn as nn
from typing import Type


def load_loss_class(loss_name: str) -> Type[nn.Module]:
    """Load a loss function class by name.
    
    Priority order:
    1. src.losses module (custom losses)
    2. torch.nn module (standard PyTorch losses)
    3. Fully qualified path (e.g., 'custom.module.LossClass')
    
    Args:
        loss_name: Name of loss class (e.g., 'CrossEntropyLoss', 'src.losses.CrossEntropyLoss')
    
    Returns:
        Loss class (not instantiated)
        
    Raises:
        ImportError: If loss class cannot be loaded
    """
    
    # 1. Try src.losses first (custom losses)
    try:
        module = importlib.import_module('src.losses')
        if hasattr(module, loss_name):
            return getattr(module, loss_name)
        # If not found as attribute, try as module.class
        if '.' not in loss_name:
            try:
                submodule = importlib.import_module(f'src.losses.{loss_name.lower()}')
                return getattr(submodule, loss_name)
            except (ImportError, AttributeError):
                pass
    except ImportError:
        pass
    
    # 2. Try torch.nn (standard PyTorch losses)
    try:
        return getattr(nn, loss_name)
    except AttributeError:
        pass
    
    # 3. Try fully qualified path (has dot)
    if '.' in loss_name:
        parts = loss_name.rsplit('.', 1)
        module_name = parts[0]
        class_name = parts[1]
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot load {loss_name}: {e}")
    
    # Not found anywhere
    raise ImportError(
        f"Loss '{loss_name}' not found in:\n"
        f"  - src.losses module\n"
        f"  - torch.nn module\n"
        f"Try specifying full path like 'src.losses.CustomLoss'"
    )


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Get instantiated loss function from name.
    
    Args:
        loss_name: Name of loss class (e.g., 'CrossEntropyLoss')
    
    Returns:
        Instantiated loss function
    """
    loss_class = load_loss_class(loss_name)
    return loss_class(**kwargs)
