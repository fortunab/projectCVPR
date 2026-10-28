"""
ConvNeXt Model Factory adapted for PolypGen Colonoscopy Binary Classification.
"""

import os
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torchvision.models as tv_models


def create_convnext_model(
    model_name: str = "convnext_tiny",
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.2
) -> Tuple[nn.Module, Dict[str, Any]]:
    model_config = {
        "model_name": model_name,
        "num_classes": num_classes,
        "img_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }

    try:
        import timm
        print(f"[ConvNeXt-Model] Loading '{model_name}' via timm for {num_classes}-class colonoscopy...")
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        cfg = getattr(model, 'default_cfg', {})
        if 'input_size' in cfg:
            model_config['img_size'] = cfg['input_size'][-1]
        if 'mean' in cfg:
            model_config['mean'] = cfg['mean']
        if 'std' in cfg:
            model_config['std'] = cfg['std']
        return model, model_config
    except Exception as e:
        print(f"[ConvNeXt-Model] timm load failed ({e}), using Torchvision fallback...")

    if "small" in model_name.lower():
        weights = tv_models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = tv_models.convnext_small(weights=weights)
        in_feats = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_feats, num_classes))
        model_config["model_name"] = "torchvision_convnext_small"
    else:
        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)
        in_feats = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_feats, num_classes))
        model_config["model_name"] = "torchvision_convnext_tiny"

    return model, model_config


def get_convnext_parameter_groups(model: nn.Module, backbone_lr: float = 2e-5, head_lr: float = 2e-4, weight_decay: float = 0.05) -> list:
    head_names = ["classifier", "head", "fc"]
    decay_backbone, no_decay_backbone = [], []
    decay_head, no_decay_head = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = any(hn in name.lower() for hn in head_names)
        is_no_decay = (len(param.shape) <= 1) or name.endswith(".bias") or ("norm" in name.lower())
        if is_head:
            (no_decay_head if is_no_decay else decay_head).append(param)
        else:
            (no_decay_backbone if is_no_decay else decay_backbone).append(param)

    return [
        {"params": decay_backbone, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": no_decay_backbone, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": decay_head, "lr": head_lr, "weight_decay": weight_decay},
        {"params": no_decay_head, "lr": head_lr, "weight_decay": 0.0}
    ]
