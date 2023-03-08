import numpy as np
import torch
from torch import nn
from src.utils.general import LOGGER, colorstr


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            g[2].append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            g[0].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            g[1].append(v.weight)
    if name == 'Adam':
        optimizer = torch.optim.Adam(g[0], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    # elif name == 'AdamW':
    #     optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    # elif name == 'RMSProp':
    #     optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[0], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[1], 'weight_decay': decay})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g[2], 'weight_decay': 0.0})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g[0])} weight (no decay), {len(g[1])} weight, {len(g[2])} bias")
    return optimizer
