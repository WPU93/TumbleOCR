import torch
from torch import optim

def add_weight_decay(model, weight_decay=1e-6, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def create_optimizer(model,cfg,filter_bias_and_bn=True):
    weight_decay = cfg.Optimizer.weight_decay
    lr = cfg.Optimizer.lr
    if weight_decay and filter_bias_and_bn:
        skip = {}   
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if cfg.Optimizer.name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, momentum=cfg.Optimizer.momentum,
            weight_decay=weight_decay,nesterov=True)
    elif cfg.Optimizer.name == "adam":
        optimizer = optim.Adam(parameters, lr=lr,weight_decay=weight_decay)
    elif cfg.Optimizer.name == "adadelta":
        optimizer = optim.adadelta(parameters, lr=lr,weight_decay=weight_decay)
    elif cfg.Optimizer.name == "rmsprop":
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=cfg.Optimizer.momentum)

    return optimizer