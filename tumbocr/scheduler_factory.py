from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .utils.tricks import GradualWarmupScheduler

def create_scheduler(optimizer,cfg):

    scheduler = None
    if cfg.Optimizer.scheduler.name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs*cfg.Train.num_batches, eta_min=4e-8)
    elif cfg.Optimizer.scheduler.name == "step":
        milestones_list = cfg.Optimizer.scheduler.milestones_list
        gamma = cfg.Optimizer.scheduler.gamma
        scheduler= MultiStepLR(optimizer, milestones=milestones_list, gamma=gamma)
    
    if cfg.Optimizer.warmup > 0:
        epochs = cfg.Global.epochs - cfg.Optimizer.warmup
        total_iter = cfg.Optimizer.warmup*cfg.Train.num_batches
        multiplier = cfg.Optimizer.init_lr/cfg.Optimizer.warmup_lr
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=multiplier, total_iter=total_iter, after_scheduler=scheduler)

    return scheduler