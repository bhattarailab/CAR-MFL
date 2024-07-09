import warnings
import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from adamp import AdamP


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def get_optimizer(optimizer_name, parameters, config, logger=None):
    if logger:
        logger.log('creating [{}] from Config({})'.format(optimizer_name, config))
    if optimizer_name == 'adam':
        if set(config.keys()) - {'learning_rate', 'betas', 'eps',
                                 'weight_decay', 'amsgrad', 'name'}:
            warnings.warn('found unused keys in {}'.format(config.keys()))
        optimizer = optim.Adam(parameters,
                               lr=config.learning_rate,
                               betas=config.get('betas', (0.9, 0.999)),
                               eps=float(config.get('eps', 1e-8)),
                               weight_decay=float(config.get('weight_decay', 0)),
                               amsgrad=config.get('amsgrad', False))
    elif optimizer_name == 'adamn' or optimizer_name == 'adamp':
        if set(config.keys()) - {'learning_rate', 'betas', 'eps',
                                 'weight_decay', 'name'}:
            warnings.warn('found unused keys in {}'.format(config.keys()))
        optimizer = AdamP(parameters,
                          lr=config.learning_rate,
                          betas=config.get('betas', (0.9, 0.999)),
                          eps=float(config.get('eps', 1e-8)),
                          weight_decay=float(config.get('weight_decay', 0)))
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(parameters,
                                lr=config.learning_rate)
    else:
        raise ValueError(f'Invalid optimizer name: {optimizer_name}')
    return optimizer


def get_lr_scheduler(scheduler_name, optimizer, config, logger=None):
    if logger:
        logger.log('creating [{}] from Config({})'.format(scheduler_name, config))
    if scheduler_name == 'reduce_lr_on_plateau':
        if set(config.keys()) - {'mode', 'factor', 'patience',
                                 'verbose', 'threshold',
                                 'threshold_mode', 'cooldown',
                                 'min_lr', 'eps', 'name'}:
            warnings.warn('found unused keys in {}'.format(config.keys()))
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=config.get('mode', 'min'),
            factor=float(config.get('factor', 0.1)),
            patience=config.get('patience', 10),
            verbose=config.get('verbose', True),
            threshold=float(config.get('threshold', 1e-4)),
            threshold_mode=config.get('threshold_mode', 'rel'),
            cooldown=float(config.get('cooldown', 0)),
            min_lr=float(config.get('min_lr', 0)),
            eps=float(config.get('eps', 1e-8)))
    elif scheduler_name == 'cosine_annealing':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.T_max)
    elif scheduler_name == 'cosine_warmup':
        lr_scheduler = WarmupCosineSchedule(optimizer, config.warmup, t_total=config.T_max)
    else:
        raise ValueError(f'Invalid scheduler name: {scheduler_name}')
    return lr_scheduler
