import math
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler

def WarmupCosineAnnealingLR(cur_iter,warm_up_iter=10,T_max=50,lr_max=0.1,lr_min=1e-5):
    """
    From: https://blog.marquis.eu.org/posts/2e3746c6/
    Usage: scheduler = LambdaLR(optimizer, lr_lambda=WarmupCosineAnnealingLR)
    """
    if cur_iter < warm_up_iter:
        return (lr_max - lr_min) * (cur_iter / warm_up_iter) + lr_min
    else:
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi)
        )


def get_linear_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class NoamLR(_LRScheduler):
    """
    Adapted from https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py

    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, model_size, warmup_steps):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps**(-1.5))

        return [base_lr * scale for base_lr in self.base_lrs]