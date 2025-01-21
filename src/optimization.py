import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_better_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    warm_up_type: str,
    num_training_steps: int,
    num_cycles: float,
    decay_pow: float = 1.0,
    warmup_pow: float = 1.0,
    min_lr_rate: float = 0.0,
):
    if current_step < num_warmup_steps and warm_up_type == "linear":
        factor = float(current_step) / float(max(1, num_warmup_steps))
        return max(0, factor)
    elif current_step < num_warmup_steps and warm_up_type == "cosine":
        progress = float(current_step) / float(max(1, num_warmup_steps))
        factor = math.pow(0.5 * (1.0 - math.cos(math.pi * progress)), warmup_pow)
        return max(0, factor)

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = math.pow(0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)), decay_pow)
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_better_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    warm_up_type: str = "linear",
    decay_pow: float = 1.0,
    warmup_pow: float = 1.0,
    min_lr_rate: float = 0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (torch.optim.Optimizer):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (int):
            The number of steps for the warmup phase.
        num_training_steps (int):
            The total number of training steps.
        num_cycles (float, optional, defaults to 0.5):
            The number of waves in the cosine schedule (the default is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (int, optional, defaults to -1):
            The index of the last epoch when resuming training.
        warm_up_type (str, optional, defaults to "linear"):
            The type of warmup. Options are "linear" or "cosine".
        decay_pow (float, optional, defaults to 1.0):
            The power of the decay function.
        warmup_pow (float, optional, defaults to 1.0):
            The power of the warmup function.
        min_lr_rate (float, optional, defaults to 0.0):
            The minimum learning rate as a fraction of the initial learning rate.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_better_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        warm_up_type=warm_up_type,
        decay_pow=decay_pow,
        warmup_pow=warmup_pow,
        min_lr_rate=min_lr_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
