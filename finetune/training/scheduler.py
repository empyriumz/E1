import numpy as np
import torch


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        # Ensure min_lr is a float (handle case where it comes from config as string)
        self.min_lr = float(min_lr)
        # Ensure base_lrs are floats (handle case where lr comes from config as string)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, metrics=None):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            # Ensure both values are floats before comparison
            new_lr = max(float(base_lr) * float(factor), float(self.min_lr))
            param_group["lr"] = new_lr
