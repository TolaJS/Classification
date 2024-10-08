"""
author:         Tola Shobande
name:           utils.py
date:           4/09/2024
description:    Utility functions and classes for machine learning tasks.
                Includes functions for setting random seeds, a MetricTracker class
                for computing and tracking various performance metrics, and helper
                functions for formatting log messages and printing summary statistics.
                This module supports the main training and evaluation scripts with
                common utilities and metric calculations.
"""
import numpy as np
import torch
import random
import torchmetrics

import args


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # Sets the seed for generating random numbers for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    print(f"Random seed set to: {seed_value}")


class MetricTracker:
    """
    This class uses torchmetrics to compute accuracy (top-1 and top-5) and F1 score.

    Attributes:
        accuracy (torchmetrics.Accuracy): Top-1 accuracy metric.
        top5_accuracy (torchmetrics.Accuracy): Top-5 accuracy metric.
        f1_score (torchmetrics.F1Score): F1 score metric (macro-averaged).
    """
    def __init__(self, num_classes):
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.top5_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def to(self, device):
        """Move the metric computations to the specified device."""
        self.accuracy = self.accuracy.to(device)
        self.top5_accuracy = self.top5_accuracy.to(device)
        self.f1_score = self.f1_score.to(device)
        return self

    def update(self, preds, target):
        """Update all metrics with new predictions and targets."""
        self.accuracy.update(preds, target)
        self.top5_accuracy.update(preds, target)
        self.f1_score.update(preds, target)

    def compute(self):
        """Compute and return all tracked metrics."""
        return {
            'top1_acc': self.accuracy.compute(),
            'top5_acc': self.top5_accuracy.compute(),
            'f1': self.f1_score.compute()
        }

    def reset(self):
        """Reset all tracked metrics."""
        self.accuracy.reset()
        self.top5_accuracy.reset()
        self.f1_score.reset()


def format_log_message(mode, i, epoch, loss, acc, top1, top5, f1):
    """Log formatting function."""
    return f'| Mode:{mode:<5} | Iter:{i:5.1f} | Epoch:{epoch}/{args.epochs} | Train_Loss:{loss:8.3f} | Train_Acc:{acc:7.3f} | Top-1:{top1*100:7.3f} | Top-5:{top5*100:7.3f} | F1-Score:{f1:7.3f}'


def print_summary(logger, name, train_size, val_size, test_size):
    """Print summary model and data statistics."""
    logger.info('*'*17 + 'Summary' + '*'*17)
    logger.info(f'# model: {name}')
    logger.info(f'# train size: {train_size}')
    logger.info(f'# validation size: {val_size}')
    logger.info(f'# test size: {test_size}')
    logger.info('*'*41+'\n')
