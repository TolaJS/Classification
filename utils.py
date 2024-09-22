"""
author:     Tola Shobande
name:       utils.py
description:
"""
import numpy as np
import torch
import random

import torchmetrics


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
    def __init__(self, num_classes):
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.top5_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def update(self, preds, target):
        self.accuracy.update(preds, target)
        self.top5_accuracy.update(preds, target)
        self.f1_score.update(preds, target)

    def compute(self):
        return {
            'top1_acc': self.accuracy.compute(),
            'top5_acc': self.top5_accuracy.compute(),
            'f1': self.f1_score.compute()
        }

    def reset(self):
        self.accuracy.reset()
        self.top5_accuracy.reset()
        self.f1_score.reset()
