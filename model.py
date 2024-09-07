"""
author:         Tola Shobande
name:           model.py
date:           4/09/2024
description:
"""

import torch
import args
import torch.nn as nn
from torchvision import models


class Classifier(nn.Module):
    """

    """

    def __init__(self, num_classes, freeze_weights=True):
        super(Classifier, self).__init__()
        self.model = models.resnet50(weights="IMAGENET1K_V2")

        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


def get_model(num_classes, freeze_weights=True):
    return Classifier(num_classes=num_classes, freeze_weights=freeze_weights)


def load_model(num_classes, path):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(path))
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
