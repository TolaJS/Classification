"""
author:         Tola Shobande
name:           model.py
date:           4/09/2024
description:    This script defines a custom image classification model based on the ResNet-50 architecture with an 
                option to freeze its pre-trained weights and a custom fully connected layer for a specific number of
                classes. The script includes utility functions to instantiate, load, and save the model.
"""

import torch
import args
import torch.nn as nn
from torchvision import models


class Classifier(nn.Module):
    """
    This is a custom image classification model built upon a pre-trained ResNet-50 architecture.
    The model allows for fine-tuning of the final fully connected layers while freezing the earlier layers.

    Attributes:
    ----
    model : torchvision.models.resnet.ResNet
        The pre-trained ResNet-50 model from torchvision with a modified fully connected layer.
    freeze_weights : bool
        Determines whether to freeze the pre-trained weights of ResNet-50.
    num_classes : int
        The number of classes for the classification task.
    """

    def __init__(self, num_classes: int, freeze_weights=True):
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
        """
        Defines the forward pass of the model, passing input through the modified ResNet-50 architecture.
        :param x: torch.Tensor :
            The input tensor to be passed through the network.
        :returns: torch.Tensor :
            The log probabilities for each class, returned as an output of the LogSoftmax layer.
        """
        return self.model(x)
