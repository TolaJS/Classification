"""
author:         Tola Shobande
name:           data.py
date:           4/09/2024
description:    Data loader and data augmentation using Dataset module
"""

from torch.utils.data import Dataset
from torchvision import datasets, transforms as T

class ProjectDataset():
    def __init__(self, mode):
        self.mode = mode

    def data_transforms(self):
        """
        This is the image transformation pipeline for the Training and validation datasets.
        This method creates two different sets of transformations:
        1. For training mode: Applies a series of random transformations to augment the dataset,
        including resizing, flips, rotations, color jittering, and normalization. Add more transformations
        or remove the ones unnecessary for training.
        2. For validation mode: Applies only resizing and normalization.

        :return: A torchvision.transforms.Compose object containing the appropriate sequence of transformations based
        on the current mode (train or val).
        """
        if self.mode == 'train':
            transforms = [
                T.Resize((224, 224)),   # Change size as needed
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(45),   # Change angle as needed
                T.ToTensor(),
                T.ColorJitter(),
                T.RandomErasing(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return T.Compose(transforms)
        elif self.mode == 'val':
            transforms = [
                T.Resize((224, 224)),   # Change size as needed
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return T.Compose(transforms)

    def __getitem__(self, item):
        pass
