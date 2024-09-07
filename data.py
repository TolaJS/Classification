"""
author:         Tola Shobande
name:           data.py
date:           4/09/2024
description:    This file manages image data for machine learning, using PyTorch and torchvision to load, split,
                and transform datasets. It supports training, validation, and testing modes with customized data
                augmentation techniques, and provides methods for dataset access and manipulation.
"""

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms as T


class ProjectDataset:
    def __init__(self, mode, root_dir, split_ratio=(0.7, 0.2, 0.1)):
        self.mode = mode
        self.root_dir = root_dir
        self.split_ratio = split_ratio
        self.full_dataset = datasets.ImageFolder(root_dir, transform=self.data_transforms())
        self.train_dataset, self.val_dataset, self.test_dataset = self.split_dataset(self.full_dataset)

        if self.mode == 'train':
            self.dataset = self.train_dataset
        elif self.mode == 'val':
            self.dataset = self.val_dataset
        elif self.mode == 'test':
            self.dataset = self.test_dataset
        else:
            raise ValueError("Mode should be 'train', 'val', or 'test'.")

    def data_transforms(self):
        """
        This is the image transformation pipeline for the Training and validation datasets.
        This method creates two different sets of transformations:
        1. For training mode: Applies a series of random transformations to augment the dataset,
        including resizing, flips, rotations, color jittering, and normalization. Add more transformations
        or remove the ones unnecessary for training.
        2. For validation mode: Applies only resizing and normalization.

        :return: A torchvision.transforms.Compose object containing the appropriate sequence of transformations based on the current mode (train or val).
        """
        if self.mode == 'train':
            transforms = [
                T.Resize((224, 224)),  # Change size as needed
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(45),  # Change angle as needed
                T.ToTensor(),
                T.ColorJitter(),
                T.RandomErasing(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return T.Compose(transforms)
        elif self.mode in ['val', 'test']:
            transforms = [
                T.Resize((224, 224)),  # Change size as needed
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return T.Compose(transforms)

    def split_dataset(self, dataset):
        """
        Splits the dataset into training, validation and testing sets based on the `self.split_ratio` attribute.
        :param dataset: The full dataset loaded using torchvison `datasets.ImageFolder`
        :return: Tuple (train_dataset, val_dataset, test_dataset)
        """
        train_len = int(len(dataset) * self.split_ratio[0])
        val_len = int(len(dataset) * self.split_ratio[1])
        test_len = len(dataset) - train_len - val_len
        torch.manual_seed(7)

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        """
        Returns the number of samples in the assigned dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns the sample at index *idx* in the dataset.
        :param idx: The index of the sample in the dataset.
        :return: Tuple (image, label)
        """
        return self.dataset[idx]
