"""
author:         Tola Shobande
name:           data.py
date:           4/09/2024
description:    Data loader and data augmentation using Dataset module
"""

from torch.utils.data import Dataset
from torchvision import datasets, transforms as T

class ProjectDataset():
    def __init__(self,mode):
        self.mode = mode

    def data_transforms(self):
        if self.mode == 'train':
            transforms = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(45),
                T.ToTensor(),
                T.ColorJitter(),
                T.RandomErasing(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ])
            pass
        elif self.mode == 'val':
            pass



    def __getitem__(self, item):
        pass


