"""
author:         Tola Shobande
name:           main.py
date:           4/09/2024
description:    ...
"""

import os
import args
import torch
import logging
from model import Classifier
from data import ProjectDataset
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter


def train(train_loader, model, criterion, optimiser, epoch, scaler, scheduler, device):
    model.train()
    losses = AverageMeter()
    writer = SummaryWriter(log_dir=f"logs/epoch_{epoch}")

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimiser.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        losses.update(loss.item(), inputs.size(0))

    scheduler.step()

    writer.add_scalar('Loss/epoch_train', losses.avg, epoch)
    logging.info(f"Epoch [{epoch}] - Loss: {losses.avg:.4f}")
    writer.close()



def evaluate():
    pass


def test():
    pass


def main():
    """

    :return:
    """
    """Data Loading"""
    train_data = ProjectDataset(mode='train', root_dir='dataset', seed=args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = ProjectDataset(mode='val', root_dir='dataset', seed=args.seed)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    test_data = ProjectDataset(mode='test', root_dir='dataset', seed=args.seed)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    """Model Initialisation"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Classifier(num_classes=args.num_classes)
    model.to(device)

    """Parameters"""
    #   OPTIMISER
    base_params = [  # Parameters in the model except FC layer
        param for name, param in model.named_parameters() if "fc" not in str(name)
    ]

    optimiser = torch.optim.Adam(
        [
            {"params": base_params},
            {"params": model.model.fc.parameters(), "lr": 0.01},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == '__main__':
    main()
