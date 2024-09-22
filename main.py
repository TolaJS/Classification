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
from datetime import datetime
from tqdm import tqdm
from model import Classifier
from uuid import uuid4
from data import ProjectDataset
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
from utils import *
from torch.utils.tensorboard import SummaryWriter


def train(train_loader, model, criterion, optimiser, epoch, scaler, scheduler, device):
    """
    :param train_loader:
    :param model:
    :param criterion:
    :param optimiser:
    :param epoch:
    :param scaler:
    :param scheduler:
    :param device:
    :return:
    """
    model.train()
    losses = AverageMeter()  # Track the average loss
    metrics_tracker = MetricTracker(args.num_classes)
    correct = 0
    total = 0

    #   Pass images and labels to correct device
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        #   Zero the parameter gradient
        optimiser.zero_grad()

        #   Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        #   Backward pass and optimiser step
        #   TODO Try this later without scaler to see the difference
        """
        loss.backward()
        optimiser.step()
        """
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        #   Metrics Update
        losses.update(loss.item(), images.size(0))  # Update loss
        metrics_tracker.update(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)  # Get predicted labels
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum().item()

    #   Update scheduler
    scheduler.step()

    # todo figure out how to use logging
    # logging.info(f"Epoch [{epoch}] - Loss: {losses.avg:.4f}")
    train_accuracy = 100 * (correct / total)
    metrics = metrics_tracker.compute()
    metrics_tracker.reset()
    return losses.avg, train_accuracy, metrics


def test(test_loader, model, criterion, epoch, train_loss, start, device, eval_mode=True):
    """
    :param test_loader:
    :param model:
    :param criterion:
    :param epoch:
    :param train_loss:
    :param start:
    :param device:
    :param eval_mode:
    :return:
    """
    model.eval()
    losses = AverageMeter()
    metrics_tracker = MetricTracker(args.num_classes)
    correct = 0
    total = 0
    desc = "Evaluating" if eval_mode else "Testing"

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=desc):
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))
            metrics_tracker.update(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum().item()

    train_accuracy = 100 * (correct / total)
    metrics = metrics_tracker.compute()
    metrics_tracker.reset()
    if eval_mode:
        return losses.avg, train_accuracy
    return train_accuracy, metrics


def main():
    """
    TODO Sort out and understand the tensorboard logging
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

    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")

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

    log_dir = (f"logs/runs/"
               f"{model.__class__.__name__}/"
               f"lr_{args.lr}_bs_{args.batch_size}/"
               f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
               f"{uuid4().hex[:6]}")
    writer = SummaryWriter(log_dir)  # todo fix the positioning of this later


if __name__ == '__main__':
    main()
