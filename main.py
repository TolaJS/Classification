"""
author:         Tola Shobande
name:           main.py
date:           4/09/2024
description:    ...
"""

import os
import args
import logging
import torch
from datetime import datetime
from model import Classifier
from uuid import uuid4
from data import ProjectDataset
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
from utils import *
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Log')


def train(train_loader, model, criterion, optimiser, scaler, scheduler, device, epoch):
    """
    :param train_loader: training data loader
    :param model:
    :param criterion:
    :param optimiser:
    :param scaler:
    :param scheduler:
    :param device:
    :param epoch:
    :return:
    """
    model.train()
    losses = AverageMeter()  # Track the average loss
    metrics_tracker = MetricTracker(args.num_classes).to(device)
    correct = 0
    total = 0

    #   Pass images and labels to correct device
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        #   Zero the parameter gradient
        optimiser.zero_grad()

        #   Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        #   Backward pass and optimiser step
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        #   Metrics Update
        losses.update(loss.item(), images.size(0))  # Update loss
        metrics_tracker.update(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)  # Get predicted labels
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum().item()

        metrics = metrics_tracker.compute()
        log = format_log_message(mode='Train', i=i, epoch=epoch, loss=losses.avg, acc=100 * (correct / total),
                                 top1=metrics['top1_acc'], top5=metrics['top5_acc'], f1=metrics['f1'])
        logger.info(log)

    #   Update scheduler
    scheduler.step()

    train_accuracy = 100 * (correct / total)
    metrics = metrics_tracker.compute()
    metrics_tracker.reset()
    return losses.avg, train_accuracy, metrics


def test(test_loader, model, criterion, device, eval_mode=True):
    """
    :param test_loader:
    :param model:
    :param criterion:
    :param device:
    :param eval_mode:
    :return:
    """
    model.eval()
    losses = AverageMeter()
    metrics_tracker = MetricTracker(args.num_classes)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))
            metrics_tracker.update(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum().item()

    test_accuracy = 100 * (correct / total)
    metrics = metrics_tracker.compute()
    metrics_tracker.reset()
    if eval_mode:
        return losses.avg, test_accuracy
    return test_accuracy, metrics


def main():
    """Data Loading"""
    train_data = ProjectDataset(mode='train', root_dir='dataset', seed=args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    train_len = len(train_loader.dataset)

    val_data = ProjectDataset(mode='val', root_dir='dataset', seed=args.seed)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    val_len = len(val_loader.dataset)

    test_data = ProjectDataset(mode='test', root_dir='dataset', seed=args.seed)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    test_len = len(test_loader.dataset)

    """Model Initialisation"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Classifier(num_classes=args.num_classes)
    model.to(device)
    model_name = model.model.__class__.__name__

    if not os.path.exists("./checkpoint/"):
        os.mkdir("./checkpoint/")

    """Parameters"""
    #   OPTIMISER
    base_params = [  # Parameters in the model except FC layer
        param for name, param in model.named_parameters() if "fc" not in str(name)
    ]

    optimiser = torch.optim.Adam(
        params=[
            {"params": base_params},
            {"params": model.model.fc.parameters(), "lr": 0.01},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    #   SCHEDULER
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=args.gamma)
    #   LOSS
    criterion = torch.nn.CrossEntropyLoss()
    #   SCALER
    scaler = torch.cuda.amp.GradScaler()

    log_dir = (f"logs/runs/"
               f"{model.__class__.__name__}/"
               f"lr_{args.lr}_bs_{args.batch_size}/"
               f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
               f"{uuid4().hex[:6]}")
    writer = SummaryWriter(log_dir)
    handler = logging.FileHandler(f'{log_dir}/log.txt')
    logger.addHandler(handler)

    #   EXECUTION
    if not args.eval_mode:
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path))
            model.to(device)
        else:
            raise ValueError("Model path not set in args.py")
        test_accuracy, test_metrics = test(test_loader, model, criterion, device, args.eval_mode)

        print_summary(logger, model_name, train_len, val_len, test_len)
        logger.info("=> Testing Results")
        logger.info("Top-1 Accuracy: {:.2f}%".format(test_metrics['top1_acc']))
        logger.info("Top-5 Accuracy: {:.2f}%".format(test_metrics['top5_acc']))
        logger.info("F1-Score: {:.2f}%".format(test_metrics['f1']))
    else:
        print_summary(logger, model_name, train_len, val_len, test_len)
        logger.info("=> Start Training")
        for epoch in range(args.epochs):
            train_loss, train_acc, train_metrics = train(train_loader, model, criterion, optimiser,
                                                         scaler, scheduler, device, epoch + 1)
            val_loss, val_acc = test(val_loader, model, criterion, device)

            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Accuracy', train_acc, epoch)
            writer.add_scalar('Train/Top-1 Accuracy', train_metrics['top1_acc'], epoch)
            writer.add_scalar('Train/Top-5 Accuracy', train_metrics['top5_acc'], epoch)
            writer.add_scalar('Train/F1-Score', train_metrics['f1'], epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Accuracy', val_acc, epoch)

            filename = f"Model-{model_name}-lr_{args.lr}_bs_{args.batch_size}-E{epoch + 1}.pth"
            save_path = os.path.join(args.save_path, filename)
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
