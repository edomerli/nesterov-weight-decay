
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import wandb
import torchvision
from torchvision import transforms

def seed_everything(seed):
    """Seed all sources of randomness for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_wandb(config):
    wandb.login(key="14a7d0e7554bbddd13ca1a8d45472f7a95e73ca4")
    run_name = f"{config.dataset}_{config.model_name}_{'nesterovwd' if config.nesterov_wd else ''}_{config.optimizer}_wd{config.wd}_lr{config.lr}_ep{config.epochs}"
    wandb.init(project="nesterov-weight-decay", name=run_name, config=vars(config), sync_tensorboard=True)

    wandb.define_metric("train/epoch")
    wandb.define_metric("train/loss", step_metric="train/epoch")
    wandb.define_metric("train/error", step_metric="train/epoch")
    wandb.define_metric("train/params_l2_squared", step_metric="train/epoch")
    wandb.define_metric("train/grad_l2_squared", step_metric="train/epoch")
    wandb.define_metric("train/lr", step_metric="train/epoch")
    wandb.define_metric("train/weight_decay", step_metric="train/epoch")

    wandb.define_metric("test/epoch")
    wandb.define_metric("test/loss", step_metric="test/epoch")
    wandb.define_metric("test/error", step_metric="test/epoch")
    wandb.define_metric("test/best_error", step_metric="test/epoch")

DATASET_TO_MEAN_STD = {
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

DATASET_TO_CLASS = {
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
}

DATASET_TO_NUM_CLASSES = {
    "CIFAR10": 10,
    "CIFAR100": 100,
}

def setup_data(config):
    mean, std = DATASET_TO_MEAN_STD[config.dataset]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset_class = DATASET_TO_CLASS[config.dataset]
    trainset = dataset_class(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = dataset_class(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    num_classes = DATASET_TO_NUM_CLASSES[config.dataset]

    return trainloader, testloader, num_classes