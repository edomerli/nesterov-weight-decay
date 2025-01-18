
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import wandb

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
    run_name = f"{'nesterovwd' if config.nesterov_wd else 'original'}_{config.optimizer}_wd{config.wd}_lr{config.lr}_epochs{config.epochs}"
    wandb.init(project="nesterov-weight-decay", name=run_name, config=vars(config), sync_tensorboard=True)

    wandb.define_metric("train/epoch")
    wandb.define_metric("train/loss", step_metric="train/epoch")
    wandb.define_metric("train/accuracy", step_metric="train/epoch")
    wandb.define_metric("train/params_l2", step_metric="train/epoch")
    wandb.define_metric("train/lr", step_metric="train/epoch")
    wandb.define_metric("train/weight_decay", step_metric="train/epoch")

    wandb.define_metric("test/epoch")
    wandb.define_metric("test/loss", step_metric="test/epoch")
    wandb.define_metric("test/accuracy", step_metric="test/epoch")
    wandb.define_metric("test/best_accuracy", step_metric="test/epoch")
    wandb.define_metric("test/cosine_similarity", step_metric="test/epoch")
