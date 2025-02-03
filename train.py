import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Adam
from tqdm.autonotebook import tqdm
from math import sqrt
import json
import wandb
from argparse import Namespace

from resnets import ResNet18, ResNet34
from nesterov_wd.adam_nesterov_wd import AdamNesterovWD
from utils import seed_everything, setup_wandb, setup_data

# --------------------------------------------------------------------------------
# SETUP EXPERIMENT
# --------------------------------------------------------------------------------
config_dict = {
    "dataset": "CIFAR10",
    "model_name": "ResNet18",
    "epochs": 200,
    "lr": 0.001,
    "wd": 5e-4,
    "nesterov_wd": False,
    "optimizer": "Adam",
    # less important params
    "torch_version": torch.__version__,
    "seed": 6,
}

print("==> Configuration")
print(json.dumps(config_dict, indent=4))
config = Namespace(**config_dict)

print('==> Setting up wandb..')
setup_wandb(config)

print('==> Seeding..')
seed_everything(config.seed)

print('==> Preparing data..')
trainloader, testloader, num_classes = setup_data(config)

print('==> Building model..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config.model_name == 'ResNet18':
    model = ResNet18(num_classes=num_classes).to(device)
elif config.model_name == 'ResNet34':
    model = ResNet34(num_classes=num_classes).to(device)
else:
    raise NotImplementedError(f"Model {config.model_name} not implemented")
if device.type == 'cuda':
    model = torch.nn.DataParallel(model)
wandb.watch(model, log='all')

print('==> Optimizer..')
if config.optimizer == 'Adam':
    opt = Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
elif config.optimizer == 'AdamW':
    # NOTE: for AdamW, weight_decay is divided by lr as in the SWD paper 
    # ("On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm Perspective")
    if config.nesterov_wd:
        opt = AdamNesterovWD(model.parameters(), lr=config.lr, weight_decay=config.wd/config.lr)
    else:
        opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd/config.lr)
else:
    raise NotImplementedError(f"Optimizer {config.optimizer} not implemented")

loss_fn = CrossEntropyLoss()

print('==> Scheduler..')
# NOTE: epochs must be fixed to 200 for these lambdas to work as expected
if config.dataset == 'CIFAR10':
    lambda_lr = lambda epoch: 0.1 ** (epoch // 80) # lr decay at 80 and 160 epochs
elif config.dataset == 'CIFAR100':
    lambda_lr = lambda epoch: 0.1 ** (epoch // 100 + epoch // 150)  # lr decay at 100 and 150 epochs 
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr)

print('==> Training..')
best_err = 100  # best test error

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in tqdm(trainloader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # compute squared l2 norm of parameters and gradients
    l2_squared = 0
    l2_grad_squared = 0
    for p in model.parameters():
        l2_squared += p.detach().data.norm().cpu().item()**2
        l2_grad_squared += p.grad.detach().data.norm().cpu().item()**2

    wandb.log({
        "train/error": 100.* (1.0 - correct/total),
        "train/loss": train_loss/total,
        "train/params_l2_squared": l2_squared,
        "train/grad_l2_squared": l2_grad_squared,
        "train/lr": opt.param_groups[0]['lr'],
        "train/weight_decay": opt.param_groups[0]['weight_decay'],
        "train/epoch": epoch,
    })

def test(epoch):
    global best_err
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    err = 100.* (1.0 - correct/total)
    if err < best_err:
        best_err = err

    wandb.log({
        "test/loss": test_loss/total,
        "test/error": err,
        "test/best_error": best_err,
        "test/epoch": epoch,
    })


for epoch in tqdm(range(config.epochs)):
    train(epoch)
    test(epoch)    
    scheduler.step()

wandb.finish()