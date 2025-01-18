import torch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, AdamW, Adam
from torchvision import transforms
from tqdm.autonotebook import tqdm
from math import sqrt
import torchvision
import json
import wandb
from argparse import Namespace

from nesterov_wd.adam_nesterov_wd import AdamNesterovWD
from nesterov_wd.sgd_nesterov_wd import SGD_NesterovWD
from utils import seed_everything, setup_wandb, init_params
from custom_schedulers import CustomMultiStepLR, CustomRandomLR

# --------------------------------------------------------------------------------
# SETUP EXPERIMENT
# --------------------------------------------------------------------------------
config_dict = {
    "epochs": 500,
    "lr": 0.1,
    "wd": 0.01,
    "nesterov_wd": False,
    "optimizer": "SGD",
    "scheduler": "multistep",
    # less important params
    "lr_flow": 1e-3,
    "first_decay": 0.5,
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
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet.resnet18(num_classes=10).to(device)
init_params(model)
if device.type == 'cuda':
    model = torch.nn.DataParallel(model)
wandb.watch(model, log='all')

print('==> Optimizer..')
if config.optimizer == 'SGD':
    if config.nesterov_wd:
        opt = SGD_NesterovWD(model.parameters(), lr=config.lr, weight_decay=config.wd)
    else:
        opt = SGD(model.parameters(), lr=config.lr, weight_decay=config.wd)
else:
    if config.optimizer == 'AdamW':
        if config.nesterov_wd:
            opt = AdamNesterovWD(model.parameters(), lr=config.lr, weight_decay=config.wd)
        else:
            opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    elif config.optimizer == 'Adam_L2_penalty':
        opt = Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

loss_fn = CrossEntropyLoss()

print('==> Scheduler..')
if config.lr_flow is not None:
    gamma_lr = config.lr_flow/config.lr
else:
    gamma_lr = 1.0
# TODO: try with Random LR in a range at each epoch
if config.scheduler == 'multistep':
    scheduler = CustomMultiStepLR(opt, [int(config.epochs*config.first_decay)], gamma_lr=gamma_lr, gamma_wd=1.0)
elif config.scheduler == 'random':
    scheduler = CustomRandomLR(opt, [5e-2, 1e-4])
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs, eta_min=0)


print('==> Training..')
best_acc = 0  # best test accuracy

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    cosine_similarity = 0
    for inputs, targets in tqdm(trainloader, leave=False):
        params_before = [param.detach().clone() for param in model.parameters()]

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

        params_after = [param.detach().clone() for param in model.parameters()]
        cosine_similarity += sum([torch.dot(param1.view(-1), param2.view(-1))/(param1.norm()*param2.norm())
                                 for param1, param2 in zip(params_before, params_after)])/len(params_before)

    l2 = sum([(param**2).sum()
             for param in model.parameters()]).detach().cpu().item()
    wandb.log({
        "train/accuracy": 100.*correct/total,
        "train/loss": train_loss/len(trainloader),
        "train/params_l2": sqrt(l2),
        "train/cosine_similarity": cosine_similarity/len(trainloader),
        "train/lr": opt.param_groups[0]['lr'],
        "train/weight_decay": opt.param_groups[0]['weight_decay'],
        "train/epoch": epoch,
    })

def test(epoch):
    global best_acc
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

    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc

    wandb.log({
        "test/loss": test_loss/len(testloader),
        "test/accuracy": acc,
        "test/best_accuracy": best_acc,
        "test/epoch": epoch,
    })


for epoch in tqdm(range(config.epochs)):
    train(epoch)
    test(epoch)    
    scheduler.step()