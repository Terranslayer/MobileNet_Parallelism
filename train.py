from importlib import reload

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from datafunc import MyDataLoader, train_test_split
from model import MobileNetV3
from torch.nn.parallel import DistributedDataParallel as DDP
import timeit
import torch.distributed as dist
import torch.multiprocessing as mp
import socket
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
# dataset settings
batch_size = 256
IM_SIZE = 224  # resize image
NORMALIZE = ([0.485, 0.456, 0.406],
             [0.229, 0.224, 0.225])


train_transformer = transforms.Compose([
    transforms.Resize(IM_SIZE),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*NORMALIZE)
])


test_transformer = transforms.Compose([
    transforms.Resize(IM_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(*NORMALIZE)
])

cifar_train = datasets.CIFAR100('data/',
                             transform=train_transformer,
                             download=True)

cifar_val = datasets.CIFAR100('data/',
                               transform=test_transformer,
                               train=True)

train_indices, val_indices = \
    train_test_split(np.arange(len(cifar_train)), .75, cifar_train.targets)

# model parameters
CLASSES_COUNT = len(cifar_train.classes)
ALPHA = 1.
ARCHITECTURE = 'small'
DROPOUT = 0.8

from functions import train, accuracy

def setup(rank, world_size):
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    sock = socket.socket()
    sock.bind(('',0))
    port = sock.getsockname()[1]
    os.environ['Master_ADDR'] = 'local host'
    os.environ['Master_PORT'] = '12355'
    dist.init_process_group('nccl',rank=rank,world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def model_init(rank,world_size):
    setup(rank,world_size)
    train_loader = MyDataLoader(cifar_train, batch_size, train_indices, shuffle=True)
    val_loader = MyDataLoader(cifar_val, batch_size, val_indices, shuffle=True)
    mobilenet = MobileNetV3()
    mobilenet.create_model(classes_count=CLASSES_COUNT, architecture=ARCHITECTURE,alpha=ALPHA, dropout=DROPOUT)
    # mobilenet = nn.DataParallel(mobilenet)
    mobilenet.to(rank)
    mobilenet = DDP(mobilenet, device_ids=[rank])
    LR = 1e-2
    OPTIM_MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-5
    # backend = 'nccl'
    # world_size = 2
    # processes = []
    # mp.set_start_method("spawn")


    optimizer = torch.optim.Adam(mobilenet.parameters(),
                                lr=LR,
                                weight_decay = WEIGHT_DECAY
                                )

    loss_func = nn.CrossEntropyLoss()

    EPOCHS = 2

    factor = 0.5
    patience = 2
    threshold = 0.001

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=factor, patience=patience,
        verbose=True, threshold=threshold
    )

    train_history, best_parameters = \
        train(mobilenet, train_loader, loss_func, optimizer, device,
                EPOCHS, accuracy, val_loader, scheduler)

    torch.save(mobilenet.module.state_dict(),'model.pkl')

    cleanup()

def run(model_init, world_size):
    mp.spawn(model_init, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run(model_init=model_init, world_size=2)
    end.record()

    torch.cuda.synchronize()
    print("Time is: ", start.elapsed_time(end))


'''
new_model = MobileNetV3()
new_model.load_model('model.pkl')
new_model = new_model.to(device)

optimizer = torch.optim.Adam(new_model.parameters(),
                            lr=LR,
                            weight_decay = WEIGHT_DECAY
                            )

loss_func = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=factor, patience=patience,
    verbose=True, threshold=threshold
)

EPOCHS = 1
train_history, best_parameters = \
    train(new_model, train_loader, loss_func, optimizer,
          EPOCHS, accuracy, val_loader, scheduler)
'''