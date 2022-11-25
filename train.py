from importlib import reload

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from datafunc import DatasetFromSubset
from model import MobileNetV3
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import socket
import os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset settings
batch_size = 256
print_freq = 10
EPOCHS = 2
start_epoch = 0
LR = 1e-2
OPTIM_MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
IM_SIZE = 224  # resize image
NORMALIZE = ([0.485, 0.456, 0.406],
             [0.229, 0.224, 0.225])
evaluate = True


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

cifar = datasets.CIFAR100('data/',
                             download=True)

'''
cifar_val = datasets.CIFAR100('data/',
                               transform=test_transformer,
                               train=False)

train_indices, val_indices = \
    train_test_split(np.arange(len(cifar_train)), .75, cifar_train.targets)
    '''

# model parameters
CLASSES_COUNT = len(cifar.classes)
ALPHA = 1.
ARCHITECTURE = 'small'
DROPOUT = 0.8

from functions import train, validate, save_checkpoint

def setup(rank, nprocs):
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    sock = socket.socket()
    sock.bind(('',0))
    port = sock.getsockname()[1]
    init_method = 'tcp://' + str(IPAddr) + ':' + '29500'
    dist.init_process_group('nccl', init_method=init_method,rank=rank,world_size=nprocs)

def cleanup():
    dist.destroy_process_group()

def model_init(rank,nprocs):
    setup(rank,nprocs)
    splited_batch_size = int(batch_size/nprocs) #seperate batch size according to N of processors
    train_subset, val_subset = random_split(cifar, [0.75, 0.25]) #split dataset into train & test
    # apply corespond transformer to train & test dataset, apply sampler, create dataloder
    train_dataset = DatasetFromSubset(train_subset, transform=train_transformer)
    val_dataset = DatasetFromSubset(val_subset, transform=test_transformer)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = DataLoader(train_dataset,
    batch_size=splited_batch_size,
    num_workers=2,
    pin_memory=True,
    sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset,
    batch_size=splited_batch_size,
    num_workers=2,
    pin_memory=True,
    sampler=val_sampler)

    # train_loader = MyDataLoader(cifar_train, splited_batch_size, train_indices, shuffle=True)
    # val_loader = MyDataLoader(cifar_val, batch_size, val_indices, shuffle=True)


    mobilenet = MobileNetV3()
    mobilenet.create_model(classes_count=CLASSES_COUNT, architecture=ARCHITECTURE,alpha=ALPHA, dropout=DROPOUT)
    torch.cuda.set_device(rank)
    mobilenet.cuda(rank)
    mobilenet = DDP(mobilenet, device_ids=[rank])


    optimizer = torch.optim.Adam(mobilenet.parameters(),
                                lr=LR,
                                weight_decay = WEIGHT_DECAY
                                )

    loss_func = nn.CrossEntropyLoss().cuda(rank)

    '''
    factor = 0.5
    patience = 2
    threshold = 0.001

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=factor, patience=patience,
        verbose=True, threshold=threshold
    )
    '''

    if evaluate == True:
        validate(val_loader, mobilenet,loss_func, rank, nprocs, print_freq)

    for epoch in range(start_epoch, EPOCHS):
        train_sampler.set_epoch(epoch) # ensure shuffle in every epoch
        val_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, LR)

        #train for one epoch
        train(train_loader, mobilenet, loss_func, optimizer, epoch, rank, nprocs, print_freq)

        #evaluate on validation set
        acc1 = validate(val_loader, mobilenet, loss_func, rank, nprocs, print_freq)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': 'mobilenet',
                    'state_dict': mobilenet.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)

    cleanup()

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run(model_init, nprocs):
    mp.spawn(model_init, args=(nprocs,), nprocs=nprocs)

if __name__ == "__main__":
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nprocs = torch.cuda.device_count() # get gpu number
    start.record()
    run(model_init=model_init, nprocs=nprocs)
    end.record()

    torch.cuda.synchronize()
    print("Total Time is: ", start.elapsed_time(end))


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