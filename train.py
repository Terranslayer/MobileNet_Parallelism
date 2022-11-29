import csv
import torch
import time
import torch.nn as nn
from torchvision import datasets, transforms
from datafunc import DatasetFromSubset
from model import MobileNetV3
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import random
import os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from datetime import timedelta

# dataset settings
batch_size = 256
print_freq = 10
EPOCHS = 4
start_epoch = 0
LR = 1e-2
debug = False # print some parameters when it's on
OPTIM_MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
dist_file = "dist_file" #file name of init_method of init_process_group
best_acc1 = .0
IM_SIZE = 224  # resize image
NORMALIZE = ([0.485, 0.456, 0.406],
             [0.229, 0.224, 0.225])
evaluate = False
'''
set backend to "nccl" to get the best performance, but this will result hang if n_nodes != n_gpus in case HPC
set backendd to gloo can run the program when n_nodes != n_gpus, however this method will randomly select 
used ports which will result in connection refused error, so you may need to try several times to get a succesful
run. This is about due to gloo's low level communication code, so I cannot do much with it
'''
backend = "nccl"


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

# model parameters
CLASSES_COUNT = len(cifar.classes)
ALPHA = 1.
ARCHITECTURE = 'small'
DROPOUT = 0.8

from functions import train, validate, save_checkpoint

def model_init(gpu,ngpus_per_node,world_rank,dist_url,world_size):
    global best_acc1
    # print("Get here!")
    rank = world_rank * ngpus_per_node + gpu

    '''
    # TCP init
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    sock = socket.socket()
    sock.bind(('',0))
    port = sock.getsockname()[1]
    init_method = 'tcp://' + str(IPAddr) + ':' + '29501'
    '''

    #test code
    if debug:
        print("Inside Model Init:")
        print("   GPU: ", gpu)
        # print("   Ngpus_per_node: ", ngpus_per_node)
        #print("local rank: ", local_rank)
        print("   Rank: ", rank)
        print("   Current NodeID: ",world_rank)
        #print("world size: ", world_size)
        #print("dist url: ", dist_url)

    # print("Can get here!!!")
    dist.init_process_group(backend=backend, init_method=dist_url,rank=rank,world_size=world_size,timeout=timedelta(seconds=60))
    # print("But cannot get here???")
    splited_batch_size = int(batch_size/ngpus_per_node) #seperate batch size according to N of processors
    train_subset, val_subset = random_split(cifar, [0.75, 0.25]) #split dataset into train & test
    # apply corespond transformer to train & test dataset, apply sampler, create dataloder
    train_dataset = DatasetFromSubset(train_subset, transform=train_transformer)
    val_dataset = DatasetFromSubset(val_subset, transform=test_transformer)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = DataLoader(train_dataset,
    batch_size=splited_batch_size,
    shuffle=(train_sampler is None),
    num_workers=1,
    pin_memory=True,
    sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset,
    batch_size=splited_batch_size,
    num_workers=1,
    pin_memory=True,
    sampler=val_sampler)


    mobilenet = MobileNetV3()
    mobilenet.create_model(classes_count=CLASSES_COUNT, architecture=ARCHITECTURE,alpha=ALPHA, dropout=DROPOUT)
    torch.cuda.set_device(gpu)
    mobilenet.cuda(gpu)
    mobilenet = DDP(mobilenet, device_ids=[gpu])


    optimizer = torch.optim.Adam(mobilenet.parameters(),
                                lr=LR,
                                weight_decay = WEIGHT_DECAY
                                )

    loss_func = nn.CrossEntropyLoss().cuda(gpu)

    if evaluate == True:
        validate(val_loader, mobilenet,loss_func, gpu, ngpus_per_node, print_freq)

    log_csv = "distributed_csv"
    start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        np.random.seed(epoch)
        random.seed(epoch)

        train_sampler.set_epoch(epoch) # ensure shuffle in every epoch
        val_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, LR)

        #train for one epoch
        train(train_loader, mobilenet, loss_func, optimizer, epoch, gpu, ngpus_per_node, print_freq)

        #evaluate on validation set
        acc1 = validate(val_loader, mobilenet, loss_func, gpu,ngpus_per_node, print_freq)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        epoch_end = time.time()

        with open(log_csv, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start)), epoch_end - epoch_start]
            csv_write.writerow(data_row)

        if rank % ngpus_per_node == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': 'mobilenet',
                    'state_dict': mobilenet.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)

    end = time.time()
    t_time = end - start
    print("The Elapsed Time is: ", t_time)
    dist.destroy_process_group()

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":

    if False:
        #os env test
        import pprint
        env_var = os.environ
        print("User's Environment variable:")
        pprint.pprint(dict(env_var), width = 1)

    #get slurm parameter
    node_id = int(os.environ["SLURM_NODEID"])
    w_rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])
    ngpus_per_node = torch.cuda.device_count()
    job_id = os.environ["SLURM_JOBID"]
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    #create dist train file
    dist_url = "file://{}.{}".format(os.path.realpath(dist_file), job_id)

    if debug:
        print("dist-url:{} at PROCID {} / {}".format(dist_url, w_rank, world_size))

    if debug:
        print("In local machine, before spawn():")
        print("world_rank: ", w_rank)
        print("world size: ", world_size)
        print("ngpus per node: ", ngpus_per_node)
        print("node id: ", node_id)
    context = mp.spawn(model_init, args=(ngpus_per_node,node_id,dist_url,world_size), nprocs=ngpus_per_node,join=False)
    context.join(10)