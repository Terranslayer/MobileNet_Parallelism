from copy import deepcopy

import torch
import torch.nn as nn
from tqdm import tqdm
import time
import torch.distributed as dist
import shutil

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EMA(nn.Module):
    """
    Exponential mean average
    """
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

'''
def do_epoch(model, optimizer, loss_func, data_loader, device,
             mode='T', metric=None, title=None):
    """
    Compute one epoch
    :param model: (nn.Module) model
    :param optimizer: (torch.optim) optimization method. Ignored if mode='V'
    :param loss_func: (func) loss functions
    :param data_loader: (MyDataLoader) val batches generator (X, y). Default None
    :param mode: (str) 'T' - Train or 'V' - Validate. Default 'T'
    :param metric: (func) target metric
    :param title: (str) description in progress bar
    :return:
        epoch_loss: mean loss
        epoch_metric: mean metric
    """
    if mode not in ['V', 'T']:
        raise ValueError('mode should be "T" or "V"')
    # History
    epoch_loss = 0.
    epoch_metric = 0.

    ema = EMA(0.999)
    if mode == 'T':
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

    with tqdm(total=len(data_loader)) as progress_bar:
        for ind, (X, y) in enumerate(data_loader, 1):
            description = ''
            if title is not None:
                description += title
            description += f'Mode: {mode} |'
            X_tens, y_tens = torch.as_tensor(X, dtype=torch.float, device="cuda:"+str(device)), \
                             torch.as_tensor(y, dtype=torch.long, device="cuda:"+str(device))
            X_tens=X_tens.cuda("cuda:"+str(device))
            y_tens=y_tens.cuda("cuda:"+str(device))
            predict = model(X_tens).squeeze(dim=-1)
            loss = loss_func(predict, y_tens)
            epoch_loss += loss.item()
            description += f'Loss: {epoch_loss / ind: 7.4} |'
            # backward
            if mode == 'T':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.data = ema(name, param.data)
            #  metric calculate
            if metric is not None:
                epoch_metric += metric(predict, y_tens)
                description += f'Metric: {epoch_metric / ind: 7.4} |'

            progress_bar.set_description(description)
            progress_bar.update()
    return epoch_loss / len(data_loader), epoch_metric / len(data_loader)
'''

'''
def train(model, train_loader, loss_func, optimizer, device, epoch_count=10,
          metric=None, val_loader=None, scheduler=None):
    """
    Training model
    :param model: (torch.nn.Module) model for train
    :param train_loader: (MyDataLoader) train batches generator (X, y)
    :param loss_func: (func) loss functions
    :param optimizer: (torch.optim) optimization method
    :param epoch_count: (int) epochs count. Default 10
    :param metric: (func) target metric
    :param val_loader: (MyDataLoader) val batches generator (X, y). Default None
    :param scheduler: (torch.utils)
    :return:
        history_info: dict of training history consist "Tloss", "Tmetric",
                    "Vloss", "Vmetric"
        best_model_param: model parameters at the highest Vmetric value
    """
    # Train_history
    history_info = {'Tloss': [], 'Tmetric': [],
                    'Vloss': [], 'Vmetric': []}
    # best Val_score and model params
    best_score = 0.
    best_model_param = {}

    datasets = {}

    if train_loader is not None:
        datasets.update({'T': train_loader})
    if val_loader is not None:
        datasets.update({'V': val_loader})

    for epoch in range(epoch_count):
        for mode, data in datasets.items():
            # title for progress bar
            title = f'[{epoch+1: 3}/{epoch_count}]|'
            model.train(mode == 'T')
            epoch_loss, epoch_metric = \
                do_epoch(model, optimizer, loss_func, data, device,
                         mode, metric, title)
            history_info[mode + 'loss'].append(epoch_loss)
            history_info[mode + 'metric'].append(epoch_metric)

            if metric is not None:
                # save best metric value and model parameters
                if best_score < epoch_metric and mode == 'V':
                    best_score = epoch_metric
                    best_model_param = deepcopy(model.state_dict())

            # scheduler step
            if scheduler is not None and mode == 'V':
                scheduler.step(epoch_metric)

    return history_info, best_model_param
'''

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs # this average is necessary!
    return rt

def train(train_loader, model, criterion, optimizer, epoch, gpu, ngpus_per_node, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(gpu, non_blocking=True) # 把当前mini-batch的x和y，推入gpu内存
        target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, gpu, topk=(1, 5))
        dist.barrier()

        reduced_loss = reduce_mean(loss, ngpus_per_node)
        reduced_acc1 = reduce_mean(acc1, ngpus_per_node)
        reduced_acc5 = reduce_mean(acc5, ngpus_per_node)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, gpu, ngpus_per_node, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, gpu, topk=(1, 5))
            dist.barrier()

            reduced_loss = reduce_mean(loss, ngpus_per_node)
            reduced_acc1 = reduce_mean(acc1, ngpus_per_node)
            reduced_acc5 = reduce_mean(acc5, ngpus_per_node)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = state['arch'] + '.' + filename
    torch.save(state, filename)
    if is_best:
        filename2 = state['arch'] + '.model_best.pth.tar'
        shutil.copyfile(filename, filename2)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, local_rank, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        predy = torch.max(output, 1)[1].data.squeeze()
        acc = (predy == target).sum().item()/float(target.size(0))
        acc = torch.tensor(acc).cuda(local_rank)
        res = []
        res.append(acc)
        res.append(acc)
        return res

'''
def accuracy(predict_proba, ground_truth):
    """
    Compute accuracy
    :param predict_proba: (torch.Tensor) prediction probability 2D dim
    :param ground_truth: (torch.Tensor) ground truth labels
    :return: (float) accuracy
    """
    label_index = torch.argmax(predict_proba, dim=-1)
    true_predict = (label_index == ground_truth).sum().item()
    return true_predict / ground_truth.size()[0]
    '''