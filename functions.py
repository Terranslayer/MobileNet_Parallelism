from copy import deepcopy

import torch
import torch.nn as nn
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def do_epoch(model, optimizer, loss_func, data_loader,
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
        for ind, input in enumerate(data_loader, 1):
            X,y = input
            description = ''
            if title is not None:
                description += title
            description += f'Mode: {mode} |'
            X_tens, y_tens = torch.as_tensor(X, dtype=torch.float, device=DEVICE), \
                             torch.as_tensor(y, dtype=torch.long, device=DEVICE)
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


def train(model, train_loader, loss_func, optimizer, epoch_count=10,
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
                do_epoch(model, optimizer, loss_func, data,
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