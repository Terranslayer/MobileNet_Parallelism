import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset


def train_test_split(data, train_size, stratify=None):
    """
    Split data on two folds
    :param data: (iterable) data for split
    :param train_size: (float) should be (0, 1)
    :param stratify: (labels) stratify folds
    :return: (tuple(np.array)) train, test
    """
    if stratify is None:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split_index = int(len(indices) * train_size)
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        return data[train_indices], data[test_indices]
    else:
        unique_values = np.unique(stratify)
        train_data = []
        test_data = []
        for u_value in unique_values:
            u_train, u_test = train_test_split(data[stratify == u_value], train_size)
            train_data.append(u_train)
            test_data.append(u_test)
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        return np.concatenate(train_data), np.concatenate(test_data)


class MyDataLoader:
    def __init__(self, data, batch_size, indices=None, shuffle=False):
        """
        Create batches
        :param data: (iterable)
        :param batch_size: (int)
        :param indices: (list or np.array) Default None
        :param shuffle: (bool) Default None
        """
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data = data
        if indices is None:
            self.data_len = len(data)
            self.indices = np.arange(self.data_len)
        else:
            self.data_len = len(indices)
            self.indices = np.array(indices)
        self.len_ = int(np.ceil(self.data_len / batch_size))

        self.X_shape = tuple(self.data[0][0].shape)

    def __len__(self):
        return self.len_

    def idx_to_class(self, indices):
        """
        convert indexes to label
        :param indices: (iterable) 1D array
        :return: (list) label
        """
        classes = []
        for index in indices:
            classes.append(self.data.classes[index])
        return classes

    def create_batch(self, indices):
        batch_size = len(indices)
        X_batch = torch.zeros((batch_size, *self.X_shape))
        y_batch = torch.zeros(batch_size)
        for n_batch, index in enumerate(indices):
            X, y = self.data[index]
            X_batch[n_batch] = X
            y_batch[n_batch] = y
        return X_batch, y_batch

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for n_batch in range(self.len_):
            start_index = n_batch * self.batch_size
            end_index = min(self.data_len, start_index + self.batch_size)
            batch_indices = self.indices[start_index: end_index]
            X_batch, y_batch = self.create_batch(batch_indices)
            yield X_batch, y_batch


def load_class_name(path):
    """
    load class names from file
    :param path: (str) file path
    :return: (list) labels
    """
    file = open(path, 'rb')
    out = pickle.load(file)
    file.close()
    return out


def create_test_dataset(data_folder):
    """
    Loads a CIFAR-100 test dataset and selects
    100 random images, then saves them as .jpg files
    :param data_folder: folder to save images
    """
    dataset = datasets.CIFAR100('data/', False, download=True)
    Path(data_folder).mkdir()
    for i in range(100):
        img, label = random.choice(dataset)
        img.save(data_folder + str(i) +
                 '_' + dataset.classes[label] + '.jpg')


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class Dataset:
    def __init__(self, data_folder, transform):
        """
        Load .jpg files from data folder
        :param data_folder: (str)
        :param transform: (torchvision.transforms)
        """
        self.classes = load_class_name('classes_name.pkl')
        data_dir = Path(data_folder)
        self.files = list(data_dir.rglob('*.jpg'))
        if len(self.files) == 0:
            create_test_dataset(data_folder)
            self.files = list(data_dir.rglob('*.jpg'))
        print(len(self.files))
        self.len_ = len(self.files)
        self.file_names = [path.name for path in self.files]
        self.transform = transform

    def __len__(self):
        return self.len_

    @staticmethod
    def load_sample(file):
        image = Image.open(file)
        return image

    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        x = self.transform(x)
        y = self.file_names[index]
        return x, y