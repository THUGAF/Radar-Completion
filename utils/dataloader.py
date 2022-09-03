import os
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import numpy as np
import netCDF4 as nc


class TrainingDataset(Dataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset.
        input_steps (int): Number of input steps.
        forecast_steps: (int): Number of forecast steps.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.
    """

    def __init__(self, root: str, input_steps: int, forecast_steps: int, lon_range: List[int], 
                 lat_range: List[int]):
        super().__init__()
        self.root = root
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.lon_range = lon_range
        self.lat_range = lat_range

        self.sample_num = []
        self.files = []
        self.total_steps = self.input_steps + self.forecast_steps
        self.date_list = sorted(os.listdir(self.root))
        for date in self.date_list:
            file_list = sorted(os.listdir(os.path.join(self.root, date)))
            self.sample_num.append(len(file_list) - self.total_steps + 1)
            for file_ in file_list:
                self.files.append(os.path.join(self.root, date, file_))
        
        self.sample_num = np.array(self.sample_num)
        self.sample_cumsum = np.cumsum(self.sample_num)

    def __getitem__(self, index):
        files = self.locate_files(index)
        tensor, timestamp = self.load_nc(files)
        return tensor, timestamp
    
    def __len__(self,):
        return sum(self.sample_num)
        
    def locate_files(self, index):
        date_order = np.where(index - self.sample_cumsum < 0)[0][0]
        if date_order == 0:
            file_anchor = index
        else:
            file_anchor = index + date_order * (self.total_steps - 1)
        files = self.files[file_anchor: file_anchor + self.total_steps]
        return files
    
    def load_nc(self, files):
        tensor = []
        timestamp = []
        for file_ in files:
            nc_file = nc.Dataset(file_)
            dbz = torch.from_numpy(nc_file.variables['DBZ'][:])
            dbz = dbz[:, self.lat_range[0]: self.lat_range[1], self.lon_range[0]: self.lon_range[1]]
            second = float(nc_file.variables['time'][:])
            tensor.append(dbz)
            timestamp.append(np.int64(second))
            nc_file.close()
        tensor = torch.stack(tensor)
        timestamp = torch.LongTensor(timestamp)
        return tensor, timestamp


class SampleDataset(TrainingDataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset.
        input_steps (int): Number of input steps.
        forecast_steps (int): Number of input steps. 
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.
    """

    def __init__(self, root: str, sample_index: int, input_steps: int, forecast_steps: int, 
                 lon_range: List[int], lat_range: List[int]):
        super().__init__(root, input_steps, forecast_steps, lon_range, lat_range)
        self.sample_index = sample_index

    def __getitem__(self, index: int):
        files = self.locate_files(self.sample_index)
        tensor, timestamp = self.load_nc(files)
        return tensor, timestamp

    def __len__(self):
        return 1


def load_data(root: str, input_steps: int, forecast_steps: int, batch_size: int, num_workers: int, 
              train_ratio: float, valid_ratio: float, lon_range: List[int], lat_range: List[int]):
    r"""Load training and test data.

    Args:
        root (str): Path to the dataset.
        input_steps (int): Number of input steps.
        forecast_steps (int): Number of forecast steps.
        batch_size (int): Batch size.
        num_workers (int): Number of processes.
        train_ratio (float): Training ratio of the whole dataset.
        valid_ratio (float): Validation ratio of the whole dataset.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.
    Returns:
        DataLoader: Dataloader for training and test.
    """

    dataset = TrainingDataset(root, input_steps, forecast_steps, lon_range, lat_range)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    train_node = round(train_ratio * dataset_size)
    val_node = round(valid_ratio * dataset_size)
    train_indices = indices[:train_node]
    val_indices = indices[train_node: train_node + val_node]
    test_indices = indices[train_node + val_node:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, pin_memory=True)

    print('\nTrain Loader')
    print('----Batch Num:', len(train_loader))

    print('\nVal Loader')
    print('----Batch Num:', len(val_loader))

    print('\nTest Loader')
    print('----Batch Num:', len(test_loader))

    return train_loader, val_loader, test_loader


def load_sample(root: str, sample_index: int, input_steps: int, forecast_steps: int, 
                lon_range: List[int], lat_range: List[int]):
    r"""Load sample data.

    Args:
        root (str): Path to the dataset.
        sample_index (int): Index of the sample. Default is None, meaning the last sample is selected.
        input_steps (int): Number of input steps.
        forecast_steps (int): Number of forecast steps.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.

    Returns:
        DataLoader: Dataloader for sample.
    """

    sample_set = SampleDataset(root, sample_index, input_steps, forecast_steps, lon_range, lat_range)
    sample_loader = DataLoader(sample_set, batch_size=1)
    return sample_loader
