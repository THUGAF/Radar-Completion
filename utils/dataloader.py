import os
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import netCDF4 as nc


class TrainingDataset(Dataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.
    """

    def __init__(self, root: str, lon_range: List[int], lat_range: List[int]):
        super().__init__()
        self.lon_range = lon_range
        self.lat_range = lat_range

        self.sample_num = 0
        self.files = []
        date_list = sorted(os.listdir(root))
        for date in date_list:
            file_list = sorted(os.listdir(os.path.join(root, date)))
            for file_ in file_list:
                self.files.append(os.path.join(root, date, file_))
                self.sample_num += 1

    def __getitem__(self, index):
        filename = self.files[index]
        tensor = self.load_nc(filename)
        return tensor
    
    def __len__(self,):
        return self.sample_num
    
    def load_nc(self, filename):
        nc_file = nc.Dataset(filename)
        tensor = torch.from_numpy(nc_file.variables['DBZ'][:])
        tensor = tensor[:, self.lat_range[0]: self.lat_range[1], self.lon_range[0]: self.lon_range[1]]
        nc_file.close()
        return tensor


class SampleDataset(TrainingDataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.
    """

    def __init__(self, root: str, sample_index: int, lon_range: List[int], lat_range: List[int]):
        super().__init__(root, lon_range, lat_range)
        self.sample_index = sample_index

    def __getitem__(self, index: int):
        filename = self.files[self.sample_index]
        tensor = self.load_nc(filename)
        return tensor

    def __len__(self):
        return 1


def load_data(root: str, batch_size: int, num_workers: int, train_ratio: float, valid_ratio: float, 
              lon_range: List[int], lat_range: List[int]):
    r"""Load training and test data.

    Args:
        root (str): Path to the dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of processes.
        train_ratio (float): Training ratio of the whole dataset.
        valid_ratio (float): Validation ratio of the whole dataset.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.
    Returns:
        DataLoader: Dataloader for training and test.
    """

    dataset = TrainingDataset(root, lon_range, lat_range)
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


def load_sample(root: str, sample_index: int, lon_range: List[int], lat_range: List[int]):
    r"""Load sample data.

    Args:
        root (str): Path to the dataset.
        sample_index (int): Index of the sample. Default is None, meaning the last sample is selected.
        lon_range (List[int]): Longitude range for images.
        lat_range (List[int]): Latitude range for images.

    Returns:
        DataLoader: Dataloader for sample.
    """

    sample_set = SampleDataset(root, sample_index, lon_range, lat_range)
    sample_loader = DataLoader(sample_set, batch_size=1)
    return sample_loader
