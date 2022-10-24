from typing import List, Tuple, Union
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import utils.reader as reader


class TrainingDataset(Dataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset. 
        elevation_id (int or List[int]): ID of elevations.
        azimuth_range (List[int]): Range of azimuth. 
        radial_range (List[int]): Range of radial distance. 
        
    """
    def __init__(self, root: str, elevation_id: Union[int, List[int]], azimuth_range: List[int], radial_range: List[int]):
        super().__init__()
        self.elevation_id = elevation_id
        self.azimuth_range = azimuth_range
        self.radial_range = radial_range
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
        elev, ref = reader.read_radar_bin(filename)
        elev, ref = elev[self.elevation_id], ref[self.elevation_id]
        ref = ref[:, self.azimuth_range[0]: self.azimuth_range[1], self.radial_range[0]: self.radial_range[1]]
        elev, ref = torch.from_numpy(elev).type(torch.FloatTensor), torch.from_numpy(ref).type(torch.FloatTensor)
        return elev, ref
    
    def __len__(self):
        return self.sample_num


class SampleDataset(TrainingDataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset.
        sample_index (int): Index of the sample. Default is None, meaning the last sample is selected.
        elevation_id (int or List[int]): ID of elevations.
        azimuth_range (List[int]): Range of azimuth. 
        radial_range (List[int]): Range of radial distance. 
    """

    def __init__(self, root: str, sample_index: int, elevation_id: Union[int, List[int]], azimuth_range: List[int], radial_range: List[int]):
        super().__init__(root, elevation_id, azimuth_range, radial_range)
        self.sample_index = sample_index
        self.elevation_id = elevation_id
        self.azimuth_range = azimuth_range
        self.radial_range = radial_range
        
    def __getitem__(self, index: int):
        filename = self.files[self.sample_index]
        elev, ref = reader.read_radar_bin(filename)
        elev, ref = elev[self.elevation_id], ref[self.elevation_id]
        ref = ref[:, self.azimuth_range[0]: self.azimuth_range[1], self.radial_range[0]: self.radial_range[1]]
        elev, ref = torch.from_numpy(elev).type(torch.FloatTensor), torch.from_numpy(ref).type(torch.FloatTensor)
        return elev, ref

    def __len__(self):
        return 1


def load_data(root: str, batch_size: int, num_workers: int, train_ratio: float, valid_ratio: float, 
              elevation_id: Union[int, List[int]] = 0, azimuth_range: List[int] = [0, 360], radial_range: List[int] = [0, 460]) \
              -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Load training and test data.

    Args:
        root (str): Path to the dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of processes.
        train_ratio (float): Training ratio of the whole dataset.
        valid_ratio (float): Validation ratio of the whole dataset.
        elevation_id (int or List[int]): ID of elevations. Default: 0
        azimuth_range (List[int]): Range of azimuth. Default: [0, 360]
        radial_range (List[int]): Range of radial distance. Default: [0, 460]
    
    Returns:
        DataLoader: Dataloader for training and test.
    """

    dataset = TrainingDataset(root, elevation_id, azimuth_range, radial_range)
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

    print('Total dataset length: ', dataset_size)
    print('\nTrain Loader')
    print('----Batch Num:', len(train_loader))
    print('\nVal Loader')
    print('----Batch Num:', len(val_loader))
    print('\nTest Loader')
    print('----Batch Num:', len(test_loader))

    return train_loader, val_loader, test_loader


def load_sample(root: str, sample_index: int = -1, elevation_id: Union[int, List[int]] = 0, azimuth_range: List[int] = [0, 360], 
                radial_range: List[int] = [0, 460]) -> DataLoader:
    r"""Load sample data.

    Args:
        root (str): Path to the dataset.
        sample_index (int): Index of the sample. Default is -1, meaning the last sample is selected.
        elevation_id (int or List[int]): ID of elevations. Default: 0
        azimuth_range (List[int]): Range of azimuth. Default: [0, 360]
        radial_range (List[int]): Range of radial distance. Default: [0, 460]
    
    Returns:
        DataLoader: Dataloader for sample.
    """

    sample_set = SampleDataset(root, elevation_id, sample_index, azimuth_range, radial_range)
    sample_loader = DataLoader(sample_set, batch_size=1)
    return sample_loader
