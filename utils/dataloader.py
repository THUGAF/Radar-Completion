from typing import List, Tuple, Union
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class TrainingDataset(Dataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset. 
        elevation_id (int or List[int]): ID of elevations.
        azimuthal_range (List[int]): Range of azimuth. 
        radial_range (List[int]): Range of radial distance. 
        augment_ratio (int): Ratio for data augment. Default: 1.
    """

    def __init__(self, root: str, elevation_id: Union[int, List[int]], azimuthal_range: List[int], 
                 radial_range: List[int], augment_ratio: int = 1):
        super().__init__()
        self.elevation_id = elevation_id
        self.azimuthal_range = azimuthal_range
        self.radial_range = radial_range
        self.files = sorted(glob.glob(os.path.join(root, '*/*.pt')))
        self.files = self.files * augment_ratio
        if augment_ratio > 1:
            self.files = np.reshape(self.files, (augment_ratio, -1))
            self.files = self.files.transpose().ravel().tolist()

    def __getitem__(self, index: int):
        filename = self.files[index]
        t, elev, ref = torch.load(filename)
        elev, ref = elev[self.elevation_id], ref[self.elevation_id]
        ref = ref[:, self.azimuthal_range[0]: self.azimuthal_range[1], 
                  self.radial_range[0]: self.radial_range[1]]
        return t, elev, ref
    
    def __len__(self):
        return len(self.files)


class CaseDataset(TrainingDataset): 
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset.
        case_index (int): Index of the case. Default is None, meaning the last case is selected.
        elevation_id (int or List[int]): ID of elevations.
        azimuth_range (List[int]): Range of azimuth. 
        radial_range (List[int]): Range of radial distance. 
    """

    def __init__(self, root: str, case_index: int, elevation_id: Union[int, List[int]], 
                 azimuthal_range: List[int], radial_range: List[int]):
        super().__init__(root, elevation_id, azimuthal_range, radial_range)
        self.case_index = case_index
        self.elevation_id = elevation_id
        self.azimuthal_range = azimuthal_range
        self.radial_range = radial_range
        
    def __getitem__(self, index: int):
        filename = self.files[self.case_index[index]]
        t, elev, ref = torch.load(filename)
        elev, ref = elev[self.elevation_id], ref[self.elevation_id]
        ref = ref[:, self.azimuthal_range[0]: self.azimuthal_range[1], 
                  self.radial_range[0]: self.radial_range[1]]
        return t, elev, ref

    def __len__(self):
        return 1


def load_data(root: str, batch_size: int, num_workers: int, train_ratio: float, valid_ratio: float, 
              elevation_id: Union[int, List[int]] = 0, azimuthal_range: List[int] = [0, 360], 
              radial_range: List[int] = [0, 460], augment_ratio: int = 1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load training and test data.

    Args:
        root (str): Path to the dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of processes.
        train_ratio (float): Training ratio of the whole dataset.
        valid_ratio (float): Validation ratio of the whole dataset.
        elevation_id (int or List[int]): ID of elevations. Default: 0
        azimuthal_range (List[int]): Range of azimuth. Default: [0, 360]
        radial_range (List[int]): Range of radial distance. Default: [0, 460]
        augment_ratio (int): Ratio for data augment. Default: 1.

    Returns:
        DataLoader: Dataloader for training and test.
    """

    dataset = TrainingDataset(root, elevation_id, azimuthal_range, radial_range, augment_ratio)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_node = round(train_ratio * dataset_size)
    val_node = round(valid_ratio * dataset_size)
    train_val_indices = indices[:train_node + val_node]
    test_indices = indices[train_node + val_node:]
    
    train_val_set = Subset(dataset, train_val_indices)
    train_set, val_set = random_split(train_val_set, lengths=[train_node, len(train_val_set) - train_node])
    test_set = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, pin_memory=True)

    print('\nDataset Length:', len(dataset))
    print('Train Set Length:', len(train_set))
    print('Val Set Length:', len(val_set))
    print('Test Set Length:', len(test_set))
    print('Train Loader Batch Num:', len(train_loader))
    print('Val Loader Batch Num:', len(val_loader))
    print('Test Loader Batch Num:', len(test_loader))

    return train_loader, val_loader, test_loader


def load_case(root: str, case_indices: List[int], elevation_id: Union[int, List[int]] = 0, 
              azimuthal_range: List[int] = [0, 360], radial_range: List[int] = [0, 460]) -> DataLoader:
    """Load case data.

    Args:
        root (str): Path to the dataset.
        case_indices (int): Indices of the cases.
        elevation_id (int or List[int]): ID of elevations. Default: 0
        azimuthal_range (List[int]): Range of azimuth. Default: [0, 360]
        radial_range (List[int]): Range of radial distance. Default: [0, 460]
    
    Returns:
        DataLoader: Dataloader for case.
    """

    case_set = CaseDataset(root, case_indices, elevation_id, azimuthal_range, radial_range)
    case_loader = DataLoader(case_set, batch_size=1)
    return case_loader
