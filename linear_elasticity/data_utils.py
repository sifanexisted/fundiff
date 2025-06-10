import os

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset


class BaseDataset(Dataset):
    # This dataset class is used for homogenization
    def __init__(
        self,
        input_files,
        input_keys,
        downsample_factor=2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.num_files = len(input_files)
        self.inputs = []

        for input_file, input_key in zip(input_files, input_keys):

            inputs = h5py.File(input_file, "r")[input_key]

            print(inputs.shape)
            inputs = np.array(inputs)
            inputs = np.transpose(inputs, (0, 2, 3, 1))

            inputs = inputs[:, :: self.downsample_factor, :: self.downsample_factor, [0, 4, 8]
                     ]

            self.inputs.append(inputs)

        self.inputs = np.vstack(self.inputs)

    def __len__(self):
        # Assuming all datasets have the same length, use the length of the first one
        return self.inputs.shape[0]

    def __getitem__(self, index):
        # Choose an input file randomly each time
        batch_inputs = np.array(self.inputs[index])
        return batch_inputs


def create_dataset(config):
    data_path = config.dataset.data_path

    train_suffixes = ['part1', 'part2', 'part3']

    input_keys = [f"output_data_{suffix}" for suffix in train_suffixes]
    input_files = [os.path.join(data_path, f"{key}.mat") for key in input_keys]

    dataset = BaseDataset(input_files, input_keys, config.dataset.downsample_factor)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate indices
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    train_size = int(0.9 * len(dataset))

    # Create train and test datasets
    train_dataset = Subset(dataset, indices[:train_size])
    test_dataset = Subset(dataset, indices[train_size:])

    return train_dataset, test_dataset





