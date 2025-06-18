import numpy as np

from einops import rearrange, repeat

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from function_diffusion.utils.data_utils import BaseDataset


def create_dataset(config):
    kf_data = np.load(config.dataset.data_path, allow_pickle=True).item()

    u = kf_data['u']
    v = kf_data['v']
    w = kf_data['w']

    u = rearrange(u, 'b t h w -> (b t) h w')
    v = rearrange(v, 'b t h w -> (b t) h w')
    w = rearrange(w, 'b t h w -> (b t) h w')

    uvw = np.stack([u, v], axis=-1)

    dataset = BaseDataset(uvw, config.dataset.downsample_factor)

    # Shuffle the dataset
    torch.manual_seed(88)
    np.random.seed(88)

    # Generate indices
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    train_size = int(0.9 * len(dataset))

    # Create train and test datasets
    train_dataset = Subset(dataset, indices[:train_size])
    test_dataset = Subset(dataset, indices[train_size:])

    return train_dataset, test_dataset




