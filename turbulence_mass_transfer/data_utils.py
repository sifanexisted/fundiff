import numpy as np
from einops import rearrange, repeat

from jax import random, jit

from function_diffusion.utils.data_utils import BaseDataset


def get_dataset(config):
    data = np.load(config.dataset.data_path, allow_pickle=True).item()

    u = data['x_velocity']
    v = data['y_velocity']
    p = data['pressure']
    # udm = data['udm']
    sdf = data['sdf']

    outputs = np.stack([u, v, p, sdf], axis=-1)  # (b, h, w, c)

    # Shuffle dataset
    outputs = random.permutation(random.PRNGKey(0), outputs, axis=0)
    outputs = np.array(outputs, dtype=np.float32)

    train_outputs = outputs[:config.dataset.num_train_samples]
    test_outputs = outputs[config.dataset.num_train_samples:]

    # Normalize the data
    mean = train_outputs.mean(axis=(0, 1, 2))
    std = train_outputs.std(axis=(0, 1, 2))

    train_outputs = (train_outputs - mean) / std
    test_outputs = (test_outputs - mean) / std

    return train_outputs, test_outputs, mean, std


def create_dataset(config):
    train_outputs, test_outputs, mean, std = get_dataset(config)

    train_outputs = rearrange(train_outputs, 'b h w c -> (b c) h w')
    test_outputs = rearrange(test_outputs, 'b h w c -> (b c) h w')

    train_outputs = train_outputs[..., None]
    test_outputs = test_outputs[..., None]

    train_dataset = BaseDataset(train_outputs, config.dataset.downsample_factor)
    test_dataset = BaseDataset(test_outputs, config.dataset.downsample_factor)

    return train_dataset, test_dataset


def create_diffusion_dataset(config):
    train_outputs, test_outputs, mean, std = get_dataset(config)

    train_dataset = BaseDataset(train_outputs, config.dataset.downsample_factor)
    test_dataset = BaseDataset(test_outputs, config.dataset.downsample_factor)

    return train_dataset, test_dataset
