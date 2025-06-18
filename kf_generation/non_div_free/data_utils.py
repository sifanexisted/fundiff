import os
from functools import partial

import h5py
import numpy as np
from einops import rearrange, repeat

import jax
import jax.numpy as jnp

from jax import random, jit

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from function_diffusion.utils.data_utils import BaseDataset


class BatchParser:
    def __init__(self, config, h, w):
        self.config = config
        self.num_query_points = config.training.num_queries

        x_star = jnp.linspace(0, 1, h)
        y_star = jnp.linspace(0, 1, w)
        x_star, y_star = jnp.meshgrid(x_star, y_star, indexing="ij")

        self.coords = jnp.hstack([x_star.flatten()[:, None], y_star.flatten()[:, None]])

    @partial(jit, static_argnums=(0,))
    def random_query(self, batch, rng_key=None):
        batch_inputs = batch
        batch_outputs = rearrange(batch, "b h w c -> b (h w) c")

        query_index = random.choice(
            rng_key, batch_outputs.shape[1], (self.num_query_points,), replace=False
        )
        batch_coords = self.coords[query_index]
        batch_outputs = batch_outputs[:, query_index]

        # Repeat the coords  across devices
        batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

        return batch_coords, batch_inputs, batch_outputs

    @partial(jit, static_argnums=(0,))
    def query_all(self, batch):
        batch_inputs = batch

        batch_outputs = rearrange(batch, "b h w c -> b (h w) c")
        batch_coords = self.coords

        # Repeat the coords  across devices
        batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

        return batch_coords, batch_inputs, batch_outputs


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


