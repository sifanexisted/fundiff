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


class BaseDataset(Dataset):
    # This dataset class is used for homogenization
    def __init__(
            self,
            x,
            downsample_factor=1,
            num_samples=None
    ):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.num_samples = num_samples  # Number of samples to use for training, if None use all samples by default

        self.x = x[:, ::downsample_factor, ::downsample_factor]

    def __len__(self):
        # Assuming all datasets have the same length, use the length of the first one
        if self.num_samples is not None:
            return self.num_samples
        else:
            return len(self.x)

    def __getitem__(self, index):
        batch = self.x[index]

        return batch


# class BatchParser:
#     def __init__(self, config, h, w):
#         self.config = config
#         self.num_query_points = config.training.num_queries
#
#         x_star = jnp.linspace(0, 1, h)
#         y_star = jnp.linspace(0, 1, w)
#         x_star, y_star = jnp.meshgrid(x_star, y_star, indexing="ij")
#
#         self.coords = jnp.hstack([x_star.flatten()[:, None], y_star.flatten()[:, None]])
#
#     @partial(jit, static_argnums=(0,))
#     def random_query(self, batch, rng_key=None):
#         batch_inputs = batch
#         batch_outputs = rearrange(batch, "b h w c -> b (h w) c")
#
#         query_index = random.choice(
#             rng_key, batch_outputs.shape[1], (self.num_query_points,), replace=False
#         )
#         batch_coords = self.coords[query_index]
#         batch_outputs = batch_outputs[:, query_index]
#
#         # Repeat the coords  across devices
#         batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())
#
#         return batch_coords, batch_inputs, batch_outputs
#
#     @partial(jit, static_argnums=(0,))
#     def query_all(self, batch):
#         batch_inputs = batch
#
#         batch_outputs = rearrange(batch, "b h w c -> b (h w) c")
#         batch_coords = self.coords
#
#         # Repeat the coords  across devices
#         batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())
#
#         return batch_coords, batch_inputs, batch_outputs


class BatchParser:
    def __init__(self, config, h, w):
        self.config = config
        self.num_query_points = config.training.num_queries

        x_star = jnp.linspace(0, 1, h)
        y_star = jnp.linspace(0, 1, w)
        x_star, y_star = jnp.meshgrid(x_star, y_star, indexing="ij")

        self.coords = jnp.hstack([x_star.flatten()[:, None], y_star.flatten()[:, None]])

    # @partial(jit, static_argnums=(0,))
    def random_query(self, batch, downsample=1, rng_key=None):
        batch_inputs = batch
        batch_outputs = rearrange(batch, "b h w c -> b (h w) c")

        query_index = random.choice(
            rng_key, batch_outputs.shape[1], (self.num_query_points,), replace=False
        )
        batch_coords = self.coords[query_index]
        batch_outputs = batch_outputs[:, query_index]

        # Repeat the coords  across devices
        batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

        # Downsample the inputs
        batch_inputs = batch_inputs[:, ::downsample, ::downsample]

        return batch_coords, batch_inputs, batch_outputs

    @partial(jit, static_argnums=(0,))
    def query_all(self, batch):
        batch_inputs = batch

        batch_outputs = rearrange(batch, "b h w c -> b (h w) c")
        batch_coords = self.coords

        # Repeat the coords  across devices
        batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

        return batch_coords, batch_inputs, batch_outputs


def create_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=True):
    num_devices = jax.device_count()

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size * num_devices,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_loader