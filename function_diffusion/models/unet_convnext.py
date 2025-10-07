"""
Mixed adaptation from:

    Liu et al. 2022, A ConvNet for the 2020s.
    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Ronneberger et al., 2015. Convolutional Networks for Biomedical Image Segmentation.

If you use this implementation, please cite original work above.

JAX/Flax implementation with channels-last format.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Optional, Sequence
import functools


class LayerNorm(nn.Module):
    """LayerNorm for channels-last format.

    Input shape: (batch_size, *spatial_dims, channels)
    """
    epsilon: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x):
        # Normalize over the channel dimension (last axis)
        features = x.shape[-1]

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)

        normed = (x - mean) / jnp.sqrt(var + self.epsilon)

        if self.use_scale:
            scale = self.param('scale', nn.initializers.ones, (features,))
            normed = normed * scale

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (features,))
            normed = normed + bias

        return normed


class Upsample(nn.Module):
    """Upsample layer with transposed convolution."""
    out_dim: int
    n_spatial_dims: int = 2

    @nn.compact
    def __call__(self, x):
        x = LayerNorm()(x)

        # ConvTranspose with channels-last format
        kernel_size = (2,) * self.n_spatial_dims
        strides = (2,) * self.n_spatial_dims

        x = nn.ConvTranspose(
            features=self.out_dim,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=True
        )(x)

        return x


class Downsample(nn.Module):
    """Downsample layer with strided convolution."""
    out_dim: int
    n_spatial_dims: int = 2

    @nn.compact
    def __call__(self, x):
        x = LayerNorm()(x)

        # Conv with channels-last format
        kernel_size = (2,) * self.n_spatial_dims
        strides = (2,) * self.n_spatial_dims

        x = nn.Conv(
            features=self.out_dim,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=True
        )(x)

        return x


class Block(nn.Module):
    """ConvNeXt Block for channels-last format.

    Flow: DwConv -> LayerNorm -> Linear -> GELU -> Linear -> Scale -> Residual

    Args:
        dim: Number of channels
        n_spatial_dims: Number of spatial dimensions
        drop_path_rate: Stochastic depth rate
        layer_scale_init_value: Init value for layer scale
    """
    dim: int
    n_spatial_dims: int
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        residual = x

        # Depthwise convolution (groups = channels)
        kernel_size = (7,) * self.n_spatial_dims
        x = nn.Conv(
            features=self.dim,
            kernel_size=kernel_size,
            padding='SAME',
            feature_group_count=self.dim,  # Depthwise
            use_bias=True
        )(x)

        # LayerNorm
        x = LayerNorm()(x)

        # Pointwise convolutions (implemented as Dense since input is channels-last)
        x = nn.Dense(features=4 * self.dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.dim)(x)

        # Layer scale
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                'gamma',
                lambda rng, shape: self.layer_scale_init_value * jnp.ones(shape),
                (self.dim,)
            )
            x = gamma * x

        # Stochastic depth
        if self.drop_path_rate > 0.0 and not deterministic:
            rng = self.make_rng('dropout')
            keep_prob = 1.0 - self.drop_path_rate
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1],)
            mask = jax.random.bernoulli(rng, keep_prob, mask_shape)
            x = x * mask / keep_prob

        # Residual connection
        x = residual + x

        return x


class Stage(nn.Module):
    """ConvNeXt Stage.

    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels
        n_spatial_dims: Number of spatial dimensions
        depth: Number of blocks in the stage
        drop_path_rate: Stochastic depth rate
        layer_scale_init_value: Init value for layer scale
        mode: 'down', 'up', or 'neck'
        skip_project: Whether to project concatenated skip connections
    """
    in_dim: int
    out_dim: int
    n_spatial_dims: int
    depth: int = 1
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6
    mode: str = 'down'
    skip_project: bool = False

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Skip connection projection (for decoder when concatenating)
        if self.skip_project:
            x = nn.Conv(
                features=self.in_dim,
                kernel_size=(1,) * self.n_spatial_dims,
                use_bias=True
            )(x)

        # Blocks at current resolution
        for i in range(self.depth):
            x = Block(
                dim=self.in_dim,
                n_spatial_dims=self.n_spatial_dims,
                drop_path_rate=self.drop_path_rate,
                layer_scale_init_value=self.layer_scale_init_value
            )(x, deterministic=deterministic)

        # Resampling
        if self.mode == 'down':
            x = Downsample(
                out_dim=self.out_dim,
                n_spatial_dims=self.n_spatial_dims
            )(x)
        elif self.mode == 'up':
            x = Upsample(
                out_dim=self.out_dim,
                n_spatial_dims=self.n_spatial_dims
            )(x)
        # mode == 'neck': no resampling

        return x


class UNetConvNext(nn.Module):
    """UNet with ConvNeXt blocks.

    Expects input in channels-last format: (batch, *spatial_dims, channels)

    Args:
        out_dim: Number of output channels
        stages: Number of encoder/decoder stages
        blocks_per_stage: Number of ConvNeXt blocks per stage
        blocks_at_neck: Number of blocks at the bottleneck
        n_spatial_dims: Number of spatial dimensions (1, 2, or 3)
        init_features: Initial number of features
    """
    out_dim: int
    stages: int = 4
    blocks_per_stage: int = 1
    blocks_at_neck: int = 1
    n_spatial_dims: int = 2
    init_features: int = 32
    model_name: Optional[str] = None

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        features = self.init_features
        in_dim = x.shape[-1]  # Channels-last

        # Calculate feature dimensions for each stage
        encoder_dims = [features * 2 ** i for i in range(self.stages + 1)]
        decoder_dims = [features * 2 ** i for i in range(self.stages, -1, -1)]

        # Input projection
        x = nn.Conv(
            features=features,
            kernel_size=(3,) * self.n_spatial_dims,
            padding='SAME',
            use_bias=True
        )(x)

        # Encoder
        skips = []
        for i in range(self.stages):
            skips.append(x)
            x = Stage(
                in_dim=encoder_dims[i],
                out_dim=encoder_dims[i + 1],
                n_spatial_dims=self.n_spatial_dims,
                depth=self.blocks_per_stage,
                mode='down'
            )(x, deterministic=deterministic)

        # Neck (bottleneck)
        x = Stage(
            in_dim=encoder_dims[-1],
            out_dim=encoder_dims[-1],
            n_spatial_dims=self.n_spatial_dims,
            depth=self.blocks_at_neck,
            mode='neck'
        )(x, deterministic=deterministic)

        # Decoder
        for j in range(self.stages):
            if j > 0:
                # Concatenate skip connection along channel dimension
                x = jnp.concatenate([x, skips[-(j)]], axis=-1)

            x = Stage(
                in_dim=decoder_dims[j],
                out_dim=decoder_dims[j + 1],
                n_spatial_dims=self.n_spatial_dims,
                depth=self.blocks_per_stage,
                mode='up',
                skip_project=(j != 0)
            )(x, deterministic=deterministic)

        # Output projection
        x = nn.Conv(
            features=self.out_dim,
            kernel_size=(3,) * self.n_spatial_dims,
            padding='SAME',
            use_bias=True
        )(x)

        return x
