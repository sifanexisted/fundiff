# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Converted to JAX/Flax

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Optional, Tuple
import math
from einops import rearrange

ACTIVATION = {
    'gelu': nn.gelu,
    'tanh': jnp.tanh,
    'sigmoid': nn.sigmoid,
    'relu': nn.relu,
    'leaky_relu': lambda x: nn.leaky_relu(x, negative_slope=0.1),
    'softplus': nn.softplus,
    'elu': nn.elu,
    'silu': nn.silu
}


class AFNO2D(nn.Module):
    """
    AFNO (Adaptive Fourier Neural Operator) 2D layer.

    Attributes:
        width: channel dimension size
        num_blocks: how many blocks to use in the block diagonal weight matrices

        modes: number of Fourier modes to keep
        hidden_size_factor: factor to scale hidden size
        act: activation function name
    """
    width: int = 32
    num_blocks: int = 8
    modes: int = 32
    hidden_size_factor: int = 1
    act: str = 'gelu'

    def setup(self):
        assert self.width % self.num_blocks == 0, \
            f"width {self.width} should be divisible by num_blocks {self.num_blocks}"

        self.block_size = self.width // self.num_blocks
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)
        self.activation = ACTIVATION[self.act]

        # Initialize weights
        w1_init = nn.initializers.uniform(scale=self.scale)
        b1_init = nn.initializers.uniform(scale=self.scale)
        w2_init = nn.initializers.uniform(scale=self.scale)
        b2_init = nn.initializers.uniform(scale=self.scale)

        self.w1 = self.param('w1', w1_init,
                             (2, self.num_blocks, self.block_size,
                              self.block_size * self.hidden_size_factor))
        self.b1 = self.param('b1', b1_init,
                             (2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = self.param('w2', w2_init,
                             (2, self.num_blocks, self.block_size * self.hidden_size_factor,
                              self.block_size))
        self.b2 = self.param('b2', b2_init,
                             (2, self.num_blocks, self.block_size))

    def __call__(self, x, spatial_size=None):
        """
        Args:
            x: Input tensor of shape (B, H, W, C) - note channel last!
        Returns:
            Output tensor of shape (B, H, W, C)
        """
        B, H, W, C = x.shape
        x_orig = x

        # FFT - JAX uses different convention
        x = jnp.fft.rfft2(x, axes=(1, 2), norm="ortho")

        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        kept_modes = self.modes

        # Initialize output tensors
        o1_real = jnp.zeros((B, x.shape[1], x.shape[2], self.num_blocks,
                             self.block_size * self.hidden_size_factor))
        o1_imag = jnp.zeros((B, x.shape[1], x.shape[2], self.num_blocks,
                             self.block_size * self.hidden_size_factor))
        o2_real = jnp.zeros(x.shape)
        o2_imag = jnp.zeros(x.shape)

        # First layer
        x_real_kept = x[:, :kept_modes, :kept_modes].real
        x_imag_kept = x[:, :kept_modes, :kept_modes].imag

        o1_real_kept = self.activation(
            jnp.einsum('...bi,bio->...bo', x_real_kept, self.w1[0]) -
            jnp.einsum('...bi,bio->...bo', x_imag_kept, self.w1[1]) +
            self.b1[0]
        )
        o1_real = o1_real.at[:, :kept_modes, :kept_modes].set(o1_real_kept)

        o1_imag_kept = self.activation(
            jnp.einsum('...bi,bio->...bo', x_imag_kept, self.w1[0]) +
            jnp.einsum('...bi,bio->...bo', x_real_kept, self.w1[1]) +
            self.b1[1]
        )
        o1_imag = o1_imag.at[:, :kept_modes, :kept_modes].set(o1_imag_kept)

        # Second layer
        o2_real_kept = (
                jnp.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[0]) -
                jnp.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[1]) +
                self.b2[0]
        )
        o2_real = o2_real.at[:, :kept_modes, :kept_modes].set(o2_real_kept)

        o2_imag_kept = (
                jnp.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[0]) +
                jnp.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[1]) +
                self.b2[1]
        )
        o2_imag = o2_imag.at[:, :kept_modes, :kept_modes].set(o2_imag_kept)

        # Combine real and imaginary parts
        x = o2_real + 1j * o2_imag
        x = x.reshape(B, x.shape[1], x.shape[2], C)

        # Inverse FFT
        x = jnp.fft.irfft2(x, s=(H, W), axes=(1, 2), norm="ortho")

        # Residual connection
        x = x + x_orig

        return x


class Mlp(nn.Module):
    """MLP module with configurable hidden features."""
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act: str = 'gelu'
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        activation = ACTIVATION[self.act]

        x = nn.Dense(hidden_features)(x)
        x = activation(x)
        x = nn.Dense(out_features)(x)
        return x


class Block(nn.Module):
    """Transformer-style block with AFNO mixer and MLP."""
    mixing_type: str = 'afno'
    double_skip: bool = True
    width: int = 32
    n_blocks: int = 4
    mlp_ratio: float = 1.0
    modes: int = 32
    drop: float = 0.0
    drop_path: float = 0.0
    act: str = 'gelu'
    h: int = 14
    w: int = 8

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Args:
            x: Input of shape (B, H, W, C)
        """
        residual = x

        # First norm + mixer
        x = nn.GroupNorm(num_groups=8)(x)

        if self.mixing_type == "afno":
            x = AFNO2D(width=self.width, num_blocks=self.n_blocks,
                       modes=self.modes,
                       hidden_size_factor=1, act=self.act)(x)

        if self.double_skip:
            x = x + residual
            residual = x

        # Second norm + MLP
        x = nn.GroupNorm(num_groups=8)(x)

        mlp_hidden_dim = int(self.width * self.mlp_ratio)
        activation = ACTIVATION[self.act]

        # MLP as 1x1 convolutions - in Flax, we use channel-last
        x = nn.Conv(features=mlp_hidden_dim, kernel_size=(1, 1),
                    strides=(1, 1))(x)
        x = activation(x)
        x = nn.Conv(features=self.width, kernel_size=(1, 1),
                    strides=(1, 1))(x)

        x = x + residual

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    img_size: Tuple[int, int]
    patch_size: int = 16
    embed_dim: int = 768
    out_dim: int = 128
    act: str = 'gelu'

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input of shape (B, H, W, C)
        """
        B, H, W, C = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        activation = ACTIVATION[self.act]

        # Sequential convolutions
        x = nn.Conv(features=self.embed_dim,
                    kernel_size=(self.patch_size, self.patch_size),
                    strides=(self.patch_size, self.patch_size))(x)
        x = activation(x)
        x = nn.Conv(features=self.out_dim, kernel_size=(1, 1), strides=(1, 1))(x)

        return x


class DPOT(nn.Module):
    """
    DPOT (Deep Pseudo-differential Operator Transform) model.

    Attributes:
        in_T: Number of input timesteps
        img_size: Spatial resolution as (H, W)
        n_channel: Number of input channels
        patch_size: Size of patches
        mixing_type: Type of mixing layer
        out_timesteps: Number of output timesteps
        n_blocks: Number of blocks in AFNO
        embed_dim: Embedding dimension
        out_layer_dim: Output layer dimension
        depth: Number of transformer blocks
        modes: Number of Fourier modes
        mlp_ratio: MLP expansion ratio
        n_cls: Number of classification classes
        act: Activation function
        time_agg: Time aggregation type
    """
    img_size: Tuple[int, int]
    out_dim: int = 1
    patch_size: int = 16
    mixing_type: str = 'afno'
    n_blocks: int = 4
    embed_dim: int = 768
    out_layer_dim: int = 32
    depth: int = 12
    modes: int = 32
    mlp_ratio: float = 1.0
    act: str = 'gelu'
    model_name: Optional[str] = None

    def setup(self):
        self.activation = ACTIVATION[self.act]

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            out_dim=self.embed_dim,
            act=self.act
        )

        # Calculate output size after patch embedding
        out_h = self.img_size[0] // self.patch_size
        out_w = self.img_size[1] // self.patch_size
        self.latent_size = (out_h, out_w)

        # Position embedding
        pos_init = nn.initializers.truncated_normal(stddev=0.02)
        self.pos_embed = self.param('pos_embed', pos_init,
                                    (1, out_h, out_w, self.embed_dim))

    @nn.compact
    def __call__(self, x, train: bool = True):
        B, H, W, C = x.shape

        # Add grid coordinates
        grid = self.get_grid_2d(x)
        x = jnp.concatenate([x, grid], axis=-1)  # B, H, W,  C+2

        x = self.patch_embed(x)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for i in range(self.depth):
            x = Block(
                mixing_type=self.mixing_type,
                modes=self.modes,
                width=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                n_blocks=self.n_blocks,
                double_skip=False,
                h=self.img_size[0],
                w=self.img_size[1],
                act=self.act
            )(x, train=train)

        x = nn.ConvTranspose(
            features=self.out_layer_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size)
        )(x)
        x = self.activation(x)
        x = nn.Conv(features=self.out_layer_dim, kernel_size=(1, 1))(x)
        x = self.activation(x)
        x = nn.Conv(features=self.out_dim,
                    kernel_size=(1, 1))(x)

        return x

    def get_grid_2d(self, x):
        """Create 2D coordinate grid."""
        B, H, W, C = x.shape

        gridx = jnp.linspace(0, 1, H).reshape(1, H, 1, 1)
        gridx = jnp.broadcast_to(gridx, (B, H, W, 1))

        gridy = jnp.linspace(0, 1, W).reshape(1, 1, W, 1)
        gridy = jnp.broadcast_to(gridy, (B, H, W, 1))

        grid = jnp.concatenate([gridx, gridy], axis=-1)
        return grid




