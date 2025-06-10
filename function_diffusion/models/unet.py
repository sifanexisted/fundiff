import jax.numpy as jnp

import flax.linen as nn

from typing import Optional, Callable, Dict


class ConvBlock(nn.Module):
    features: int
    num_groups: int = 1
    norm: bool = True
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        conv1 = nn.Conv(features=self.features, kernel_size=(3, 3), padding="VALID")
        conv2 = nn.Conv(features=self.features, kernel_size=(3, 3), padding="VALID")

        if self.norm:
            norm1 = nn.GroupNorm(self.num_groups)
            norm2 = nn.GroupNorm(self.num_groups)
        else:
            norm1 = norm2 = lambda x: x

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
        x = self.activation(norm1(conv1(x)))
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
        x = self.activation(norm2(conv2(x)))

        return x


class Down(nn.Module):
    features: int
    num_groups: int = 1
    norm: bool = True
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        x = ConvBlock(self.features, self.num_groups, self.norm, self.activation)(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        return x


class Up(nn.Module):
    features: int
    num_groups: int = 1
    norm: bool = True
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x1, x2):
        up = nn.ConvTranspose(
            features=x1.shape[-1] // 2, kernel_size=(2, 2), strides=(2, 2)
        )
        conv = ConvBlock(self.features, self.num_groups, self.norm, self.activation)

        x = up(x1)
        x = jnp.concatenate([x2, x], -1)
        x = conv(x)
        return x


class UNetEncoder(nn.Module):
    emb_dim: int
    num_groups: int = 1
    norm: bool = True
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        # x: (256, 256)
        in_dim = self.emb_dim
        h = ConvBlock(in_dim, activation=self.activation)(x)  # 256x256x64

        in_dim *= 2
        x1 = Down(in_dim, self.num_groups, self.norm, self.activation)(h)  # 128x128x128

        in_dim *= 2
        x2 = Down(in_dim, self.num_groups, self.norm, self.activation)(x1)  # 64x64x256

        in_dim *= 2
        x3 = Down(in_dim, self.num_groups, self.norm, self.activation)(x2)  # 32x32x512

        in_dim *= 2
        x4 = Down(in_dim, self.num_groups, self.norm, self.activation)(x3)  # 16x16x1024

        return [h, x1, x2, x3, x4]


class UNetDecoder(nn.Module):
    out_dim: int
    num_groups: int = 1
    norm: bool = True
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        h, x1, x2, x3, x4 = x

        in_dim = x4.shape[-1]
        in_dim //= 2
        x = Up(in_dim, self.num_groups, self.norm, self.activation)(x4, x3)
        in_dim //= 2
        x = Up(in_dim, self.num_groups, self.norm, self.activation)(x, x2)
        in_dim //= 2
        x = Up(in_dim, self.num_groups, self.norm, self.activation)(x, x1)
        in_dim //= 2
        x = Up(in_dim, self.num_groups, self.norm, self.activation)(x, h)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
        x = nn.Conv(self.out_dim, (3, 3), padding="VALID")(x)

        return x


class UNet(nn.Module):
    emb_dim: int
    out_dim: int
    activation: Callable = nn.gelu
    model_name: Optional[str] = None

    @nn.compact
    def __call__(self, x):
        x = UNetEncoder(emb_dim=self.emb_dim, activation=self.activation)(x)
        x = UNetDecoder(out_dim=self.out_dim, activation=self.activation)(x)
        return x
