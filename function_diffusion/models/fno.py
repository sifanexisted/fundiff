import jax.numpy as jnp
from jax import lax, jit, grad, random, tree_map

import flax.linen as nn

from typing import Optional, Callable, Dict


# Reference: https://github.com/astanziola/fourier-neural-operator-flax
def normal(stddev=1e-2, dtype=jnp.float32) -> Callable:
    def init(key, shape, dtype=dtype):
        keys = random.split(key)
        return random.normal(keys[0], shape) * stddev

    return init


def complex_mut2d(x, kernel):
    """
    x: (b, modes1, modes2, c)
    kernel: (2, c, out_dim, modes1, modes2)
    out: (b, modes1, modes2, out_dim)
    """
    kernel_r = kernel[0]
    kernel_i = kernel[1]

    out = jnp.einsum("bijc,coij->bijo", x, kernel_r + 1j * kernel_i)
    return out


class SpectralConv2d(nn.Module):
    out_dim: int = 32
    modes1: int = 12
    modes2: int = 12

    @nn.compact
    def __call__(self, x):
        # x.shape: (b, h, w, c)

        # Initialize parameters
        in_dim = x.shape[-1]
        scale = 1 / (in_dim * self.out_dim)
        in_dim = x.shape[-1]
        h = x.shape[1]
        w = x.shape[2]

        # Checking that the modes are not more than the input size
        assert self.modes1 <= h // 2 + 1
        assert self.modes2 <= w // 2 + 1
        assert h % 2 == 0  # Only tested for even-sized inputs
        assert w % 2 == 0  # Only tested for even-sized inputs

        kernel_1 = self.param(
            "kernel_1_r",
            normal(scale, jnp.float32),
            (2, in_dim, self.out_dim, self.modes1, self.modes2),
            jnp.float32,
        )

        kernel_2 = self.param(
            "kernel_2_r",
            normal(scale, jnp.float32),
            (2, in_dim, self.out_dim, self.modes1, self.modes2),
            jnp.float32,
        )

        # Perform fft of the input
        x_ft = jnp.fft.rfftn(x, axes=(1, 2))

        # Multiply the center of the spectrum by the kernel
        out_ft = jnp.zeros_like(x_ft)

        s1 = complex_mut2d(x_ft[:, : self.modes1, : self.modes2, :], kernel_1)
        s2 = complex_mut2d(x_ft[:, -self.modes1 :, : self.modes2, :], kernel_2)

        out_ft = out_ft.at[:, : self.modes1, : self.modes2, :].set(s1)
        out_ft = out_ft.at[:, -self.modes1 :, : self.modes2, :].set(s2)

        # Go back to the spatial domain
        y = jnp.fft.irfftn(out_ft, axes=(1, 2))

        return y


class FourierStage(nn.Module):
    emb_dim: int = 32
    modes1: int = 12
    modes2: int = 12
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        x_fourier = SpectralConv2d(
            out_dim=self.emb_dim, modes1=self.modes1, modes2=self.modes2
        )(x)
        x_local = nn.Conv(
            self.emb_dim,
            (1, 1),
        )(x)
        return self.activation(x_fourier + x_local)


class FNO2d(nn.Module):
    modes1: int = 12
    modes2: int = 12
    emb_dim: int = 32
    out_dim: int = 1
    depth: int = 4
    activation: Callable = nn.gelu
    padding: int = 0  # Padding for non-periodic inputs
    model_name: Optional[str] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Lift the input to a higher dimension
        x = nn.Dense(self.emb_dim)(x)

        # Pad input
        if self.padding > 0:
            x = jnp.pad(
                x,
                ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
                mode="constant",
            )

        for _ in range(self.depth):
            x = FourierStage(
                emb_dim=self.emb_dim,
                modes1=self.modes1,
                modes2=self.modes2,
                activation=self.activation,
            )(x)

        # Unpad
        if self.padding > 0:
            x = x[:, : -self.padding, : -self.padding, :]

        # Project to the output dimension
        x = nn.Dense(self.emb_dim)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_dim)(x)

        return x
