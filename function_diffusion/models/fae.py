from einops import rearrange, repeat

import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform, normal, xavier_uniform

import flax.linen as nn
from typing import Optional, Callable, Dict, Union, Tuple


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        )
    return jnp.expand_dims(pos_embed, 0)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size[0], dtype=jnp.float32)
    grid_w = jnp.arange(grid_size[1], dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return jnp.expand_dims(pos_embed, 0)


class PatchEmbed(nn.Module):
    patch_size: tuple = (16, 16)
    emb_dim: int = 768
    use_norm: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        b, h, w, c = inputs.shape
        x = nn.Conv(
            self.emb_dim,
            (self.patch_size[0], self.patch_size[1]),
            (self.patch_size[0], self.patch_size[1]),
            kernel_init=self.kernel_init,
            name="proj",
        )(inputs)
        x = jnp.reshape(x, (b, -1, self.emb_dim))
        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=1e-5)(x)
        return x


class MlpBlock(nn.Module):
    dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.dim, kernel_init=self.kernel_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, kernel_init=self.kernel_init)(x)
        return x


class SelfAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(x, x)
        x = x + inputs

        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class CrossAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, q_inputs, kv_inputs):
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(q, kv)
        x = x + q_inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class PerceiverBlock(nn.Module):
    emb_dim: int
    depth: int
    num_heads: int = 8
    num_latents: int = 64
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):  # (B, L,  D) --> (B, L', D)
        latents = self.param('latents',
                             normal(),
                             (self.num_latents, self.emb_dim)  # (L', D)
                             )

        latents = repeat(latents, 'l d -> b l d', b=x.shape[0])  # (B, L', D)
        # Transformer
        for _ in range(self.depth):
            latents = CrossAttnBlock(self.num_heads,
                                     self.emb_dim,
                                     self.mlp_ratio,
                                     self.layer_norm_eps)(latents, x)

        latents = nn.LayerNorm(epsilon=self.layer_norm_eps)(latents)
        return latents


class Encoder(nn.Module):
    patch_size: int
    grid_size: Tuple
    emb_dim: int
    num_latents: int
    depth: int
    num_heads: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5
    pos_emb_init: Callable = get_2d_sincos_pos_embed

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape

        # Patch embedding
        x = PatchEmbed(self.patch_size, self.emb_dim)(x)

        pos_emb = self.variable(
            "pos_emb",
            "enc_pos_emb",
            self.pos_emb_init,
            self.emb_dim,
            (self.grid_size[0] // self.patch_size[0], self.grid_size[1] // self.patch_size[1]),
        )

        # Interpolate positional embeddings to match the input shape
        pos_emb_interp = pos_emb.value.reshape(1,
                                             self.grid_size[0] // self.patch_size[0],
                                             self.grid_size[1] // self.patch_size[1],
                                             self.emb_dim)
        pos_emb_interp = jax.image.resize(pos_emb_interp,
                                            (1, h // self.patch_size[0], w // self.patch_size[1], self.emb_dim),
                                            method='bilinear')
        pos_emb_interp = rearrange(pos_emb_interp, 'b h w d -> b (h w) d')
        x = x + pos_emb_interp

        # Embed into tokens of the same length as the latents
        x = PerceiverBlock(emb_dim=self.emb_dim, depth=2, num_heads=self.num_heads, num_latents=self.num_latents)(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

        # Transformer
        for _ in range(self.depth):
            x = SelfAttnBlock(
                self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
            )(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        return x


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied

    def setup(self):
        # Initialize period parameters and store them in a flax frozen dict
        self.period_params = {f"period_{idx}": period for idx, period in enumerate(self.period)}

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x


class Decoder(nn.Module):
    fourier_freq: float = 1.0
    period: Union[None, Dict] = None
    dec_depth: int = 2
    dec_num_heads: int = 8
    dec_emb_dim: int = 256
    mlp_ratio: int = 1
    out_dim: int = 1
    num_mlp_layers: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, coords):
        b, n, c = x.shape

        # # Embed periodic boundary conditions if specified
        if self.period is True:
            # Hardcode the periodicity, assuming the domain is [0, 1]x[0, 1]
            coords = PeriodEmbs(period=(2 * jnp.pi, 2 * jnp.pi), axis=(0, 1))(coords)

        coords = FourierEmbs(embed_scale=self.fourier_freq, embed_dim=self.dec_emb_dim)(coords)
        coords = repeat(coords, 'd -> b n d', n=1, b=b)

        x = nn.Dense(self.dec_emb_dim)(x)
        for _ in range(self.dec_depth):
            coords = CrossAttnBlock(num_heads=self.dec_num_heads,
                               emb_dim=self.dec_emb_dim,
                               mlp_ratio=self.mlp_ratio,
                               layer_norm_eps=self.layer_norm_eps)(coords, x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)
        # x = nn.Dense(self.out_dim)(x)

        x = Mlp(num_layers=self.num_mlp_layers,
                hidden_dim=self.dec_emb_dim,
                out_dim=self.out_dim,
                layer_norm_eps=self.layer_norm_eps)(x)

        return x