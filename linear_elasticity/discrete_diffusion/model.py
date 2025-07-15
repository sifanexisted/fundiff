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
            (h // self.patch_size[0], w // self.patch_size[1]),
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

        # x = x + pos_emb.value

        # Transformer
        for _ in range(self.depth):
            x = SelfAttnBlock(
                self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
            )(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        return x


class Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x

class Decoder(nn.Module):
    patch_size: Tuple[int, int] = (16, 16)
    image_size: Tuple[int, int] = (256, 256)
    num_mlp_layers: int = 2
    dec_emb_dim: int = 256
    out_dim: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, l, d = x.shape  # l = H / P * W / P
        patch_height = self.image_size[0] // self.patch_size[0]
        patch_width = self.image_size[1] // self.patch_size[1]

        if self.num_mlp_layers > 0:
            x = Mlp(self.num_mlp_layers, self.dec_emb_dim, self.patch_size[0] * self.patch_size[1] * self.out_dim)(x)
        else:
            x = nn.Dense(self.patch_size[0] * self.patch_size[1] * self.out_dim)(x)

        x = jnp.reshape(
            x,
            (
                b,
                patch_height,
                patch_width,
                self.patch_size[0],
                self.patch_size[1],
                -1,
                ),
            )
        x = jnp.swapaxes(x, 2, 3)
        x = jnp.reshape(
            x,
            (
                b,
                patch_height * self.patch_size[0],
                patch_width * self.patch_size[1],
                -1,
                ),
            )
        # x shape (B, H, W, C)

        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    emb_dim: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]




class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    emb_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        # Calculate adaLn modulation parameters.
        c = nn.gelu(c)
        c = nn.Dense(6 * self.emb_dim, kernel_init=nn.initializers.constant(0.))(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)

        # Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
                                                 num_heads=self.num_heads)(x_modulated, x_modulated)
        x = x + (gate_msa[:, None] * attn_x)

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x


pos_emb_init = get_1d_sincos_pos_embed

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    model_name: Optional[str]
    grid_size: tuple
    emb_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    out_dim: int

    @nn.compact
    def __call__(self, x, t, c=None):
        # (x = (B, L, C) image, t = (B,) timesteps, c = (B, L, C) conditioning
        # h = self.grid_size[0]
        #
        b, l, _ = x.shape

        pos_emb = self.variable(
            "pos_emb",
            "enc_emb",
            get_1d_sincos_pos_embed,
            self.emb_dim,
            l,
        )

        x = nn.Dense(self.emb_dim)(x)
        x = x + pos_emb.value

        if c is not None:
            c = nn.Dense(self.emb_dim)(c)
            x = x + c

        t = TimestepEmbedder(self.emb_dim)(t)  # (B, emb_dim)

        for _ in range(self.depth):
            x = DiTBlock(self.emb_dim, self.num_heads, self.mlp_ratio)(x, t)
        # x = FinalLayer(self.out_dim, self.emb_dim)(x, t) # (B, num_patches, p*p*c)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.out_dim)(x)

        return x