"""
Simplified from MPP Code base for fixed history training.

JAX/Flax implementation with channels-last format.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Sequence, Optional
import numpy as np


class hMLPStem(nn.Module):
    """Image to Patch Embedding

    Args:
        hidden_dim: Hidden dimension size
        groups: Number of groups for GroupNorm
        n_spatial_dims: Number of spatial dimensions (1, 2, or 3)
    """
    hidden_dim: int
    groups: int = 8
    n_spatial_dims: int = 2

    @nn.compact
    def __call__(self, x):
        # Input: (B, *spatial, C) channels-last

        # First conv block
        x = nn.Conv(
            features=self.hidden_dim // 4,
            kernel_size=(4,) * self.n_spatial_dims,
            strides=(4,) * self.n_spatial_dims,
            use_bias=False
        )(x)
        x = nn.GroupNorm(num_groups=self.groups, use_bias=True, use_scale=True)(x)
        x = nn.gelu(x)

        # Second conv block
        x = nn.Conv(
            features=self.hidden_dim // 4,
            kernel_size=(2,) * self.n_spatial_dims,
            strides=(2,) * self.n_spatial_dims,
            use_bias=False
        )(x)
        x = nn.GroupNorm(num_groups=self.groups, use_bias=True, use_scale=True)(x)
        x = nn.gelu(x)

        # Third conv block
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(2,) * self.n_spatial_dims,
            strides=(2,) * self.n_spatial_dims,
            use_bias=False
        )(x)
        x = nn.GroupNorm(num_groups=self.groups, use_bias=True, use_scale=True)(x)

        return x


class hMLPOutput(nn.Module):
    """Patch to Image De-bedding

    Args:
        out_dim: Output dimension
        hidden_dim: Hidden dimension size
        groups: Number of groups for GroupNorm
        n_spatial_dims: Number of spatial dimensions (1, 2, or 3)
    """
    out_dim: int
    hidden_dim: int = 768
    groups: int = 8
    n_spatial_dims: int = 2

    @nn.compact
    def __call__(self, x):
        # Input: (B, *spatial, C) channels-last

        # First deconv block
        x = nn.ConvTranspose(
            features=self.hidden_dim // 4,
            kernel_size=(2,) * self.n_spatial_dims,
            strides=(2,) * self.n_spatial_dims,
            use_bias=False
        )(x)
        x = nn.GroupNorm(num_groups=self.groups, use_bias=True, use_scale=True)(x)
        x = nn.gelu(x)

        # Second deconv block
        x = nn.ConvTranspose(
            features=self.hidden_dim // 4,
            kernel_size=(2,) * self.n_spatial_dims,
            strides=(2,) * self.n_spatial_dims,
            use_bias=False
        )(x)
        x = nn.GroupNorm(num_groups=self.groups, use_bias=True, use_scale=True)(x)
        x = nn.gelu(x)

        # Third deconv block
        x = nn.ConvTranspose(
            features=self.out_dim,
            kernel_size=(4,) * self.n_spatial_dims,
            strides=(4,) * self.n_spatial_dims,
            use_bias=False
        )(x)

        return x


class AxialAttentionBlock(nn.Module):
    """Axial Attention Block with parallel MLP

    Performs attention along each spatial axis independently and combines results.

    Args:
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        n_spatial_dims: Number of spatial dimensions (2 or 3)
        drop_path_rate: Stochastic depth rate
        layer_scale_init_value: Initial value for layer scale parameter
    """
    hidden_dim: int = 768
    num_heads: int = 8
    n_spatial_dims: int = 2
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6

    def setup(self):
        self.head_dim = self.hidden_dim // self.num_heads

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Input: (B, *spatial, C) channels-last
        residual = x

        # Layer norm
        x = nn.LayerNorm(use_bias=False)(x)

        # Fused projection for Q, K, V, and feedforward
        total_features = self.hidden_dim * 3 + 4 * self.hidden_dim
        x_fused = nn.Dense(features=total_features)(x)

        # Split into Q, K, V, and FF
        # q, k, v, ff = jnp.split(
        #     x_fused,
        #     jnp.cumsum(jnp.array(fused_heads[:-1])),
        #     axis=-1
        # )

        split_indices = [self.hidden_dim, 2 * self.hidden_dim, 3 * self.hidden_dim]
        q, k, v, ff = jnp.split(x_fused, split_indices, axis=-1)

        # Reshape for multi-head attention: (..., C) -> (..., num_heads, head_dim)
        batch_size = x.shape[0]
        spatial_shape = x.shape[1:-1]

        def reshape_heads(t):
            # Reshape to (..., num_heads, head_dim)
            return t.reshape(batch_size, *spatial_shape, self.num_heads, self.head_dim)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Apply Q, K normalization
        q = nn.LayerNorm(use_bias=False)(q)
        k = nn.LayerNorm(use_bias=False)(k)

        # Perform axial attention
        out = jnp.zeros_like(q)

        if self.n_spatial_dims == 2:
            # Shape: (B, H, W, num_heads, head_dim)
            # Axis 0: along height (fixing W)
            q_h = jnp.transpose(q, (0, 2, 1, 3, 4))  # (B, W, H, heads, dim)
            k_h = jnp.transpose(k, (0, 2, 1, 3, 4))
            v_h = jnp.transpose(v, (0, 2, 1, 3, 4))

            attn_h = self._compute_attention(q_h, k_h, v_h)
            attn_h = jnp.transpose(attn_h, (0, 2, 1, 3, 4))  # Back to (B, H, W, heads, dim)
            out = out + attn_h

            # Axis 1: along width (fixing H)
            attn_w = self._compute_attention(q, k, v)
            out = out + attn_w

        elif self.n_spatial_dims == 3:
            # Shape: (B, H, W, D, num_heads, head_dim)
            # Axis 0: along H
            q_h = jnp.transpose(q, (0, 2, 3, 1, 4, 5))  # (B, W, D, H, heads, dim)
            k_h = jnp.transpose(k, (0, 2, 3, 1, 4, 5))
            v_h = jnp.transpose(v, (0, 2, 3, 1, 4, 5))
            attn_h = self._compute_attention(q_h, k_h, v_h)
            attn_h = jnp.transpose(attn_h, (0, 3, 1, 2, 4, 5))
            out = out + attn_h

            # Axis 1: along W
            q_w = jnp.transpose(q, (0, 1, 3, 2, 4, 5))  # (B, H, D, W, heads, dim)
            k_w = jnp.transpose(k, (0, 1, 3, 2, 4, 5))
            v_w = jnp.transpose(v, (0, 1, 3, 2, 4, 5))
            attn_w = self._compute_attention(q_w, k_w, v_w)
            attn_w = jnp.transpose(attn_w, (0, 1, 3, 2, 4, 5))
            out = out + attn_w

            # Axis 2: along D
            attn_d = self._compute_attention(q, k, v)
            out = out + attn_d

        # Reshape back: (..., num_heads, head_dim) -> (..., hidden_dim)
        out = out.reshape(batch_size, *spatial_shape, self.hidden_dim)

        # Output projection
        out = nn.Dense(features=self.hidden_dim)(out)

        # Parallel MLP
        mlp_out = nn.gelu(ff)
        mlp_out = nn.Dense(features=self.hidden_dim)(mlp_out)

        # Combine attention and MLP
        x = out + mlp_out

        # Layer scale
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                'gamma',
                lambda rng, shape: self.layer_scale_init_value * jnp.ones(shape),
                (self.hidden_dim,)
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
        return residual + x

    def _compute_attention(self, q, k, v):
        """Compute scaled dot-product attention.

        Input shapes should be: (B, ..., seq_len, num_heads, head_dim)
        where ... represents fixed spatial dimensions and seq_len is the axis to attend over.
        """
        # Reshape to (B * ..., num_heads, seq_len, head_dim) for batched attention
        original_shape = q.shape
        batch_and_spatial = np.prod(original_shape[:-3])
        seq_len = original_shape[-3]

        q_reshaped = q.reshape(batch_and_spatial, seq_len, self.num_heads, self.head_dim)
        k_reshaped = k.reshape(batch_and_spatial, seq_len, self.num_heads, self.head_dim)
        v_reshaped = v.reshape(batch_and_spatial, seq_len, self.num_heads, self.head_dim)

        # Transpose to (B * ..., num_heads, seq_len, head_dim)
        q_reshaped = jnp.transpose(q_reshaped, (0, 2, 1, 3))
        k_reshaped = jnp.transpose(k_reshaped, (0, 2, 1, 3))
        v_reshaped = jnp.transpose(v_reshaped, (0, 2, 1, 3))

        # Compute attention scores
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum('...qhd,...khd->...hqk', q_reshaped, k_reshaped) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, v_reshaped)

        # Transpose back and reshape to original shape
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(original_shape)

        return attn_output


class AViT(nn.Module):
    """Axial Vision Transformer

    Uses axial attention to predict forward dynamics. This simplified version
    processes spatial data with axial attention blocks.

    Args:
        out_dim: Output dimension
        n_spatial_dims: Number of spatial dimensions (2 or 3)
        spatial_resolution: Tuple of spatial dimensions
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        processor_blocks: Number of attention blocks
        drop_path: Maximum drop path rate (linearly scaled across blocks)
    """
    out_dim: int
    n_spatial_dims: int
    spatial_resolution: Tuple[int, ...]
    hidden_dim: int = 768
    num_heads: int = 12
    num_groups: int = 8
    processor_blocks: int = 8
    drop_path: float = 0.0
    model_name: Optional[str] = None

    def setup(self):
        # Patch size hardcoded at 16
        self.patch_size = 16

        # Calculate drop path rates (linearly scaled)
        self.drop_path_rates = np.linspace(0, self.drop_path, self.processor_blocks)

        # Calculate positional encoding size
        pe_spatial_size = tuple(int(k // self.patch_size) for k in self.spatial_resolution)
        self.pe_shape = pe_spatial_size + (self.hidden_dim,)

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Input: (B, *spatial, C) channels-last
        dim_in = x.shape[-1]

        # Positional encoding
        absolute_pe = self.param(
            'absolute_pe',
            lambda rng, shape: 0.02 * jax.random.normal(rng, shape),
            self.pe_shape
        )

        # Embedding
        x = hMLPStem(
            hidden_dim=self.hidden_dim,
            n_spatial_dims=self.n_spatial_dims,
            groups=self.num_groups
        )(x)

        # Add positional encoding
        x = x + absolute_pe

        # Process through attention blocks
        for i in range(self.processor_blocks):
            x = AxialAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                n_spatial_dims=self.n_spatial_dims,
                drop_path_rate=self.drop_path_rates[i],
            )(x, deterministic=deterministic)

        # Decode
        x = hMLPOutput(
            out_dim=self.out_dim,
            hidden_dim=self.hidden_dim,
            n_spatial_dims=self.n_spatial_dims,
            groups=self.num_groups
        )(x)

        return x


