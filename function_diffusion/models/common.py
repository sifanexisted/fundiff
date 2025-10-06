# Acknowledgment: This implementation builds upon by
# Ashish Kumar’s FlaxDiff (https://github.com/AshishKumar4/FlaxDiff).
# Sincere thanks to the author for the clear and well-structured reference.


import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import Optional, Any, Callable, Sequence, Union
from flax.typing import Dtype, PrecisionLike
from typing import Dict, Callable, Sequence, Any, Union
import einops
from functools import partial
import math
from einops import rearrange

# Kernel initializer to use
def kernel_init(scale=1.0, dtype=jnp.float32):
    scale = max(scale, 1e-10)
    return nn.initializers.variance_scaling(scale=scale, mode="fan_avg", distribution="truncated_normal", dtype=dtype)


class WeightStandardizedConv(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """
    features: int
    kernel_size: Sequence[int] = 3
    strides: Union[None, int, Sequence[int]] = 1
    padding: Any = 1
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    param_dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, x):
        """
        Applies a weight standardized convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        x = x.astype(self.dtype)

        conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides = self.strides,
            padding=self.padding,
            dtype=self.dtype,
            param_dtype = self.param_dtype,
            parent=None)

        kernel_init = lambda  rng, x: conv.init(rng,x)['params']['kernel']
        bias_init = lambda  rng, x: conv.init(rng,x)['params']['bias']

        # standardize kernel
        kernel = self.param('kernel', kernel_init, x)
        eps = 1e-5 if self.dtype == jnp.float32 else 1e-3
        # reduce over dim_out
        redux = tuple(range(kernel.ndim - 1))
        mean = jnp.mean(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        var = jnp.var(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        standardized_kernel = (kernel - mean)/jnp.sqrt(var + eps)

        bias = self.param('bias',bias_init, x)

        return(conv.apply({'params': {'kernel': standardized_kernel, 'bias': bias}},x))

class PixelShuffle(nn.Module):
    scale: int

    @nn.compact
    def __call__(self, x):
        up = einops.rearrange(
            x,
            pattern="b h w (h2 w2 c) -> b (h h2) (w w2) c",
            h2=self.scale,
            w2=self.scale,
        )
        return up

class TimeEmbedding(nn.Module):
    features:int
    nax_positions:int=10000

    def setup(self):
        half_dim = self.features // 2
        emb = jnp.log(self.nax_positions) / (half_dim - 1)
        emb = jnp.exp(-emb * jnp.arange(half_dim, dtype=jnp.float32))
        self.embeddings = emb

    def __call__(self, x):
        x = jax.lax.convert_element_type(x, jnp.float32)
        emb = x[:, None] * self.embeddings[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

class FourierEmbedding(nn.Module):
    features:int
    scale:int = 16

    def setup(self):
        self.freqs = jax.random.normal(jax.random.PRNGKey(42), (self.features // 2, ), dtype=jnp.float32) * self.scale

    def __call__(self, x):
        x = jax.lax.convert_element_type(x, jnp.float32)
        emb = x[:, None] * (2 * jnp.pi * self.freqs)[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

class TimeProjection(nn.Module):
    features:int
    activation:Callable=jax.nn.gelu

    @nn.compact
    def __call__(self, x):
        x = nn.DenseGeneral(
            self.features, 
        )(x)
        x = self.activation(x)
        x = nn.DenseGeneral(
            self.features, 
        )(x)
        x = self.activation(x)
        return x

class SeparableConv(nn.Module):
    features:int
    kernel_size:tuple=(3, 3)
    strides:tuple=(1, 1)
    use_bias:bool=False
    padding:str="SAME"
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        depthwise = nn.Conv(
            features=in_features, kernel_size=self.kernel_size,
            strides=self.strides,
            feature_group_count=in_features, use_bias=self.use_bias,
            padding=self.padding,
            dtype=self.dtype,
            precision=self.precision
        )(x)
        pointwise = nn.Conv(
            features=self.features, kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=self.use_bias,
            dtype=self.dtype,
            precision=self.precision
        )(depthwise)
        return pointwise

class ConvLayer(nn.Module):
    conv_type:str
    features:int
    kernel_size:tuple=(3, 3)
    strides:tuple=(1, 1)
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        # conv_type can be "conv", "separable", "conv_transpose"
        if self.conv_type == "conv":
            self.conv = nn.Conv(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dtype=self.dtype,
                precision=self.precision
            )
        elif self.conv_type == "w_conv":
            self.conv = WeightStandardizedConv(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding="SAME",
                param_dtype=self.dtype,
                dtype=self.dtype,
                precision=self.precision
            )
        elif self.conv_type == "separable":
            self.conv = SeparableConv(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dtype=self.dtype,
                precision=self.precision
            )
        elif self.conv_type == "conv_transpose":
            self.conv = nn.ConvTranspose(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dtype=self.dtype,
                precision=self.precision
            )

    def __call__(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    features:int
    scale:int
    activation:Callable=jax.nn.swish
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, residual=None):
        out = x
        # out = PixelShuffle(scale=self.scale)(out)
        B, H, W, C = x.shape
        out = jax.image.resize(x, (B, H * self.scale, W * self.scale, C), method="nearest")
        out = ConvLayer(
            "conv",
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            precision=self.precision,
        )(out)
        if residual is not None:
            out = jnp.concatenate([out, residual], axis=-1)
        return out

class Downsample(nn.Module):
    features:int
    scale:int
    activation:Callable=jax.nn.swish
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, residual=None):
        out = ConvLayer(
            "conv",
            features=self.features,
            kernel_size=(3, 3),
            strides=(2, 2),
            dtype=self.dtype,
            precision=self.precision,
        )(x)
        if residual is not None:
            if residual.shape[1] > out.shape[1]:
                residual = nn.avg_pool(residual, window_shape=(2, 2), strides=(2, 2), padding="SAME")
            out = jnp.concatenate([out, residual], axis=-1)
        return out


def l2norm(t, axis=1, eps=1e-6): # Increased epsilon from 1e-12
    denom = jnp.clip(jnp.linalg.norm(t, ord=2, axis=axis, keepdims=True), eps)
    out = t/denom
    return (out)


class ResidualBlock(nn.Module):
    conv_type:str
    features:int
    kernel_size:tuple=(3, 3)
    strides:tuple=(1, 1)
    padding:str="SAME"
    activation:Callable=jax.nn.swish
    direction:str=None
    res:int=2
    norm_groups:int=8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    named_norms:bool=False
    norm_epsilon: float = 1e-4 # Added epsilon parameter, increased default
    
    def setup(self):
        if self.norm_groups > 0:
            norm = partial(nn.GroupNorm, self.norm_groups, epsilon=self.norm_epsilon)
            self.norm1 = norm(name="GroupNorm_0") if self.named_norms else norm()
            self.norm2 = norm(name="GroupNorm_1") if self.named_norms else norm()
        else:
            norm = partial(nn.RMSNorm, epsilon=self.norm_epsilon)
            self.norm1 = norm()
            self.norm2 = norm()

    @nn.compact
    def __call__(self, x:jax.Array, temb:jax.Array, textemb:jax.Array=None, extra_features:jax.Array=None):
        residual = x
        out = self.norm1(x)
        # out = nn.RMSNorm()(x)
        out = self.activation(out)

        out = ConvLayer(
            self.conv_type,
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            name="conv1",
            dtype=self.dtype,
            precision=self.precision
        )(out)

        temb = nn.DenseGeneral(
            features=self.features, 
            name="temb_projection",
            dtype=self.dtype,
            precision=self.precision)(temb)
        temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
        # scale, shift = jnp.split(temb, 2, axis=-1)
        # out = out * (1 + scale) + shift
        out = out + temb

        out = self.norm2(out)
        # out = nn.RMSNorm()(out)
        out = self.activation(out)

        out = ConvLayer(
            self.conv_type,
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            name="conv2",
            dtype=self.dtype,
            precision=self.precision
        )(out)

        if residual.shape != out.shape:
            residual = ConvLayer(
                self.conv_type,
                features=self.features,
                kernel_size=(1, 1),
                strides=1,
                name="residual_conv",
                dtype=self.dtype,
                precision=self.precision
            )(residual)
        out = out + residual

        out = jnp.concatenate([out, extra_features], axis=-1) if extra_features is not None else out

        return out