# Acknowledgment: This implementation builds upon by
# Ashish Kumar’s FlaxDiff (https://github.com/AshishKumar4/FlaxDiff).
# Sincere thanks to the author for the clear and well-structured reference.


import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Callable, Sequence, Any, Union, Tuple, Optional
from flax.typing import Dtype, PrecisionLike
import einops
import functools
import math
from common import kernel_init
import jax.experimental.pallas.ops.tpu.flash_attention

class EfficientAttention(nn.Module):
    """
    Based on the pallas attention implementation.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_bias: bool = True
    # kernel_init: Callable = kernel_init(1.0)
    force_fp32_for_softmax: bool = True

    def setup(self):
        inner_dim = self.dim_head * self.heads
        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        dense = functools.partial(
            nn.Dense,
            self.heads * self.dim_head,
            precision=self.precision, 
            use_bias=self.use_bias, 
            # kernel_init=self.kernel_init, 
            dtype=self.dtype
        )
        self.query = dense(name="to_q")
        self.key = dense(name="to_k")
        self.value = dense(name="to_v")
        
        self.proj_attn = nn.DenseGeneral(
            self.query_dim, 
            use_bias=False, 
            precision=self.precision, 
            # kernel_init=self.kernel_init,
            dtype=self.dtype,
            name="to_out_0"
        )
        # self.attnfn = make_fast_generalized_attention(qkv_dim=inner_dim, lax_scan_unroll=16)
    
    def _reshape_tensor_to_head_dim(self, tensor):
        batch_size, _, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        return tensor
    
    def _reshape_tensor_from_head_dim(self, tensor):
        batch_size, _, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size, 1, seq_len, dim * head_size)
        return tensor

    @nn.compact
    def __call__(self, x:jax.Array, context=None):
        # print(x.shape)
        # x has shape [B, H * W, C]
        context = x if context is None else context
        
        orig_x_shape = x.shape
        if len(x.shape) == 4:
            B, H, W, C = x.shape
            x = x.reshape((B, 1, H * W, C))
        else:
            B, SEQ, C = x.shape
            x = x.reshape((B, 1, SEQ, C))
        
        if len(context.shape) == 4:
            B, _H, _W, _C = context.shape
            context = context.reshape((B, 1, _H * _W, _C))
        else:
            B, SEQ, _C = context.shape
            context = context.reshape((B, 1, SEQ, _C))
        
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        
        query = self._reshape_tensor_to_head_dim(query)
        key = self._reshape_tensor_to_head_dim(key)
        value = self._reshape_tensor_to_head_dim(value)
        
        hidden_states = jax.experimental.pallas.ops.tpu.flash_attention.flash_attention(
            query, key, value, None
        )
        
        hidden_states = self._reshape_tensor_from_head_dim(hidden_states)
        
        
        # hidden_states = nn.dot_product_attention(
        #     query, key, value, dtype=self.dtype, broadcast_dropout=False, dropout_rng=None, precision=self.precision
        # )
        
        proj = self.proj_attn(hidden_states)
        
        proj = proj.reshape(orig_x_shape)
        
        return proj

class NormalAttention(nn.Module):
    """
    Simple implementation of the normal attention.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_bias: bool = True
    # kernel_init: Callable = kernel_init(1.0)
    force_fp32_for_softmax: bool = True

    def setup(self):
        inner_dim = self.dim_head * self.heads
        dense = functools.partial(
            nn.DenseGeneral,
            features=[self.heads, self.dim_head], 
            axis=-1, 
            precision=self.precision, 
            use_bias=self.use_bias, 
            # kernel_init=self.kernel_init, 
            dtype=self.dtype
        )
        self.query = dense(name="to_q")
        self.key = dense(name="to_k")
        self.value = dense(name="to_v")

        self.proj_attn = nn.DenseGeneral(
            self.query_dim, 
            axis=(-2, -1), 
            precision=self.precision, 
            use_bias=self.use_bias, 
            dtype=self.dtype, 
            name="to_out_0",
            # kernel_init=self.kernel_init
            # kernel_init=jax.nn.initializers.xavier_uniform()
        )

    @nn.compact
    def __call__(self, x, context=None):
        # x has shape [B, H, W, C]
        orig_x_shape = x.shape
        if len(x.shape) == 4:
            B, H, W, C = x.shape
            x = x.reshape((B, H*W, C))
        context = x if context is None else context
        if len(context.shape) == 4:
            context = context.reshape((B, H*W, C))
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        
        hidden_states = nn.dot_product_attention(
            query, key, value, dtype=self.dtype, broadcast_dropout=False, 
            dropout_rng=None, precision=self.precision, force_fp32_for_softmax=self.force_fp32_for_softmax,
            deterministic=True
        )
        proj = self.proj_attn(hidden_states)
        proj = proj.reshape(orig_x_shape)
        return proj

class FlaxGEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    precision: Any = jax.lax.Precision.DEFAULT

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype, precision=self.precision)

    def __call__(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=-1)
        return hidden_linear * nn.gelu(hidden_gelu)
    
class FlaxFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
    [`FeedForward`] class, with the following simplifications:
    - The activation function is currently hardcoded to a gated linear unit from:
    https://arxiv.org/abs/2002.05202
    - `dim_out` is equal to `dim`.
    - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dtype: jnp.dtype = jnp.float32
    precision: Any = jax.lax.Precision.DEFAULT

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = FlaxGEGLU(self.dim, self.dtype, precision=self.precision)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision)

    def __call__(self, hidden_states):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states

class BasicTransformerBlock(nn.Module):
    # Has self and cross attention
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_bias: bool = True
    # kernel_init: Callable = kernel_init(1.0)
    use_flash_attention:bool = False
    use_cross_only:bool = False
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-4
    
    def setup(self):
        if self.use_flash_attention:
            attenBlock = EfficientAttention
        else:
            attenBlock = NormalAttention
            
        self.attention1 = attenBlock(
         query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention1',
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            # kernel_init=self.kernel_init,
            force_fp32_for_softmax=self.force_fp32_for_softmax
        )
        self.attention2 = attenBlock(
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention2',
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            # kernel_init=self.kernel_init,
            force_fp32_for_softmax=self.force_fp32_for_softmax
        )
        
        self.ff = FlaxFeedForward(dim=self.query_dim)
        self.norm1 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm2 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm3 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        
    @nn.compact
    def __call__(self, hidden_states, context=None):
        if self.only_pure_attention:
            return self.attention2(hidden_states, context)
        
        # self attention
        if not self.use_cross_only:
            hidden_states = hidden_states + self.attention1(self.norm1(hidden_states))
        
        # cross attention
        hidden_states = hidden_states + self.attention2(self.norm2(hidden_states), context)
        # feed forward
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        
        return hidden_states

class TransformerBlock(nn.Module):
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_projection: bool = False
    use_flash_attention:bool = False
    use_self_and_cross:bool = True
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    # kernel_init: Callable = kernel_init(1.0)
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4

    @nn.compact
    def __call__(self, x, context=None):
        inner_dim = self.heads * self.dim_head
        C = x.shape[-1]
        if self.norm_inputs:
            x = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(x)
        if self.use_projection == True:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=inner_dim, 
                                       use_bias=False, precision=self.precision, 
                                    #    kernel_init=self.kernel_init,
                                       dtype=self.dtype, name=f'project_in')(x)
            else:
                projected_x = nn.Conv(
                    features=inner_dim, kernel_size=(1, 1),
                    # kernel_init=self.kernel_init,
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name=f'project_in_conv',
                )(x)
        else:
            projected_x = x
            inner_dim = C
            
        context = projected_x if context is None else context

        projected_x = BasicTransformerBlock(
            query_dim=inner_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention',
            precision=self.precision,
            use_bias=False,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            use_cross_only=(not self.use_self_and_cross),
            only_pure_attention=self.only_pure_attention,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            norm_epsilon=self.norm_epsilon
            # kernel_init=self.kernel_init
        )(projected_x, context)
        
        if self.use_projection == True:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=C, precision=self.precision, 
                                       dtype=self.dtype, use_bias=False, 
                                    #    kernel_init=self.kernel_init,
                                       name=f'project_out')(projected_x)
            else:
                projected_x = nn.Conv(
                    features=C, kernel_size=(1, 1),
                    # kernel_init=self.kernel_i nit,
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name=f'project_out_conv',
                )(projected_x)
                
        if self.only_pure_attention or self.explicitly_add_residual:
            projected_x = x + projected_x
            
        out = projected_x
        return out