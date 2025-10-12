# Acknowledgment: This implementation builds upon by
# Ashish Kumar’s FlaxDiff (https://github.com/AshishKumar4/FlaxDiff).
# Sincere thanks to the author for the clear and well-structured reference.

from functools import partial
import flax.linen as nn

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from typing import Dict, Callable, Sequence, Any, Union, Optional
import einops
from einops import rearrange, repeat

from .common import kernel_init, ConvLayer, Downsample, Upsample, FourierEmbedding, TimeProjection, ResidualBlock
from .attention import TransformerBlock
from .vit import get_2d_sincos_pos_embed, SelfAttnBlock



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




class Unet(nn.Module):
    out_dim: int = 3
    emb_features: int = 64 * 4
    feature_depths: list = (64, 128, 256, 512)
    attention_configs: list = ({"heads": 8}, {"heads": 8}, {"heads": 8}, {"heads": 8})
    num_enc_blocks: int = 4
    num_res_blocks: int = 2
    num_middle_res_blocks: int = 1
    activation: Callable = jax.nn.swish
    norm_groups: int = 8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    named_norms: bool = False  # This is for backward compatibility reasons; older checkpoints have named norms
    model_name: Optional[str] = None

    def setup(self):
        if self.norm_groups > 0:
            norm = partial(nn.GroupNorm, self.norm_groups)
            self.conv_out_norm = norm(name="GroupNorm_0") if self.named_norms else norm()
        else:
            norm = partial(nn.RMSNorm, 1e-5)
            self.conv_out_norm = norm()
        
    @nn.compact
    def __call__(self, x, temb, context):
        # print("embedding features", self.emb_features)
        temb = FourierEmbedding(features=self.emb_features)(temb)
        temb = TimeProjection(features=self.emb_features)(temb)

        if context is not None:
            b, H, W, C = context.shape

            context = PatchEmbed(patch_size=(16, 16), emb_dim=self.emb_features)(context)
            _, TS, TC = context.shape

            pos_emb = self.variable(
                "pos_emb",
                "enc_pos_emb",
                get_2d_sincos_pos_embed,
                self.emb_features,
                (H // 16, W // 16),
            )

            context = context + pos_emb.value

            for _i in range(self.num_enc_blocks):
                context = SelfAttnBlock(
                    num_heads=8,
                    emb_dim=self.emb_features,
                    mlp_ratio=1,
                    layer_norm_eps=1e-5,
                )(context)
            context = nn.LayerNorm(epsilon=1e-5)(context)

        # print("time embedding", temb.shape)
        feature_depths = self.feature_depths
        attention_configs = self.attention_configs

        conv_type = up_conv_type = down_conv_type = middle_conv_type = "conv"
        # middle_conv_type = "separable"
        
        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        downs = [x]

        # Downscaling blocks
        for i, (dim_out, attention_config) in enumerate(zip(feature_depths, attention_configs)):
            dim_in = x.shape[-1]
            # dim_in = dim_out
            for j in range(self.num_res_blocks):
                x = ResidualBlock(
                    down_conv_type,
                    name=f"down_{i}_residual_{j}",
                    features=dim_in,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                    named_norms=self.named_norms
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:   # Apply attention only on the last block
                    x = TransformerBlock(heads=attention_config['heads'], dtype=attention_config.get('dtype', jnp.float32),
                                        dim_head=dim_in // attention_config['heads'],
                                        use_flash_attention=attention_config.get("flash_attention", False),
                                        use_projection=attention_config.get("use_projection", False),
                                        use_self_and_cross=attention_config.get("use_self_and_cross", True),
                                        precision=attention_config.get("precision", self.precision),
                                        only_pure_attention=attention_config.get("only_pure_attention", True),
                                        force_fp32_for_softmax=attention_config.get("force_fp32_for_softmax", False),
                                        norm_inputs=attention_config.get("norm_inputs", True),
                                        explicitly_add_residual=attention_config.get("explicitly_add_residual", True),
                                        name=f"down_{i}_attention_{j}")(x, context)
                # print("down residual for feature level", i, "is of shape", x.shape, "features", dim_in)
                downs.append(x)
            if i != len(feature_depths) - 1:
                # print("Downsample", i, x.shape)
                x = Downsample(
                    features=dim_out,
                    scale=2,
                    activation=self.activation,
                    name=f"down_{i}_downsample",
                    dtype=self.dtype,
                    precision=self.precision
                )(x)

        # Middle Blocks
        middle_dim_out = self.feature_depths[-1]
        middle_attention = self.attention_configs[-1]
        for j in range(self.num_middle_res_blocks):
            x = ResidualBlock(
                middle_conv_type,
                name=f"middle_res1_{j}",
                features=middle_dim_out,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision,
                named_norms=self.named_norms
            )(x, temb)
            if middle_attention is not None and j == self.num_middle_res_blocks - 1:   # Apply attention only on the last block
                x = TransformerBlock(heads=middle_attention['heads'], dtype=middle_attention.get('dtype', jnp.float32), 
                                    dim_head=middle_dim_out // middle_attention['heads'],
                                    use_flash_attention=middle_attention.get("flash_attention", False),
                                    use_linear_attention=False,
                                    use_projection=middle_attention.get("use_projection", False),
                                    use_self_and_cross=False,
                                    precision=middle_attention.get("precision", self.precision),
                                    only_pure_attention=middle_attention.get("only_pure_attention", True),
                                    force_fp32_for_softmax=middle_attention.get("force_fp32_for_softmax", False),
                                    norm_inputs=middle_attention.get("norm_inputs", True),
                                    explicitly_add_residual=middle_attention.get("explicitly_add_residual", True),
                                    name=f"middle_attention_{j}")(x, context)
            x = ResidualBlock(
                middle_conv_type,
                name=f"middle_res2_{j}",
                features=middle_dim_out,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision,
                named_norms=self.named_norms
            )(x, temb)

        # Upscaling Blocks
        for i, (dim_out, attention_config) in enumerate(zip(reversed(feature_depths), reversed(attention_configs))):
            # print("Upscaling", i, "features", dim_out)
            for j in range(self.num_res_blocks):
                x = jnp.concatenate([x, downs.pop()], axis=-1)
                # print("concat==> ", i, "concat", x.shape)
                # kernel_size = (1 + 2 * (j + 1), 1 + 2 * (j + 1))
                kernel_size = (3, 3)
                x = ResidualBlock(
                    up_conv_type,# if j == 0 else "separable",
                    name=f"up_{i}_residual_{j}",
                    features=dim_out,
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                    named_norms=self.named_norms
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:   # Apply attention only on the last block
                    x = TransformerBlock(heads=attention_config['heads'], dtype=attention_config.get('dtype', jnp.float32), 
                                        dim_head=dim_out // attention_config['heads'],
                                        use_flash_attention=attention_config.get("flash_attention", False),
                                        use_projection=attention_config.get("use_projection", False),
                                        use_self_and_cross=attention_config.get("use_self_and_cross", True),
                                        precision=attention_config.get("precision", self.precision),
                                        only_pure_attention=attention_config.get("only_pure_attention", True),
                                        force_fp32_for_softmax=middle_attention.get("force_fp32_for_softmax", False),
                                        norm_inputs=attention_config.get("norm_inputs", True),
                                        explicitly_add_residual=attention_config.get("explicitly_add_residual", True),
                                        name=f"up_{i}_attention_{j}")(x, context)
            # print("Upscaling ", i, x.shape)
            if i != len(feature_depths) - 1:
                x = Upsample(
                    features=feature_depths[-i],
                    scale=2,
                    activation=self.activation,
                    name=f"up_{i}_upsample",
                    dtype=self.dtype,
                    precision=self.precision
                )(x)

        # x = self.last_up_norm(x)
        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            precision=self.precision
        )(x)
    
        x = jnp.concatenate([x, downs.pop()], axis=-1)

        x = ResidualBlock(
            conv_type,
            name="final_residual",
            features=self.feature_depths[0],
            kernel_size=(3,3),
            strides=(1, 1),
            activation=self.activation,
            norm_groups=self.norm_groups,
            dtype=self.dtype,
            precision=self.precision,
            named_norms=self.named_norms
        )(x, temb)

        x = self.conv_out_norm(x)
        x = self.activation(x)

        noise_out = ConvLayer(
            conv_type,
            features=self.out_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            # activation=jax.nn.mish
            # kernel_init=self.kernel_init(scale=0.0),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        return noise_out #, attentions


key = jax.random.PRNGKey(0xD3)

