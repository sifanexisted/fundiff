import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from flax import linen as nn
from jax import lax, jit, grad, random, debug
import jax.nn as jnn
from jax.nn import silu
from jax.nn.initializers import uniform, normal, xavier_uniform
import math, functools, dataclasses, itertools
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Union, Tuple, Any
from einops import rearrange, repeat
import ml_collections


def weight_init(key, shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':
        lim = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, shape, minval=-lim, maxval=lim)
    if mode == 'xavier_normal':
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        return std * jax.random.normal(key, shape)
    if mode == 'kaiming_uniform':
        lim = jnp.sqrt(3.0 / fan_in)
        return jax.random.uniform(key, shape, minval=-lim, maxval=lim)
    if mode == 'kaiming_normal':
        std = jnp.sqrt(1.0 / fan_in)
        return std * jax.random.normal(key, shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(nn.Module):
    in_features:  int
    out_features: int
    bias:         bool   = True
    init_mode:    str    = 'kaiming_normal'
    init_weight:  float  = 1.0
    init_bias:    float  = 0.0
    @nn.compact
    def __call__(self, x):
        W = self.param(
            'weight',
            lambda k, s: weight_init(
                k, s, self.init_mode, self.in_features, self.out_features
            ) * self.init_weight,
            (self.out_features, self.in_features),
        ).astype(x.dtype)

        y = jnp.matmul(x, W.T)
        if self.bias:
            b = self.param(
                'bias',
                lambda k, s: weight_init(
                    k, s, self.init_mode, self.in_features, self.out_features
                ) * self.init_bias,
                (self.out_features,),
            ).astype(x.dtype)
            y = y + b
        return y

def _conv2d(x, w, stride=1, padding=0, groups=1):
    return jax.lax.conv_general_dilated(
        x,
        w,
        (stride, stride),
        ((padding, padding), (padding, padding)),
        feature_group_count=groups,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )

def _conv_transpose2d(x, w, stride=2, padding=0, groups=1):
    k_h, k_w = w.shape[2], w.shape[3]
    pad_h    = k_h - 1 - padding
    pad_w    = k_w - 1 - padding

    return jax.lax.conv_general_dilated(
        lhs=x,
        rhs=w,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        lhs_dilation=(stride, stride),
        feature_group_count=groups,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )

class Conv2d(nn.Module):
    in_channels: int
    out_channels: int
    kernel: int
    bias: bool = True
    up: bool = False
    down: bool = False
    resample_filter: tuple = (1, 1)
    fused_resample: bool = False
    init_mode: str = "kaiming_normal"
    init_weight: float = 1.0
    init_bias: float = 0.0

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        w = None
        b = None
        if self.kernel:
            w_shape = (self.out_channels,
                       self.in_channels,
                       self.kernel,
                       self.kernel)
            fan_in  = self.in_channels  * self.kernel**2
            fan_out = self.out_channels * self.kernel**2
            w = self.param(
                "weight",
                lambda k, s: weight_init(k, s,
                                         self.init_mode,
                                         fan_in, fan_out) * self.init_weight,
                w_shape,
            ).astype(dtype)

            if self.bias:
                b = self.param(
                    "bias",
                    lambda k, s: weight_init(k, s,
                                             self.init_mode,
                                             fan_in, fan_out) * self.init_bias,
                    (self.out_channels,),
                ).astype(dtype)
        f = None
        if self.up or self.down:
            f1 = jnp.asarray(self.resample_filter, dtype=dtype)
            f = jnp.outer(f1, f1)
            f = f[None, None, :, :] / (f1.sum() ** 2)      # [1,1,H,W]

        w_pad = self.kernel // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and (w is not None):
            filt = jnp.tile(f * 4.0, (self.in_channels, 1, 1, 1))
            x = _conv_transpose2d(
                x, filt,
                stride=2,
                padding=max(f_pad - w_pad, 0),
                groups=self.in_channels
            )
            x = _conv2d(
                x, w,
                padding=max(w_pad - f_pad, 0)
            )

        elif self.fused_resample and self.down and (w is not None):
            x = _conv2d(x, w, padding=w_pad + f_pad)
            filt = jnp.tile(f, (self.out_channels, 1, 1, 1))
            x = _conv2d(
                x, filt,
                stride=2,
                padding=f_pad,
                groups=self.out_channels
            )

        else:
            if self.up:
                filt = jnp.tile(f * 4.0, (self.in_channels, 1, 1, 1))
                x = _conv_transpose2d(
                    x, filt,
                    stride=2,
                    padding=f_pad,
                    groups=self.in_channels
                )
            if self.down:
                filt = jnp.tile(f, (self.in_channels, 1, 1, 1))
                x = _conv2d(
                    x, filt,
                    stride=2,
                    padding=f_pad,
                    groups=self.in_channels
                )
            if w is not None:
                x = _conv2d(x, w, padding=w_pad)

        if b is not None:
            x = x + b.reshape((1, -1, 1, 1)).astype(dtype)
        return x

class GroupNorm(nn.Module):
    num_channels: int
    num_groups: int = 32
    min_channels_per_group: int = 4
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        C = self.num_channels
        G = min(self.num_groups, C // self.min_channels_per_group)

        gamma = self.param(
            "weight",
            lambda k, s: jnp.ones(s, dtype),
            (C,),
        ).astype(dtype)
        beta  = self.param(
            "bias",
            lambda k, s: jnp.zeros(s, dtype),
            (C,),
        ).astype(dtype)

        # reshape => [N, G, C//G, ...]
        N = x.shape[0]
        new_shape = (N, G, C // G) + x.shape[2:]
        x_grouped = x.reshape(new_shape)

        reduce_axes = tuple(range(2, x_grouped.ndim))
        mean = x_grouped.mean(axis=reduce_axes, keepdims=True)
        var = ((x_grouped - mean) ** 2).mean(axis=reduce_axes, keepdims=True)

        x_norm = (x_grouped - mean) / jnp.sqrt(var + self.eps)
        x_norm = x_norm.reshape(x.shape)     # back

        # affine
        bcast_shape = (1, C) + (1,) * (x.ndim - 2)   # (1,C,1,1): NCHW / NCF...
        y = x_norm * gamma.reshape(bcast_shape) + beta.reshape(bcast_shape)
        return y



def hw_to_seq(x: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[int,int]]:
    # x: (B, C, H, W) → (B, HW, C)
    b, c, h, w = x.shape
    x = jnp.transpose(x, (0, 2, 3, 1)).reshape(b, h*w, c)
    return x, (h, w)

def seq_to_hw(x: jnp.ndarray, hw: Tuple[int,int]) -> jnp.ndarray:
    # x: (B, HW, C) → (B, C, H, W)
    b, hw_len, c = x.shape
    h, w = hw
    x = x.reshape(b, h, w, c)
    return jnp.transpose(x, (0, 3, 1, 2))


class SelfAttn2D(nn.Module):
    channels: int
    num_heads: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        # x: (B, C, H, W)
        h = nn.GroupNorm(num_groups=32, epsilon=1e-6)(x)
        h, hw = hw_to_seq(h)  # (B, HW, C)

        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(h, h)  # (B, HW, C)

        h = seq_to_hw(h, hw)
        h = nn.Dense(self.channels)(jnp.swapaxes(h, 1, 1))  # no-op, keeps params consistent
        return x + h  # residual


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


class CrossAttn2D(nn.Module):
    channels: int
    num_heads: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, context: jnp.ndarray, *, deterministic: bool):
        # x: (B, C, H, W), context: (B, Lc, Cc)
        h = nn.GroupNorm(num_groups=32, epsilon=1e-6)(x)
        q, hw = hw_to_seq(h)                               # (B, HW, C)
        kv = nn.LayerNorm()(context)                       # (B, Lc, Cc)
        kv = nn.Dense(q.shape[-1], name="context_proj")(kv)  # project to C

        q = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(q, kv)  # (B, HW, C)

        q = seq_to_hw(q, hw)
        q = nn.Dense(self.channels, name="out")(q)         # out proj
        return x + q


class UNetBlock(nn.Module):
    in_channels: int
    out_channels: int
    emb_channels: int

    up: bool = False
    down: bool = False
    attention: bool = False
    num_heads:  Union[int, None] = None
    channels_per_head: int = 64

    dropout: float = 0.0
    skip_scale: float = 1.0
    eps: float = 1e-5

    resample_filter: tuple[int, ...] = (1, 1)
    resample_proj: bool = False
    adaptive_scale: bool = True

    param_init: dict = dataclasses.field(default_factory=dict)
    param_init_zero: dict = dataclasses.field(default_factory=lambda: {"init_weight": 0})#
    param_init_attn: Union[dict, None] = None

    @nn.compact
    def __call__(self, x, emb, context, *, deterministic: bool = False):
        needs_skip = (self.out_channels != self.in_channels) or self.up or self.down

        orig = x

        x = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel=3,
            up=self.up,
            down=self.down,
            resample_filter=self.resample_filter,
            **self.param_init,
        )(silu(GroupNorm(num_channels=self.in_channels, eps=self.eps)(x)))

        scale_shift = Linear(
            in_features=self.emb_channels,
            out_features=self.out_channels * (2 if self.adaptive_scale else 1),
            **self.param_init,
        )(emb)[:, :, None, None].astype(x.dtype)    # (N, C*{1|2}, 1, 1)

        if self.adaptive_scale:
            scale, shift = jnp.split(scale_shift, 2, axis=1)  # (N,C,1,1) ×2
            x = silu(shift + GroupNorm(num_channels=self.out_channels, eps=self.eps)(x) * (scale + 1))
        else:
            x = silu(GroupNorm(num_channels=self.out_channels, eps=self.eps)(x + scale_shift))

        x = Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel=3,
            **self.param_init_zero,
        )(nn.Dropout(rate=self.dropout)(x, deterministic=deterministic))

        if needs_skip:
            skip_out = Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel=(1 if (self.resample_proj or self.out_channels != self.in_channels) else 0),
                up=self.up,
                down=self.down,
                resample_filter=self.resample_filter,
                **self.param_init,
            )(orig)
        else:
            skip_out = orig

        x = (x + skip_out) * self.skip_scale
        B, C, H, W = x.shape

        if self.attention:
            x = SelfAttn2D(channels=self.out_channels, num_heads=self.num_heads, dropout=self.dropout)(x,
                                                                                                       deterministic=deterministic)

        if self.cross_attention and (context is not None):
            x = CrossAttn2D(channels=self.out_channels, num_heads=self.num_heads, dropout=self.dropout)(x, context,
                                                                                                        deterministic=deterministic)

        x = (x + Conv2d(
            in_channels=C,
            out_channels=C,
            kernel=1,
            **self.param_init_zero,
        )(x)) * self.skip_scale

        return x

class PositionalEmbedding(nn.Module):
    num_channels: int
    max_positions: int = 10000
    endpoint: bool = False

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        half  = self.num_channels // 2
        freqs = jnp.arange(half, dtype=jnp.float32)
        denom = half - (1 if self.endpoint else 0)
        freqs = freqs / denom
        freqs = (1.0 / self.max_positions) ** freqs # (half,)

        # [..., L, 1] * [1, half] → [..., L, half]
        angles = x.astype(jnp.float32)[..., None] * freqs
        angles = angles.astype(dtype)

        emb = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        return emb

class FourierEmbedding(nn.Module):
    num_channels: int
    scale: float = 16.0

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        half = self.num_channels // 2

        # freqs ~ N(0, scale^2)
        freqs = self.param(
            "freqs",
            lambda key, shape: self.scale * jax.random.normal(key, shape),
            (half,),
        ).astype(dtype) # (half,)

        # [..., L, 1] * [1, half] → [..., L, half]
        angles = x[..., None] * (2.0 * jnp.pi * freqs) # 
        emb = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        return emb

class SongUNet(nn.Module):
    # hyper
    img_resolution: int
    in_channels: int
    out_channels: int
    label_dim: int = 0
    augment_dim:int = 0

    model_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 2, 2)
    channel_mult_emb: int = 4
    num_blocks: int = 4
    attn_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.0
    label_dropout: float = 0.0

    embedding_type: str = "positional" # "positional" | "fourier"
    channel_mult_noise: int = 1
    encoder_type: str = "standard" # "standard" | "skip" | "residual"
    decoder_type: str = "standard" # "standard" | "skip"
    resample_filter: Tuple[int, ...] = (1, 1)

    def setup(self):
        assert self.embedding_type in ("positional", "fourier")
        assert self.encoder_type in ("standard", "skip", "residual")
        assert self.decoder_type in ("standard", "skip")

        self.emb_channels = self.model_channels * self.channel_mult_emb
        self.noise_channels = self.model_channels * self.channel_mult_noise

        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=math.sqrt(0.2))
        block_kw = dict(
            emb_channels = self.emb_channels,
            num_heads = 8,
            dropout = self.dropout,
            skip_scale = math.sqrt(0.5),
            eps = 1e-6,
            resample_filter = self.resample_filter,
            resample_proj = True,
            adaptive_scale = False,
            param_init = init,
            param_init_zero = init_zero,
            param_init_attn = init_attn,
        )

        self.map_noise = (PositionalEmbedding(num_channels=self.noise_channels, endpoint=True)
                            if self.embedding_type == "positional"
                            else FourierEmbedding   (num_channels=self.noise_channels))
        self.map_label = (Linear(self.label_dim, self.noise_channels, **init)
                            if self.label_dim else None)
        self.map_augment = (Linear(self.augment_dim, self.noise_channels, bias=False, **init)
                            if self.augment_dim else None)

        self.map_layer0 = Linear(self.noise_channels, self.emb_channels, **init)
        self.map_layer1 = Linear(self.emb_channels,   self.emb_channels, **init)

        # Encoder 
        enc = {}
        cout, caux = self.in_channels, self.in_channels

        for level, mult in enumerate(self.channel_mult):
            res = self.img_resolution >> level
            if level == 0:
                cin, cout = cout, self.model_channels
                enc[f"{res}x{res}_conv"] = Conv2d(cin, cout, kernel=3, **init)
            else:
                enc[f"{res}x{res}_down"] = UNetBlock(cout, cout, down=True, **block_kw)
                if self.encoder_type == "skip":
                    enc[f"{res}x{res}_aux_down"] = Conv2d(
                        caux, caux, kernel=0, down=True, resample_filter=self.resample_filter
                    )
                    enc[f"{res}x{res}_aux_skip"] = Conv2d(caux, cout, kernel=1, **init)
                if self.encoder_type == "residual":
                    enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        caux, cout, kernel=3, down=True,
                        resample_filter=self.resample_filter,
                        fused_resample=True, **init
                    )
                    caux = cout

            for idx in range(self.num_blocks):
                cin, cout = cout, self.model_channels * mult
                attn = res in self.attn_resolutions
                enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    cin, cout, attention=attn, **block_kw
                )

        self.enc = enc
        self.skip_channels = [
            blk.out_channels for n, blk in self.enc.items() if "aux" not in n
        ]
        skip_ch = [blk.out_channels for n, blk in enc.items() if "aux" not in n]  # list

        # Decoder
        dec = {}
        cout = skip_ch[-1]

        build_ch = skip_ch.copy()

        for level, mult in reversed(list(enumerate(self.channel_mult))):
            res = self.img_resolution >> level
            if level == len(self.channel_mult) - 1:
                dec[f"{res}x{res}_in0"] = UNetBlock(cout, cout, attention=True, **block_kw)
                dec[f"{res}x{res}_in1"] = UNetBlock(cout, cout, **block_kw)
            else:
                dec[f"{res}x{res}_up"]  = UNetBlock(cout, cout, up=True, **block_kw)

            for idx in range(self.num_blocks + 1):
                cin  = cout + build_ch.pop()        # temporal
                cout = self.model_channels * mult
                attn = (idx == self.num_blocks) and (res in self.attn_resolutions)
                dec[f"{res}x{res}_block{idx}"] = UNetBlock(cin, cout, attention=attn, **block_kw)

            if self.decoder_type == "skip" or level == 0:
                if self.decoder_type == "skip" and level < len(self.channel_mult) - 1:
                    dec[f"{res}x{res}_aux_up"] = Conv2d(
                        self.out_channels, self.out_channels,
                        kernel=0, up=True, resample_filter=self.resample_filter
                    )
                dec[f"{res}x{res}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    cout, self.out_channels, kernel=3, **init_zero
                )

        self.dec = dec 

    def __call__(self,
                 x: jnp.ndarray,  # [N, Cin, H, W]
                 noise_labels: jnp.ndarray,  # [N] or [N,]
                 class_labels: jnp.ndarray, # [N, label_dim]
                 augment_labels: jnp.ndarray | None = None,
                 context : jnp.ndarray | None = None,
                 *,
                 deterministic: bool = False):

        emb = self.map_noise(noise_labels) # (N, noise_channels)
        emb = emb.reshape(emb.shape[0], 2, -1)[:, ::-1, :].reshape(emb.shape)  # swap sin/cos

        # classifier-free guidance label-dropout
        if (self.map_label is not None) and (class_labels is not None):
            tmp = class_labels
            if (self.label_dropout > 0) and (not deterministic):
                drop_key = self.make_rng("label_drop")
                mask = jax.random.bernoulli(drop_key, 1.0 - self.label_dropout,
                                                (x.shape[0], 1))
                tmp = tmp * mask.astype(tmp.dtype)
            emb = emb + self.map_label(tmp * math.sqrt(self.label_dim))

        if (self.map_augment is not None) and (augment_labels is not None):
            emb = emb + self.map_augment(augment_labels)

        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder
        skips: List[jnp.ndarray] = []
        aux = x

        for name, blk in self.enc.items():
            if "aux_down" in name:
                aux = blk(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + blk(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + blk(aux)) / math.sqrt(2.0)
            else:
                # 判断是否带 emb 的 UNetBlock
                if isinstance(blk, UNetBlock):
                    x = blk(x, emb, deterministic=deterministic)
                else:
                    x = blk(x)
                skips.append(x)

        # Decoder
        skip_runtime = list(self.skip_channels)   #new

        aux_out = None
        tmp_aux = None
        for name, blk in self.dec.items():
            if "aux_up" in name:
                aux_out = blk(aux_out)
            elif "aux_norm" in name:
                tmp_aux = blk(x)
            elif "aux_conv" in name:
                tmp_aux = blk(silu(tmp_aux))
                aux_out = tmp_aux if aux_out is None else aux_out + tmp_aux
            else:  # UNetBlock
                if x.shape[1] != blk.in_channels:
                    #print("x if :", x.shape)
                    #x = jnp.concatenate([x, skip_runtime.pop()], axis=1)
                    x = jnp.concatenate([x, skips.pop()], axis=1)
                #else:
                    #print("x else :", x.shape)
                x = blk(x, emb, deterministic=deterministic)

        return aux_out.astype(x.dtype)   #



key = jax.random.PRNGKey(0xD3)
class VEPrecond(nn.Module): # For both VE and DDPM
    # 
    img_resolution: int
    img_channels:   int
    label_dim:      int = 0

    # 
    use_fp16:    bool   = False
    sigma_min:   float  = 2e-2
    sigma_max:   float  = 100.0

    # SongUNet
    model_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    model_name: Optional[str] = None

    def setup(self):
        self.model = SongUNet(
            img_resolution = self.img_resolution,
            in_channels    = self.img_channels,
            out_channels   = self.img_channels,
            label_dim      = self.label_dim,
            **self.model_kwargs,
        )

    def __call__(self,
                 x: jnp.ndarray,  # [N, C, H, W]
                 sigma: jnp.ndarray, # [N] or scalar
                 class_labels: Optional[jnp.ndarray] = None,
                 context : Optional[jnp.ndarray] = None,
                 *,
                 force_fp32: bool = False,
                 deterministic: bool = False,
                 **apply_kwargs) -> jnp.ndarray:

        if context is not None:
            context = PatchEmbed(emb_dim=128, patch_size=(16, 16))(context)
        
        x = rearrange(x, "n h w c -> n c h w")

        x = x.astype(jnp.float32)
        sigma = sigma.astype(jnp.float32).reshape((-1, 1, 1, 1))  # (N,1,1,1)

        if self.label_dim == 0:
            class_labels_in = None
        else:
            if class_labels is None:
                class_labels_in = jnp.zeros((x.shape[0], self.label_dim), dtype=jnp.float32)
            else:
                class_labels_in = class_labels.astype(jnp.float32).reshape((-1, self.label_dim))

        dtype = jnp.float16 if (self.use_fp16 and (not force_fp32)) else jnp.float32

        # ---------- condition ----------
        c_skip = 1.0
        c_out = sigma
        c_in = 1.0
        c_noise = jnp.log(0.5 * sigma[..., 0, 0, 0]) # (N,)

        # SongUNet
        F_x = self.model(
            (c_in * x).astype(dtype),
            c_noise, # noise_labels
            class_labels_in, # class_labels
            context = context,
            augment_labels=None,
            deterministic=deterministic,
            **apply_kwargs,
        )
        assert F_x.dtype == dtype

        D_x = c_skip * x + c_out * F_x.astype(jnp.float32)

        D_x = rearrange(D_x, "n c h w -> n h w c")
        return D_x

    def round_sigma(self, sigma):
        return jnp.asarray(sigma, dtype=jnp.float32)

class EDMPrecond(nn.Module): # For EDM
    img_resolution: int
    img_channels:   int
    label_dim:      int = 0

    use_fp16:    bool   = False
    sigma_min:   float  = 0
    sigma_max:   float  = float('inf')
    sigma_data:  float  = 0.5
    eps: float = 1e-6

    model_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    model_name: Optional[str] = None

    def setup(self):
        self.model = SongUNet(
            img_resolution = self.img_resolution,
            in_channels    = self.img_channels,
            out_channels   = self.img_channels,
            label_dim      = self.label_dim,
            **self.model_kwargs,
        )


    def __call__(self,
                 x: jnp.ndarray,  # [N, C, H, W]
                 sigma: jnp.ndarray, # [N] or scalar
                 class_labels: Optional[jnp.ndarray] = None,
                 *,
                 force_fp32: bool = False,
                 deterministic: bool = False,
                 **apply_kwargs) -> jnp.ndarray:
        
        x = rearrange(x, "n h w c -> n c h w")
        x = x.astype(jnp.float32)
        sigma = sigma.astype(jnp.float32).reshape((-1, 1, 1, 1))  # (N,1,1,1)

        if self.label_dim == 0:
            class_labels_in = None
        else:
            if class_labels is None:
                class_labels_in = jnp.zeros((x.shape[0], self.label_dim), dtype=jnp.float32)
            else:
                class_labels_in = class_labels.astype(jnp.float32).reshape((-1, self.label_dim))

        dtype = jnp.float16 if (self.use_fp16 and (not force_fp32)) else jnp.float32

        # ---------- condition ----------
        safe_one = jnp.maximum((sigma ** 2 + self.sigma_data ** 2), self.eps)
        c_skip = self.sigma_data ** 2 / safe_one
        c_out = jnp.sqrt(sigma * self.sigma_data / safe_one)
        safe_one = jnp.maximum((self.sigma_data ** 2 + sigma ** 2), self.eps)
        c_in = jnp.sqrt(1 / (self.sigma_data ** 2 + sigma ** 2))
        c_noise = jnp.log(sigma[..., 0, 0, 0]) / 4 # (N,)
        
        F_x = self.model(
            (c_in * x).astype(dtype),
            c_noise, 
            class_labels_in,
            augment_labels=None,
            deterministic=deterministic,
            **apply_kwargs,
        )
        assert F_x.dtype == dtype

        D_x = c_skip * x + c_out * F_x.astype(jnp.float32)
        #debug.print("D_x [{}, {}]", D_x.min(), D_x.max()) 

        D_x = rearrange(D_x, "n c h w -> n h w c")
        return D_x

    def round_sigma(self, sigma):
        return jnp.asarray(sigma, dtype=jnp.float32)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = EDMPrecond(
        img_resolution = 128,
        img_channels   = 3,
        model_kwargs   = dict(embedding_type="positional", model_channels=64),
    )

    x      = jnp.ones((5, 128, 128, 3))
    sigma  = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    params = model.init({'params': key, 'label_drop': key},
                    x, sigma, deterministic=True)

    def count_params(tree):
        return sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(tree))

    n_params = count_params(params['params'])
    print(f'Total parameters: {n_params:,}')        

    # forward
    y = model.apply(params, x, sigma, deterministic=True)
    print(y.shape)   # (4, 3, 32, 32)

    #model = Conv2d(in_channels=3, out_channels=5, kernel=0, up=True)

    #x = jnp.ones((7,3, 16, 16))
    #params = model.init(key, x)
    #y = model.apply(params, x)
    #print(y.shape)
