import jax.numpy as jnp
import numpy as np
import jax
import math
from PIL import Image
import wandb
from ml_collections import ConfigDict

from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap, jit, lax, random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from function_diffusion.utils.dps_utils import model_predict, get_posterior_mean_variance


def get_loss(u_pred, u_gt, mask=None):
    diff = u_pred - u_gt
    if mask is not None:
        diff = (u_pred - u_gt) * mask[..., None] # (H,W)
    return diff


def ddpm_sample_step(state, rng, x, t, batch_gt, ddpm_params, num_steps, zeta_obs=100.0, is_pred_x0=False):
    sigma_scalar = ddpm_params['sqrt_1m_alphas_bar'][t] # () scalar
    sigma_batch  = jnp.full((x.shape[0], 1, 1, 1), sigma_scalar) # (B,1,1,1)

    eps = 1e-5
    max_norm = 1e3
    max_grad = 1.0

    def loss_fn(x_in):
        x0, v = model_predict(state, x_in, sigma_batch, ddpm_params, is_pred_x0)
        #x0 = jnp.clip(x0, 0., 1.)
        obs = get_loss(x0, batch_gt)
        L_obs = jnp.linalg.norm(obs)
        return L_obs, (x0, v)
    
    (loss_val, (x0, v)), grad_x = jax.value_and_grad(loss_fn, has_aux=True)(x)
    g_norm = jnp.linalg.norm(grad_x)
    grad_x = grad_x * (max_norm / jnp.maximum(g_norm, max_norm))

    scale = jnp.where((num_steps - t) <= 0.8 * num_steps, zeta_obs, zeta_obs / 10.)
    x0 = x0 - scale * grad_x

    #debug.print("step {}/{}: x0 [{}, {}], g_norm: {}, error: {}", i, num_steps, x0.min(), x0.max(), g_norm, loss_val)

    post_mean, post_logvar = get_posterior_mean_variance(x, t, x0, v, ddpm_params)
    x = post_mean + jnp.exp(0.5 * post_logvar) * jax.random.normal(rng, x.shape)
    return x, x0
