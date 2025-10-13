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

import optax
from flax.training import train_state
from flax.core import FrozenDict
from typing import Any


class TrainState(train_state.TrainState):
    ema_params: FrozenDict[str, Any]
    ema_step_size: float

    def apply_gradients(self, *, grads, **kwargs):
        next_state = super().apply_gradients(grads=grads, **kwargs)
        new_ema_params = optax.incremental_update(
            new_tensors=next_state.params,
            old_tensors=self.ema_params,
            step_size=self.ema_step_size,
        )
        return next_state.replace(ema_params=new_ema_params)


def create_train_state(config, model, tx):
    # Initialize the model if the params are not provided, otherwise use the provided params to create the state
    x = jnp.ones(config.x_dim)
    t = jnp.ones((config.x_dim[0],))

    if config.cond_diffusion:
        context = jnp.ones(config.x_dim)
    else:
        context = None

    params = model.init(random.PRNGKey(config.seed), x=x, temb=t, context=context)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              ema_params=params,
                              ema_step_size=1 - 0.9995,
                              tx=tx)
    return state



class Diffuser:
    def __init__(self, eps_fn,  pde_res_fn, diffusion_config):
        self.eps_fn = eps_fn
        self.pde_res_fn = pde_res_fn
        self.config = diffusion_config
        self.betas = jnp.asarray(self._betas(**diffusion_config))
        self.alphas = jnp.asarray(self._alphas(self.betas))
        self.alpha_bars = jnp.asarray(self._alpha_bars(self.alphas))

    @property
    def steps(self) -> int:
        return self.config.T

    def timesteps(self, steps: int):
        timesteps = jnp.linspace(0, self.steps, steps + 1)
        timesteps = jnp.rint(timesteps).astype(jnp.int32)
        return timesteps[::-1]

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, x_0, rng):
        """See algorithm 1 in https://arxiv.org/pdf/2006.11239.pdf"""
        rng1, rng2 = random.split(rng)
        t = random.randint(rng1, (len(x_0), 1), 0, self.steps)
        x_t, eps = self.sample_q(x_0, t, rng2)
        t = t.astype(x_t.dtype)
        return x_t, t, eps

    def sample_q(self, x_0, t, rng):
        """Samples x_t given x_0 by the q(x_t|x_0) formula."""
        # (bs, 1)
        alpha_t_bar = self.alpha_bars[t]
        # (bs, 1, 1, 1)
        alpha_t_bar = jnp.expand_dims(alpha_t_bar, (1, 2))

        eps = random.normal(rng, shape=x_0.shape, dtype=x_0.dtype)
        x_t = (alpha_t_bar ** 0.5) * x_0 + ((1 - alpha_t_bar) ** 0.5) * eps
        return x_t, eps

    @partial(jax.jit, static_argnums=(0,))
    def ddpm_backward_step(self, params, x_t, t, rng, context):
        """See algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alpha_bars[t]
        sigma_t = self.betas[t] ** 0.5

        z = (t > 0) * random.normal(rng, shape=x_t.shape, dtype=x_t.dtype)
        eps = self.eps_fn(params, x_t, t, context)

        x = (1 / alpha_t ** 0.5) * (
                x_t - ((1 - alpha_t) / (1 - alpha_t_bar) ** 0.5) * eps
        ) + sigma_t * z

        return x

    def ddpm_backward(self, params, x_T, rng):
        x = x_T

        for t in range(self.steps - 1, -1, -1):
            rng, rng_ = random.split(rng)
            x = self.ddpm_backward_step(params, x, t, rng_)

        return x


    @partial(jax.jit, static_argnums=(0,))
    def ddim_backward_step(
            self, params, x_t, t, t_next, context=None
    ):
        """See section 4.1 and C.1 in https://arxiv.org/pdf/2010.02502.pdf

        Note: alpha in the DDIM paper is actually alpha_bar in DDPM paper
        """
        alpha_t = self.alpha_bars[t]
        alpha_t_next = self.alpha_bars[t_next]

        eps = self.eps_fn(params, x_t, t, context=context)

        x_0 = (x_t - (1 - alpha_t) ** 0.5 * eps) / alpha_t ** 0.5
        x_t_direction = (1 - alpha_t_next) ** 0.5 * eps
        x_t_next = alpha_t_next ** 0.5 * x_0 + x_t_direction

        return x_t_next

    def ddim_backward(self, params, x_T, steps, context=None):
        x = x_T

        ts = self.timesteps(steps)
        for t, t_next in zip(ts[:-1], ts[1:]):
            x = self.ddim_backward_step(params, x, t, t_next, context=context)

        return x

    def x0_from_xt_eps(self, x_t, eps, alpha_bar_t):
        # \hat x_0 = (x_t - sqrt(1 - \bar{alpha}_t) * eps) / sqrt(\bar{alpha}_t)
        return (x_t - jnp.sqrt(1.0 - alpha_bar_t) * eps) / jnp.sqrt(alpha_bar_t)

    @partial(jax.jit, static_argnums=(0,))
    def dps_backward_step(self, params, x_t, t, obs, scale, rng, context=None):
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alpha_bars[t]
        sigma_t = self.betas[t] ** 0.5

        # Predict epsilon and form DDPM mean (Ho et al. 2020)
        eps = self.eps_fn(params, x_t, t, context)

        mean_t = (1.0 / jnp.sqrt(alpha_t)) * (
                x_t - ((1.0 - alpha_t) / jnp.sqrt(1.0 - alpha_t_bar + 1e-12)) * eps
        )

        x0_hat = self.x0_from_xt_eps(x_t, eps, alpha_t_bar)

        def loss_fn(x0):
            return jnp.mean((x0 - obs) ** 2)

        g_x0 = jax.grad(loss_fn)(x0_hat)

        g_x0 = g_x0 / (jnp.linalg.norm(g_x0) + 1e-8)
        g_xt = g_x0 / jnp.sqrt(alpha_t_bar + 1e-12)

        # DPS strength schedule; common simple choice scales with sigma_t^2
        eta_t = scale * (sigma_t ** 2)

        # Shift mean toward data consistency
        mean_t = mean_t - eta_t * g_xt

        # Sample (noise only if t > 0)
        noise = random.normal(rng, shape=x_t.shape, dtype=x_t.dtype)
        z = jnp.where(t > 0, noise, jnp.zeros_like(noise))

        x = mean_t + sigma_t * z

        return x

    def dps_backward(self, params, x_T, obs, scale, rng, context=None):
        x = x_T

        for t in range(self.steps - 1, -1, -1):
            t = jnp.array(t, dtype=jnp.int32)
            rng, rng_ = random.split(rng)
            x = self.dps_backward_step(params, x, t, obs, scale, rng_, context)

        return x

    @partial(jax.jit, static_argnums=(0,))
    def dps_backward_step_with_pde(self, params, x_t, t, obs, w_obs, w_pde, rng):
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alpha_bars[t]
        sigma_t = self.betas[t] ** 0.5

        # Predict epsilon and form DDPM mean (Ho et al. 2020)
        eps = self.eps_fn(params, x_t, t)

        mean_t = (1.0 / jnp.sqrt(alpha_t)) * (
                x_t - ((1.0 - alpha_t) / jnp.sqrt(1.0 - alpha_t_bar + 1e-12)) * eps
        )

        x0_hat = self.x0_from_xt_eps(x_t, eps, alpha_t_bar)

        def loss_obs(x0):
            return jnp.mean((x0 - obs) ** 2)

        def loss_pde(x0):
            res = self.pde_res_fn(x0)
            return jnp.mean(res ** 2)

        g_obs_x0 = jax.grad(loss_obs)(x0_hat)
        g_obs_x0 = g_obs_x0 / (jnp.linalg.norm(g_obs_x0) + 1e-8)
        g_obs_xt = g_obs_x0 / jnp.sqrt(alpha_t_bar + 1e-12)

        g_pde_x0 = jax.grad(loss_pde)(x0_hat)
        g_pde_x0 = g_pde_x0 / (jnp.linalg.norm(g_pde_x0) + 1e-8)
        g_pde_xt = g_pde_x0 / jnp.sqrt(alpha_t_bar + 1e-12)

        # DPS strength schedule; common simple choice scales with sigma_t^2
        # Shift mean toward data consistency
        mean_t = mean_t - sigma_t ** 2 * (w_obs * g_obs_xt + w_pde * g_pde_xt)

        # Sample (noise only if t > 0)
        noise = random.normal(rng, shape=x_t.shape, dtype=x_t.dtype)
        z = jnp.where(t > 0, noise, jnp.zeros_like(noise))

        x = mean_t + sigma_t * z

        return x

    def dps_backward_with_pde(self, params, x_T, obs, w_obs, w_pde, rng):
        x = x_T

        for t in range(self.steps - 1, -1, -1):
            t = jnp.array(t, dtype=jnp.int32)
            rng, rng_ = random.split(rng)

            if t < 0.8 * self.steps:
                x = self.dps_backward_step_with_pde(params, x, t, obs, w_obs, 0.0, rng_)
            if t >= 0.8 * self.steps:
                x = self.dps_backward_step_with_pde(params, x, t, obs, w_obs / 10, w_pde, rng_)

        return x
    @classmethod
    def _linear_schedule(cls, T: int, beta_1=1e-4, beta_T=0.02):
        '''original ddpm paper'''
        return jnp.linspace(beta_1, beta_T, T, dtype=jnp.float32)
    
    @classmethod
    def _cosine_schedule(cls, T: int, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = T + 1
        t = jnp.linspace(0, T, steps, dtype=jnp.float32) / T
        alpha_bars = jnp.cos((t + s) / (1.0 + s) * jnp.pi / 2) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        return jnp.clip(betas, 0, 0.999)
    
    @classmethod
    def _sigmoid_beta_schedule(cls, T: int, start=-3, end=3, tau=1.0, clamp_min=1e-5):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = T + 1
        t = jnp.linspace(0, T, steps, dtype=jnp.float32) / T
        v_start = jnp.sigmoid(start / tau)
        v_end = jnp.sigmoid(end / tau)
        s_t = jnp.sigmoid(((t * (end - start) + start) / tau))
        alpha_bars = (-(s_t) + v_end) / (v_end - v_start)
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
        return jnp.clip(betas, clamp_min, 0.999)

    @classmethod
    def _betas(cls, T, schedule='linear', **kwargs):
        if schedule == 'linear':
            return cls._linear_schedule(T, **kwargs)
        elif schedule == 'cosine':
            return cls._cosine_schedule(T, **kwargs)
        elif schedule == 'sigmoid':
            return cls._sigmoid_beta_schedule(T, **kwargs)
        else:
            raise ValueError(f"Unknown schedule: {schedule}. Choose from 'linear', 'cosine', or 'sigmoid'.")

    @classmethod
    def _alphas(cls, betas):
        return 1 - betas

    @classmethod
    def _alpha_bars(cls, alphas):
        return jnp.cumprod(alphas)

    @staticmethod
    def expand_t(t, x):
        return jnp.full((len(x), 1), t, dtype=x.dtype)


def create_step_fn(dfiffuser, mesh):

    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch"), P()),
        out_specs=(P(), P(), P()),
        check_rep=False,
    )
    def train_step(state, batch, rng):
        def loss_fn(params, batch, rng):
            x_0, context = batch
            rng, step_rng = random.split(rng)
            x_t, t, eps = dfiffuser.forward(x_0, step_rng)
            eps_pred = dfiffuser.eps_fn(params, x_t, t, context=context)
            loss = jnp.mean((eps - eps_pred) ** 2)
            return loss, rng

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, rng), grads = grad_fn(state.params, batch, rng)
        grads = jax.lax.pmean(grads, axis_name="batch")
        state = state.apply_gradients(grads=grads)
        return state, loss, rng

    return train_step


# def create_step_fn(dfiffuser, mesh, config):
#
#     @jax.jit
#     @partial(
#         shard_map,
#         mesh=mesh,
#         in_specs=(P(), P("batch"), P()),
#         out_specs=(P(), P(), P()),
#         check_rep=False,
#     )
#     def train_step(state, batch, rng):
#         def loss_fn(params, batch, rng):
#             x_0, context = batch
#             rng, step_rng = random.split(rng)
#             x_t, t, eps = dfiffuser.forward(x_0, step_rng)
#             eps_pred = dfiffuser.eps_fn(params, x_t, t, context=context)
#             eps_loss = jnp.mean((eps - eps_pred) ** 2)
#             loss = eps_loss
#
#             aux = (rng,)
#
#             if config.use_pde_loss:
#                 x_0_est = dfiffuser.x0_from_xt_eps(x_t, eps, dfiffuser.alpha_bars[jnp.int32(t)].flatten())
#                 pde_res = dfiffuser.pde_res_fn(x_0_est)
#                 pde_loss = jnp.mean(pde_res ** 2)
#
#                 loss = eps_loss + config.pde_loss_weight * pde_loss
#                 aux = (eps_loss, pde_loss, rng)
#
#             return loss, aux
#
#         grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#         (loss, aux), grads = grad_fn(state.params, batch, rng)
#         grads = jax.lax.pmean(grads, axis_name="batch")
#         state = state.apply_gradients(grads=grads)
#         return state, loss, aux
#
#     return train_step