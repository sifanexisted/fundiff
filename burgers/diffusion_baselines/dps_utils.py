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

from function_diffusion.utils.dps_utils import model_predict, get_posterior_mean_variance, l1_loss, l2_loss, rel2_loss, flatten


def get_data_res(u_pred, u_gt, mask=None):
    res = u_pred - u_gt
    if mask is not None:
        res = (u_pred - u_gt) * mask[..., None] # (H,W)
    return res


def get_burgers_res(u, nu=0.001, x_lo=0.0, x_hi=1.0, t_lo=0.0, t_hi=1.0):
    """
    Burgers PDE residual:
        r = u_t + u * u_x - nu * u_xx
    with Euler at boundaries and central differences inside,
    implemented using jnp.gradient.
    Keeps same shape as u.

    u: (B, nt, nx, C)
    """
    B, nt, nx, C = u.shape
    dx = (x_hi - x_lo) / (nx - 1)
    dt = (t_hi - t_lo) / (nt - 1)

    # time derivative (Euler at ends, central inside)
    u_t = jnp.gradient(u, dt, axis=1)

    # first space derivative
    u_x = jnp.gradient(u, dx, axis=2)

    # second space derivative
    u_xx = jnp.gradient(u_x, dx, axis=2)

    res = u_t + u * u_x - nu * u_xx
    return res


def create_ddpm_loss_fn(model, ddpm_params,
                        loss_type: str = 'l2',
                        is_pred_x0: bool = False,
                        use_pde_loss: bool = False,
                        pde_loss_weight: float = 1e-3,
                        get_pde_residual=None):
    """
    Create DDPM loss function with optional physics-informed PDE residual.

    Args:
        model: neural network model
        ddpm_params: dict containing diffusion parameters
        loss_type: 'l1', 'l2', or 'rel2'
        is_pred_x0: whether the model predicts x0 (True) or noise (False)
        use_pde_loss: if True, include PDE residual term
        pde_loss_weight: scalar weight for PDE loss contribution
        get_pde_residual: function handle that takes x0_est and returns PDE residual
    """

    if loss_type == 'l1':
        loss_fn = l1_loss
    elif loss_type == 'l2':
        loss_fn = l2_loss
    elif loss_type == 'rel2':
        loss_fn = rel2_loss
    else:
        raise NotImplementedError(f'Loss type {loss_type} not supported')

    def ddpm_loss(params, batch):
        x, x_t, sigma, batched_t, noise = batch
        B = x_t.shape[0]

        # Target: either original image (x) or noise
        target = noise if not is_pred_x0 else x

        # Model prediction
        pred = model.apply(params, x_t, sigma)

        # Base data loss
        data_loss = loss_fn(flatten(pred), flatten(target))
        assert data_loss.shape == (B,)

        # PDE residual loss (optional)
        pde_loss = 0.0
        if use_pde_loss and (get_pde_residual is not None):
            # Estimate x0 if predicting noise
            if not is_pred_x0:
                alpha_bar_t = ddpm_params['alphas_bar'][batched_t]
                sqrt_alpha_bar = jnp.sqrt(alpha_bar_t)[..., None, None, None]
                sigma_t = jnp.sqrt(1.0 - alpha_bar_t)[..., None, None, None]
                x0_est = (x_t - sigma_t * pred) / sqrt_alpha_bar
            else:
                x0_est = pred  # already x0

            pde_res = get_pde_residual(x0_est)   # user-defined PDE operator
            pde_loss = jnp.mean(pde_res**2)

        # Apply P2 loss weighting if available
        if 'p2_loss_weight' in ddpm_params:
            p2_loss_weight = ddpm_params['p2_loss_weight']
            data_loss = data_loss * p2_loss_weight[batched_t]

        # Total
        total_loss = data_loss.mean() + (pde_loss_weight * pde_loss.mean() if use_pde_loss else 0.0)
        return total_loss

    return ddpm_loss



def create_ddpm_train_step(model, ddpm_params, mesh, loss_type='l2',
                           is_pred_x0=False,
                           use_pde_loss=False,
                           pde_loss_weight=0.1,
                           get_pde_residual=get_burgers_res):
    """Create DDPM training step with JAX sharding"""

    ddpm_loss_fn = create_ddpm_loss_fn(
        model, ddpm_params, loss_type, is_pred_x0, use_pde_loss, pde_loss_weight, get_pde_residual=get_pde_residual
    )

    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P()),
    )
    def train_step(state, batch):
        # Compute gradients
        grad_fn = jax.value_and_grad(ddpm_loss_fn, has_aux=False)
        loss, grads = grad_fn(state.params, batch)

        # Average across devices
        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")

        # Update parameters
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


def ddpm_sample_step(state, rng, x, t, batch_gt, ddpm_params, num_steps, zeta_obs=320, zeta_pde=100, is_pred_x0=False, obs_guide=True, pde_guide=False):
    sigma_scalar = ddpm_params['sqrt_1m_alphas_bar'][t] # () scalar
    sigma_batch  = jnp.full((x.shape[0], 1, 1, 1), sigma_scalar) # (B,1,1,1)

    eps = 1e-5
    max_norm = 1e3
    max_grad = 1.0

    @jax.jit
    def loss_fn(x_in):
        x0, v = model_predict(state, x_in, sigma_batch, ddpm_params, is_pred_x0)

        obs_res = get_data_res(x0, batch_gt)
        pde_res = get_burgers_res(x0)

        obs_loss = jnp.linalg.norm(obs_res)
        pde_loss = jnp.linalg.norm(pde_res)

        return obs_loss, pde_loss, x0, v

    @jax.jit
    def obs_loss_fn(x_in):
        obs_loss, _, _, _ = loss_fn(x_in)
        return obs_loss

    @jax.jit
    def pde_loss_fn(x_in):
        _, pde_loss, _, _ = loss_fn(x_in)
        return pde_loss

    # compute grads separately
    (obs_loss, pde_loss, x0, v) = loss_fn(x)

    obs_grads = jax.grad(obs_loss_fn)(x)
    pde_grads = jax.grad(pde_loss_fn)(x)

    def clip_grad(g):
        g_norm = jnp.linalg.norm(g)
        return g * (max_norm / jnp.maximum(g_norm, max_norm))

    obs_grads = clip_grad(obs_grads)
    pde_grads = clip_grad(pde_grads)

    # guidance update
    def early_step(x0):
        # only obs guidance in early steps
        return x0 - zeta_obs * obs_grads if obs_guide else x0

    def late_step(x0):
        x_new = x0
        if obs_guide:
            x_new = x_new - (zeta_obs / 10.0) * obs_grads
        if pde_guide:
            x_new = x_new - zeta_pde * pde_grads
        return x_new

    if not obs_guide and not pde_guide:
        # unconditional: no guidance
        x0_guided = x0
    else:
        x0_guided = jax.lax.cond(t <= 0.8 * num_steps, early_step, late_step, x0)

    # posterior sampling
    post_mean, post_logvar = get_posterior_mean_variance(x, t, x0_guided, v, ddpm_params)
    noise = jax.random.normal(rng, x.shape)
    x_next = post_mean + jnp.exp(0.5 * post_logvar) * noise

    return x_next, x0_guided

