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


def cosine_beta_schedule(timesteps):
    """Return cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672 """
    s = 0.008
    max_beta = 0.999
    ts = jnp.linspace(0, 1, timesteps + 1)
    alphas_bar = jnp.cos((ts + s) / (1 + s) * jnp.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return (jnp.clip(betas, 0, max_beta))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = jnp.linspace(
        beta_start, beta_end, timesteps, dtype=jnp.float64)
    return (betas)


def get_ddpm_params(config):
    schedule_name = config.beta_schedule
    timesteps = config.timesteps
    p2_loss_weight_gamma = config.p2_loss_weight_gamma
    p2_loss_weight_k = config.p2_loss_weight_k

    if schedule_name == 'linear':
        betas = linear_beta_schedule(timesteps)
    elif schedule_name == 'cosine':
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f'unknown beta schedule {schedule_name}')
    assert betas.shape == (timesteps,)
    alphas = 1. - betas
    alphas_bar = jnp.cumprod(alphas, axis=0)
    sqrt_alphas_bar = jnp.sqrt(alphas_bar)
    sqrt_1m_alphas_bar = jnp.sqrt(1. - alphas_bar)

    # calculate p2 reweighting
    p2_loss_weight = (p2_loss_weight_k + alphas_bar / (1 - alphas_bar)) ** -p2_loss_weight_gamma

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_bar': alphas_bar,
        'sqrt_alphas_bar': sqrt_alphas_bar,
        'sqrt_1m_alphas_bar': sqrt_1m_alphas_bar,
        'p2_loss_weight': p2_loss_weight
    }


def make_grid(samples, n_samples, padding=2, pad_value=0.0):
    ndarray = samples.reshape((-1, *samples.shape[2:]))[:n_samples]
    nrow = int(np.sqrt(ndarray.shape[0]))

    if not (isinstance(ndarray, jnp.ndarray) or
            (isinstance(ndarray, list) and
             all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError("array_like of tensors expected, got {}".format(
            type(ndarray)))

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                         padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full(
        (height * ymaps + padding, width * xmaps + padding, num_channels),
        pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[y * height + padding:(y + 1) * height,
                   x * width + padding:(x + 1) * width].set(ndarray[k])
            k = k + 1
    return grid

def flatten(x):
    """Flatten tensor for loss computation"""
    return x.reshape(x.shape[0], -1)


def l2_loss(logit, target):
    """L2 loss function"""
    loss = (logit - target) ** 2
    loss = jnp.mean(loss, axis=1)  # Mean over flattened dimensions
    return loss

def l1_loss(logit, target):
    """L1 loss function"""
    loss = jnp.abs(logit - target)
    loss = jnp.mean(loss, axis=1)  # Mean over flattened dimensions
    return loss

def rel2_loss(logit, target):
    """relative L2 loss function"""
    return jnp.linalg.norm(logit - target, axis=1) / (jnp.linalg.norm(target, axis=1) + 1e-8)


def q_sample_jitted(x, t, noise, sqrt_alphas_bar, sqrt_1m_alphas_bar):
    sqrt_alpha_bar = sqrt_alphas_bar[t, None, None, None]
    sqrt_1m_alpha_bar = sqrt_1m_alphas_bar[t, None, None, None]
    #print(f"+++++++++++{x.shape}, {sqrt_alpha_bar.shape}+++++++++++++")
    x_t = sqrt_alpha_bar * x + sqrt_1m_alpha_bar * noise
    return x_t

def sample_ve_batch(key, x0, sigma_min=0.02, sigma_max=100.):
    """
    x0: (B,C,H,W) clean images in [-1,1] or [0,1]
    returns (x_noisy, sigma, noise)
    """
    B = x0.shape[0]
    keys = random.split(key, 2)

    u = random.uniform(keys[0], (B, 1, 1, 1))
    ratio = sigma_max / sigma_min
    sigma = sigma_min * (ratio ** u)

    eps = random.normal(keys[1], x0.shape)
    x_sigma = x0 + sigma * eps

    weight = 1.0 / (sigma ** 2)
    return (x_sigma, sigma, weight, eps)

def sample_edm_batch(key, x0, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    """
    x0: (B,C,H,W) clean images in [-1,1] or [0,1]
    returns (x_noisy, sigma, noise)
    """
    B = x0.shape[0]
    keys = random.split(key, 2)

    u = random.uniform(keys[0], (B, 1, 1, 1))
    sigma = jnp.exp(u * P_std + P_mean)

    eps = random.normal(keys[1], x0.shape)
    x_sigma = x0 + sigma * eps

    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    return (x_sigma, sigma, weight, eps)


@jit
def get_ddmp_batch_jitted(key, x, betas, sqrt_alphas_bar, sqrt_1m_alphas_bar):
    """Generate DDPM training batch with random timesteps and noise"""
    B = x.shape[0]
    timesteps = len(betas)

    keys = random.split(key, 3)

    # Sample random timesteps for each batch element
    batched_t = random.randint(
        keys[0],
        shape=(B,),
        dtype=jnp.int32,
        minval=0,
        maxval=timesteps
    )

    # Sample noise
    noise = random.normal(keys[1], x.shape)

    # Generate noisy images x_t
    x_t = q_sample_jitted(x, batched_t, noise, sqrt_alphas_bar, sqrt_1m_alphas_bar)

    sigma = sqrt_1m_alphas_bar[batched_t]

    batch = (x, x_t, sigma, batched_t, noise)
    return batch, keys[2]

@jit
def get_ve_batch_jitted(key, x, sigma_min, sigma_max):
    keys = random.split(key, 3)
    x_sigma, sigma, weight, eps = sample_ve_batch(keys[0], x, sigma_min, sigma_max)
    return (x, x_sigma, sigma, weight, eps), keys[2]

@jit
def get_edm_batch_jitted(key, x, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    keys = random.split(key, 3)
    x_sigma, sigma, weight, eps = sample_edm_batch(keys[0], x, P_mean=-1.2, P_std=1.2, sigma_data=0.5)
    return (x, x_sigma, sigma, weight, eps), keys[2]

def create_get_ddpm_batch_fn(ddpm_params):
    """Create a JIT-compiled DDPM batch function with captured parameters"""
    betas = ddpm_params['betas']
    sqrt_alphas_bar = ddpm_params['sqrt_alphas_bar']
    sqrt_1m_alphas_bar = ddpm_params['sqrt_1m_alphas_bar']

    def get_ddmp_batch(key, x):
        return get_ddmp_batch_jitted(key, x, betas, sqrt_alphas_bar, sqrt_1m_alphas_bar)

    return get_ddmp_batch

def create_get_ve_batch_fn(ve_params):
    """Create a JIT-compiled VE batch function with captured parameters"""
    sigma_min = ve_params.sigma_min
    sigma_max = ve_params.sigma_max

    def get_ve_batch(key, x):
        return get_ve_batch_jitted(key, x, sigma_min, sigma_max)

    return get_ve_batch

def create_get_edm_batch_fn(edm_params):
    """Create a JIT-compiled EDM batch function with captured parameters"""
    P_mean = edm_params.P_mean
    P_std = edm_params.P_std
    sigma_data = edm_params.sigma_data

    def get_edm_batch(key, x):
        return get_edm_batch_jitted(key, x, P_mean, P_std, sigma_data)

    return get_edm_batch

def create_ddpm_loss_fn(model, ddpm_params, loss_type='l2', is_pred_x0=False):
    """Create DDPM loss function"""

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

        # Target is either original image (if is_pred_x0) or noise
        target = noise if not is_pred_x0 else x

        # Model prediction
        pred = model.apply(params, x_t, sigma)

        # Compute loss
        loss = loss_fn(flatten(pred), flatten(target))
        assert loss.shape == (B,)

        # Apply P2 loss weighting if available
        if 'p2_loss_weight' in ddpm_params:
            p2_loss_weight = ddpm_params['p2_loss_weight']
            loss = loss * p2_loss_weight[batched_t]

        return loss.mean()

    return ddpm_loss

def create_ve_loss_fn(model, ve_params, loss_type='l2', is_pred_x0=False):
    """Create VE loss function"""

    if loss_type == 'l1':
        loss_fn = l1_loss
    elif loss_type == 'l2':
        loss_fn = l2_loss
    elif loss_type == 'rel2':
        loss_fn = rel2_loss
    else:
        raise NotImplementedError(f'Loss type {loss_type} not supported')

    def ve_loss(params, batch):
        #x, x_t, batched_t, noise = batch
        x, x_sigma, sigma, weight, noise = batch

        B = x_sigma.shape[0]

        # Target is either original image (if is_pred_x0) or noise
        target = noise if not is_pred_x0 else x

        # Model prediction
        pred = model.apply(params, x_sigma, sigma) ## TODO

        # Compute loss
        loss = loss_fn(flatten(pred), flatten(target))
        #print("B:", B, ",  loss:", loss.shape, ",  weight:", weight.shape)
        if weight.shape != (B,):
            weight = weight[..., 0, 0, 0]
        loss = loss * weight
        #print("B:", B, ",  loss:", loss.shape)
        assert loss.shape == (B,)


        return loss.mean()

    return ve_loss

def create_edm_loss_fn(model, edm_params, loss_type='l2', is_pred_x0=False):
    """Create EDM loss function"""

    if loss_type == 'l1':
        loss_fn = l1_loss
    elif loss_type == 'l2':
        loss_fn = l2_loss
    elif loss_type == 'rel2':
        loss_fn = rel2_loss
    else:
        raise NotImplementedError(f'Loss type {loss_type} not supported')

    def edm_loss(params, batch):
        #x, x_t, batched_t, noise = batch
        x, x_sigma, sigma, weight, noise = batch

        B = x_sigma.shape[0]

        # Target is either original image (if is_pred_x0) or noise
        target = noise if not is_pred_x0 else x

        # Model prediction
        pred = model.apply(params, x_sigma, sigma) ## TODO

        # Compute loss
        loss = loss_fn(flatten(pred), flatten(target))
        assert loss.shape == (B,)
        #print("B:", B, ",  loss:", loss.shape, ",  weight:", weight.shape)
        if weight.shape != (B,):
            weight = weight[..., 0, 0, 0]
        loss = loss * weight
        #print("B:", B, ",  loss:", loss.shape)
        


        return loss.mean()

    return edm_loss

def create_ddpm_train_step(model, ddpm_params, mesh, loss_type='l2', is_pred_x0=False):
    """Create DDPM training step with JAX sharding"""

    ddpm_loss_fn = create_ddpm_loss_fn(
        model, ddpm_params, loss_type, is_pred_x0
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

def create_ve_train_step(model, ve_params, mesh, loss_type='l2', is_pred_x0=False):
    """Create VE training step with JAX sharding"""

    ve_loss_fn = create_ve_loss_fn(
        model, ve_params, loss_type, is_pred_x0
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
        grad_fn = jax.value_and_grad(ve_loss_fn, has_aux=False)
        loss, grads = grad_fn(state.params, batch)

        # Average across devices
        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")

        # Update parameters
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step

def create_edm_train_step(model, edm_params, mesh, loss_type='l2', is_pred_x0=False):
    """Create EDM training step with JAX sharding"""

    edm_loss_fn = create_edm_loss_fn(
        model, edm_params, loss_type, is_pred_x0
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
        grad_fn = jax.value_and_grad(edm_loss_fn, has_aux=False)
        loss, grads = grad_fn(state.params, batch)

        # Average across devices
        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")

        # Update parameters
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step

def noise_to_x0(noise, xt, sigma_batch):
    sigma_batch = jnp.clip(sigma_batch, a_min=1e-5, a_max=1-(1e-5))
    sqrt_alpha_bar = jnp.sqrt(1. - sigma_batch ** 2)
    x0 = (xt - sigma_batch * noise) / sqrt_alpha_bar
    return x0

def x0_to_noise(x0, xt, sigma_batch):
    sigma_batch = jnp.clip(sigma_batch, a_min=1e-5, a_max=1-(1e-5))
    sqrt_alpha_bar = jnp.sqrt(1. - sigma_batch ** 2)
    noise = (xt - sqrt_alpha_bar * x0) / sigma_batch
    return noise


def get_posterior_mean_variance(img, t, x0, v, ddpm_params):
    beta = ddpm_params['betas'][t, None, None, None]
    alpha = ddpm_params['alphas'][t, None, None, None]
    alpha_bar = ddpm_params['alphas_bar'][t, None, None, None]
    alpha_bar_last = ddpm_params['alphas_bar'][t - 1, None, None, None]
    sqrt_alpha_bar_last = ddpm_params['sqrt_alphas_bar'][t - 1, None, None, None]

    # only needed when t > 0
    coef_x0 = beta * sqrt_alpha_bar_last / (1. - alpha_bar)
    coef_xt = (1. - alpha_bar_last) * jnp.sqrt(alpha) / (1 - alpha_bar)
    posterior_mean = coef_x0 * x0 + coef_xt * img

    posterior_variance = beta * (1 - alpha_bar_last) / (1. - alpha_bar)
    posterior_log_variance = jnp.log(jnp.clip(posterior_variance, a_min=1e-16))

    return posterior_mean, posterior_log_variance


def model_predict(state, x, sigma_batch, ddpm_params, is_pred_x0):
    sigma_batch = jnp.clip(sigma_batch, a_min=1e-5, a_max=1 - (1e-5))
    log_sigma = sigma_batch[..., 0, 0, 0]  # (B,) No log here TODO
    pred = state.apply_fn(state.params, x, log_sigma)  # 

    if is_pred_x0:
        x0_pred = pred
        noise_pred = x0_to_noise(pred, x, sigma_batch)
    else:
        noise_pred = pred
        x0_pred = noise_to_x0(pred, x, sigma_batch)
    return x0_pred, noise_pred


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
