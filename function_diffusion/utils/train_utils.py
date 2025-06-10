from functools import partial

from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, pmap, random
from jax.flatten_util import ravel_pytree
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


def compute_lp_norms(pred, y, ord=2):
    """
    pred: (b, h * w, c)
    y: (b, h * w, c)
    """

    diff_norms = jnp.linalg.norm(pred - y, axis=1, ord=ord, keepdims=True)
    y_norms = jnp.linalg.norm(y, axis=1, ord=ord, keepdims=True)
    lp_error = (diff_norms / y_norms).mean()

    return diff_norms, y_norms, lp_error


class PatchHandler:
    def __init__(self, inputs, patch_size):
        self.patch_size = patch_size

        _, self.height, self.width, self.channel = inputs.shape

        self.patch_height, self.patch_width = (
            self.height // self.patch_size[0],
            self.width // self.patch_size[1],
        )

    def merge_patches(self, x):
        batch, _, _ = x.shape
        x = jnp.reshape(
            x,
            (
                batch,
                self.patch_height,
                self.patch_width,
                self.patch_size[0],
                self.patch_size[1],
                -1,
            ),
        )
        x = jnp.swapaxes(x, 2, 3)
        x = jnp.reshape(
            x,
            (
                batch,
                self.patch_height * self.patch_size[0],
                self.patch_width * self.patch_size[1],
                -1,
            ),
        )
        return x


###################################################
############# utils for diffusion models ##########
###################################################


# def create_train_diffusion_step(model, mesh):
#     @jax.jit
#     @partial(
#         shard_map,
#         mesh=mesh,
#         in_specs=(P(), P("batch")),
#         out_specs=(P(), P()),
#     )
#     def train_step(state, batch):
#         def loss_fn(params):
#             x, t, y = batch
#             pred = model.apply(params, x, t)
#             loss = jnp.mean((y - pred) ** 2)
#             return loss
#
#         # Compute gradients and update parameters
#         grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
#         loss, grads = grad_fn(state.params)
#         grads = lax.pmean(grads, "batch")
#         loss = lax.pmean(loss, "batch")
#         state = state.apply_gradients(grads=grads)
#         return state, loss
#
#     return train_step



# @jit
# def get_diffusion_batch(key, z1=None, c=None):
#     key1, key2, key3 = random.split(key, 3)
#     z0 = random.normal(key1, shape=z1.shape)
#     t = random.uniform(key2, (z1.shape[0], 1, 1))
#
#     # z_t = t * z1 + (1. - t) * z0
#     z_t = t * (z1 - z0) + z0
#     target = z1 - z0
#     batch = (z_t, t.flatten(), target)
#
#     return batch, key3


# def sample_ode(state, z0=None, num_steps=None):
#     dt = 1 / num_steps
#     traj = [z0]
#
#     z = z0
#     for i in tqdm(range(num_steps)):
#         t = jnp.ones((z.shape[0],)) * i / num_steps
#         pred = state.apply_fn(state.params, z, t)
#         z = z + pred * dt
#         traj.append(z)
#     return z, traj


def create_encoder_step(encoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=P("batch"),
    )
    def encoder_step(encoder_params, batch):
        _, x, _ = batch
        z = encoder.apply(encoder_params, x)
        return z

    return encoder_step


def create_train_diffusion_step(model, mesh, use_conditioning=False):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P()),
    )
    def train_step(state, batch):
        def loss_fn(params):
            if use_conditioning:
                x, t, c, y = batch
                pred = model.apply(params, x, t, c)
            else:
                x, t, y = batch
                pred = model.apply(params, x, t)

            loss = jnp.mean((y - pred) ** 2)
            return loss

        # Compute gradients and update parameters
        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
        loss, grads = grad_fn(state.params)
        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


@partial(jit, static_argnums=(3,))
def get_diffusion_batch(key, z1=None, c=None, use_conditioning=False):
    keys = random.split(key, 3)
    z0 = random.normal(keys[0], shape=z1.shape)  # (b, 200, 512)
    t = random.uniform(keys[1], (z1.shape[0], 1, 1))

    # z_t = t * z1 + (1. - t) * z0
    z_t = t * (z1 - z0) + z0
    target = z1 - z0

    if use_conditioning:
        batch = (z_t, t.flatten(), c, target)
    else:
        batch = (z_t, t.flatten(), target)

    return batch, keys[2]


def sample_ode(state, z0=None, c=None, num_steps=None, use_conditioning=False):
    dt = 1 / num_steps
    traj = [z0]

    z = z0
    for i in tqdm(range(num_steps)):
        t = jnp.ones((z.shape[0],)) * i / num_steps
        if use_conditioning:
            pred = state.apply_fn(state.params, z, t, c)
        else:
            pred = state.apply_fn(state.params, z, t)
        z = z + pred * dt
        traj.append(z)
    return z, traj