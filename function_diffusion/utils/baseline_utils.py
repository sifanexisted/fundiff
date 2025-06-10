from einops import rearrange, repeat

from tqdm import tqdm

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap, jit, lax, random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from flax.training import train_state

from function_diffusion.utils.data_utils import BaseDataset


@partial(jit, static_argnums=(0,))
def loss_fn(model, params, batch):
    x, y = batch
    pred = model.apply(params, x)
    loss = jnp.mean((y - pred) ** 2)
    return loss


def create_train_step(model, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P()),
    )
    def train_step(state, batch):
        grad_fn = jax.value_and_grad(partial(loss_fn, model), has_aux=False)
        loss, grads = grad_fn(state.params, batch)

        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        state = state.apply_gradients(grads=grads)
        return state, loss,

    return train_step


def create_eval_step(model, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P("batch")),
    )
    def eval_step(params, batch):
        x, y = batch
        pred = model.apply(params, x)
        return pred

    return eval_step





