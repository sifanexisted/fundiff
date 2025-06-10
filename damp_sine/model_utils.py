from tqdm import tqdm

import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap, jit, lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


@partial(jit, static_argnums=(0,))
def neural_net(decoder, decoder_params, z, x):
    coords = jnp.stack([x])
    y_pred = decoder.apply(decoder_params, z, coords)
    return y_pred.squeeze()


@partial(jit, static_argnums=(0, 1))
def loss_fn(encoder, decoder, params, batch):
    encoder_params, decoder_params = params
    coords, x, y = batch
    coords = jnp.squeeze(coords)

    z = encoder.apply(encoder_params, x)

    y_pred = vmap(
        partial(neural_net, decoder),
        in_axes=(None, None, 0), out_axes=1
    )(decoder_params, z, coords)

    y_pred = jnp.squeeze(y_pred)
    y = jnp.squeeze(y)

    loss = jnp.mean((y - y_pred) ** 2)
    return loss


def create_train_step(encoder, decoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P()),
    )
    def train_step(state, batch):
        grad_fn = jax.value_and_grad(partial(loss_fn, encoder, decoder), has_aux=False)
        loss, grads = grad_fn(state.params, batch)
        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


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


def create_decoder_step(decoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch"), P()),
        out_specs=P("batch"),
        )
    def decoder_step(decoder_params, z, coords):
        y_pred = vmap(
            partial(neural_net, decoder),
            in_axes=(None, None, 0), out_axes=1
            )(decoder_params, z, coords)
        return y_pred

    return decoder_step


def create_eval_step(encoder, decoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P("batch")),
    )
    def eval_step(params, batch):
        encoder_params, decoder_params = params
        coords, x, y = batch
        coords = jnp.squeeze(coords)

        z = encoder.apply(encoder_params, x)

        y_pred = vmap(
            partial(neural_net, decoder),
            in_axes=(None, None, 0), out_axes=1
        )(decoder_params, z, coords)

        return y_pred

    return eval_step







