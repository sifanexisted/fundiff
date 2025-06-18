import jax

import jax.numpy as jnp
from functools import partial
from jax import vmap, jit, lax, random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from flax.training import train_state


def create_autoencoder_state(config, encoder, decoder, tx):
    x = jnp.ones(config.x_dim)

    encoder_params = encoder.init(random.PRNGKey(config.seed), x=x)
    z = encoder.apply(encoder_params, x)
    decoder_params = decoder.init(random.PRNGKey(config.seed), x=z)
    params = (encoder_params, decoder_params)

    state = train_state.TrainState.create(apply_fn=decoder.apply, params=params, tx=tx)
    return state


@partial(jit, static_argnums=(0, 1))
def loss_fn(encoder, decoder, params, batch):
    encoder_params, decoder_params = params
    x, y = batch
    z = encoder.apply(encoder_params, x)
    u_pred = decoder.apply(decoder_params, z)
    loss = jnp.mean((y - u_pred) ** 2)
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
        x, _ = batch
        z = encoder.apply(encoder_params, x)
        return z

    return encoder_step


def create_decoder_step(decoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=P("batch"),
        )
    def decoder_step(decoder_params, z):
        u_pred = decoder.apply(decoder_params, z)
        return u_pred

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
        x, y = batch
        z = encoder.apply(encoder_params, x)
        u_pred = decoder.apply(decoder_params, z)

        return u_pred

    return eval_step







