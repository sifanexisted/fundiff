import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap, jit, lax, random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


@partial(jit, static_argnums=(0,))
def u_net(decoder, decoder_params, z, x, y):
    coords1 = jnp.stack([x, y], axis=-1)
    coords2 = jnp.stack([1 - x, y], axis=-1)
    coords3 = jnp.stack([x, 1 - y], axis=-1)
    coords4 = jnp.stack([1 - x, 1 - y], axis=-1)

    u1 = decoder.apply(decoder_params, z, coords1)
    u2 = decoder.apply(decoder_params, z, coords2)
    u3 = decoder.apply(decoder_params, z, coords3)
    u4 = decoder.apply(decoder_params, z, coords4)

    u = 0.25 * (u1 + u2 + u3 + u4)
    return u.squeeze()


@partial(jit, static_argnums=(0, 1))
def loss_fn(encoder, decoder, params, batch):
    encoder_params, decoder_params = params
    coords, x, y = batch
    coords = jnp.squeeze(coords)
    z = encoder.apply(encoder_params, x)

    u_pred = vmap(
        partial(u_net, decoder),
        in_axes=(None, None, 0, 0), out_axes=1
    )(decoder_params, z, coords[:, 0], coords[:, 1])

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
        u_pred = vmap(
            partial(u_net, decoder),
            in_axes=(None, None, 0, 0), out_axes=1
        )(decoder_params, z, coords[:, 0], coords[:, 1])

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
        coords, x, y = batch
        coords = jnp.squeeze(coords)

        z = encoder.apply(encoder_params, x)

        u_pred = vmap(
            partial(u_net, decoder),
            in_axes=(None, None, 0, 0), out_axes=1
            )(decoder_params, z, coords[:, 0], coords[:, 1])

        return u_pred

    return eval_step

