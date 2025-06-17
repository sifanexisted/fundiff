import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap, jit, lax, random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


@partial(jit, static_argnums=(0,))
def u_net(decoder, decoder_params, z, t, x):
    coords = jnp.stack([t, x], axis=-1)
    u = decoder.apply(decoder_params, z, coords)
    return u.squeeze()

@partial(jit, static_argnums=(0,))
def r_net(decoder, decoder_params, z, t, x):
    u_fn = partial(u_net, decoder)

    u = u_fn(decoder_params, z, t, x)
    u_t = jax.jacfwd(u_fn, argnums=2)(decoder_params, z, t, x)
    u_x = jax.jacfwd(u_fn, argnums=3)(decoder_params, z, t, x)
    u_xx = jax.jacfwd(jax.jacfwd(u_fn, argnums=3), argnums=3)(decoder_params, z, t, x)
    r = u_t + u * u_x - 0.001 * u_xx
    return r


@partial(jit, static_argnums=(0, 1, 4))
def loss_fn(encoder, decoder, params, batch, use_pde=True):
    encoder_params, decoder_params = params
    coords, x, y = batch
    coords = jnp.squeeze(coords)
    z = encoder.apply(encoder_params, x)

    u_pred = vmap(
        partial(u_net, decoder),
        in_axes=(None, None, 0, 0), out_axes=1
    )(decoder_params, z, coords[:, 0], coords[:, 1])

    r_pred = vmap(
        partial(r_net, decoder),
        in_axes=(None, None, 0, 0), out_axes=1
    )(decoder_params, z, coords[:, 0], coords[:, 1])

    y = jnp.squeeze(y)
    loss_data = jnp.mean((y - u_pred) ** 2)
    loss_res = jnp.mean((r_pred) ** 2)

    # Use different loss formulation based on use_residual flag
    loss = 100 * loss_data + loss_res if use_pde else loss_data

    return loss, (loss_data, loss_res)


def create_train_step(encoder, decoder, mesh, use_pde=True):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P(), P(), P()),
        check_rep=False
    )
    def train_step(state, batch):
        # Pass the use_residual parameter to loss_fn
        grad_fn = jax.value_and_grad(
            partial(loss_fn, encoder, decoder, use_pde=use_pde),
            has_aux=True
        )
        (loss, aux), grads = grad_fn(state.params, batch)
        loss_data, loss_res = aux

        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        loss_data = lax.pmean(loss_data, "batch")
        loss_res = lax.pmean(loss_res, "batch")

        state = state.apply_gradients(grads=grads)
        return state, loss, loss_data, loss_res

    return train_step


def create_encoder_step(encoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=P("batch"),
        check_rep=False
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
        check_rep=False
        )
    def decoder_step(decoder_params, z, coords):
        u_pred = vmap(
            partial(u_net, decoder),
            in_axes=(None, None, 0, 0), out_axes=1
        )(decoder_params, z, coords[:, 0], coords[:, 1])

        r_pred = vmap(
            partial(r_net, decoder),
            in_axes=(None, None, 0, 0), out_axes=1
        )(decoder_params, z, coords[:, 0], coords[:, 1])

        return u_pred, r_pred

    return decoder_step


def create_eval_step(encoder, decoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P("batch")),
        check_rep=False
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




