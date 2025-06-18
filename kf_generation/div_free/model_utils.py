import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap, jit, lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


@partial(jit, static_argnums=(0,))
def stream_net(decoder, decoder_params, z, x, y):
    coords = jnp.stack([x, y])
    phi = decoder.apply(decoder_params, z, coords)
    return phi.squeeze()


@partial(jit, static_argnums=(0,))
def velocity_net(decoder, decoder_params, z, x, y):
    stream_fn = partial(stream_net, decoder)
    u = jax.jacfwd(stream_fn, argnums=3)(decoder_params, z, x, y)
    v = -jax.jacfwd(stream_fn, argnums=2)(decoder_params, z, x, y)
    return u, v


@partial(jit, static_argnums=(0,))
def u_net(decoder, decoder_params, z, x, y):
    return velocity_net(decoder, decoder_params, z, x, y)[0]


@partial(jit, static_argnums=(0,))
def v_net(decoder, decoder_params, z, x, y):
    return velocity_net(decoder, decoder_params, z, x, y)[1]


@partial(jit, static_argnums=(0,))
def div_net(decoder, decoder_params, z, x, y):
    u_fn = partial(u_net, decoder)
    v_fn = partial(v_net, decoder)

    u_x = jax.jacfwd(u_fn, argnums=2)(decoder_params, z, x, y)
    v_y = jax.jacfwd(v_fn, argnums=3)(decoder_params, z, x, y)
    div = u_x + v_y
    return div


@partial(jit, static_argnums=(0, 1))
def loss_fn(encoder, decoder, params, batch):
    encoder_params, decoder_params = params
    coords, x, y = batch
    coords = jnp.squeeze(coords)
    u_true, v_true = y[..., 0], y[..., 1]

    z = encoder.apply(encoder_params, x)

    u_pred, v_pred = vmap(
        partial(velocity_net, decoder),
        in_axes=(None, None, 0, 0), out_axes=1
    )(decoder_params, z, coords[:, 0], coords[:, 1])

    loss = jnp.mean((u_true - u_pred) ** 2) + jnp.mean((v_true - v_pred) ** 2)
    return loss


def create_train_step(encoder, decoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P()),
        check_rep=False
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
        u_pred, v_pred = vmap(
            partial(velocity_net, decoder),
            in_axes=(None, None, 0, 0), out_axes=1
        )(decoder_params, z, coords[:, 0], coords[:, 1])

        div_pred = vmap(
            partial(div_net, decoder),
            in_axes=(None, None, 0, 0), out_axes=1
        )(decoder_params, z, coords[:, 0], coords[:, 1])
        return u_pred, v_pred, div_pred

    return decoder_step


def create_eval_step(encoder, decoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P("batch"), P("batch"),  P("batch")),
        check_rep=False
    )
    def eval_step(params, batch):
        encoder_params, decoder_params = params
        coords, x, y = batch
        coords = jnp.squeeze(coords)

        z = encoder.apply(encoder_params, x)

        u_pred, v_pred = vmap(
            partial(velocity_net, decoder),
            in_axes=(None, None, 0, 0), out_axes=1
        )(decoder_params, z, coords[:, 0], coords[:, 1])

        div_pred = vmap(
            partial(div_net, decoder),
            in_axes=(None, None, 0, 0), out_axes=1
        )(decoder_params, z, coords[:, 0], coords[:, 1])

        return u_pred, v_pred, div_pred

    return eval_step







