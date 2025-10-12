import jax
import jax.numpy as jnp


def get_div_res(uv, x0=0.0, x1=1.0, y0=0.0, y1=1.0):

    B, nt, nx, C = uv.shape

    u = uv[..., 0:1]
    v = uv[..., 1:2]

    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (nt - 1)

    u_x = jnp.gradient(u, dx, axis=2)
    v_y = jnp.gradient(v, dy, axis=1)

    res = u_x + v_y
    return res

