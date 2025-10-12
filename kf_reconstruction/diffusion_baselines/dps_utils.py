import jax
import jax.numpy as jnp


def get_div_res(uv, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    B, nt, nx, C = uv.shape

    u = uv[..., 0:1]
    v = uv[..., 1:2]

    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (nt - 1)

    # padding u and v
    u = jnp.pad(u, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='wrap')
    v = jnp.pad(v, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='wrap')

    u_x = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    v_y = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * dy)

    res = u_x + v_y

    return res

