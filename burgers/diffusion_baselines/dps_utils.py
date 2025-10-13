import jax
import jax.numpy as jnp

def get_burgers_res(u, nu=0.001, x0=0.0, x1=1.0, t0=0.0, t1=1.0):
    """
    Burgers PDE residual:
        r = u_t + u * u_x - nu * u_xx
    with Euler at boundaries and central differences inside,
    implemented using jnp.gradient.
    Keeps same shape as u.

    u: (B, nt, nx, C)
    """
    B, nt, nx, C = u.shape
    dx = (x1 - x0) / (nx - 1)
    dt = (t1 - t0) / (nt - 1)


    # padding u in space dimension (wrap around)
    u = jnp.pad(u, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='wrap')

    # padding u in time dimension (replicate boundary)
    u = jnp.pad(u, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='edge')

    # time derivative with finite difference
    u_t = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dt)

    # first space derivative
    u_x = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dx)

    # second space derivative
    u_xx = (u[:, 1:-1, 2:] - 2 * u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dx ** 2)

    res = u_t + u[:, 1:-1, 1:-1] * u_x - nu * u_xx
    return res


#
# import jax
# import jax.numpy as jnp
#
# @jax.jit
# def burgers_residual(u, nu=1e-3, x0=0.0, x1=1.0, t0=0.0, t1=1.0):
#     """
#     u: (B, nt, nx, C)
#     Returns: residual with same shape.
#     """
#     B, nt, nx, C = u.shape
#     dx = (x1 - x0) / (nx - 1)
#     dt = (t1 - t0) / (nt - 1)
#
#     # space derivatives (periodic in x)
#     u_x  = (jnp.roll(u, -1, axis=2) - jnp.roll(u,  1, axis=2)) / (2.0 * dx)
#     u_xx = (jnp.roll(u, -1, axis=2) - 2.0 * u + jnp.roll(u, 1, axis=2)) / (dx * dx)
#
#     # time derivative (edge replicate in t)
#     u_tm1 = jnp.concatenate([u[:, :1],  u[:, :-1]], axis=1)  # replicate first frame
#     u_tp1 = jnp.concatenate([u[:, 1:],  u[:, -1:]], axis=1)  # replicate last frame
#     u_t   = (u_tp1 - u_tm1) / (2.0 * dt)
#
#     return u_t + u * u_x - nu * u_xx