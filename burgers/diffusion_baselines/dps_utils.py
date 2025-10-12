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

    # time derivative (Euler at ends, central inside)
    u_t = jnp.gradient(u, dt, axis=1)

    # first space derivative
    u_x = jnp.gradient(u, dx, axis=2)

    # second space derivative
    u_xx = jnp.gradient(u_x, dx, axis=2)

    res = u_t + u * u_x - nu * u_xx
    return res

