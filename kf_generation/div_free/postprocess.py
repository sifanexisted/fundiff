import numpy as np
from scipy.fft import fft2, fftshift, fftfreq


def compute_energy_spectrum(u, v, Lx, Ly):
    """
    Computes the energy spectrum E(k) for a 2D velocity field.

    Parameters:
    - u, v: 2D velocity components.
    - Lx, Ly: Physical size of the domain in x and y.

    Returns:
    - k_bins_center: Center of each k bin.
    - E_k: Energy spectrum.
    """
    ny, nx = u.shape
    dx = Lx / nx
    dy = Ly / ny

    # Fourier transforms
    # u_hat = fft2(u)
    # v_hat = fft2(v)

    # Fourier transforms
    u_hat = fft2(u) / (nx * ny)  # Normalization
    v_hat = fft2(v) / (nx * ny)  # Normalization

    # Shift zero frequency component to the center
    u_hat = fftshift(u_hat)
    v_hat = fftshift(v_hat)

    # Wavenumbers
    kx = fftshift(fftfreq(nx, d=dx)) * 2 * np.pi
    ky = fftshift(fftfreq(ny, d=dy)) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kx ** 2 + ky ** 2)

    # Compute energy density
    energy_density = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)

    # Flatten the arrays
    k_mag_flat = k_mag.flatten()
    energy_flat = energy_density.flatten()

    # Define k bins
    k_max = np.max(k_mag_flat)
    dk = k_max / 50  # 100 bins
    bins = np.arange(0.0, k_max + dk, dk)
    bin_indices = np.digitize(k_mag_flat, bins)

    # Initialize energy spectrum
    E_k = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        E_k[i] = energy_flat[bin_indices == i + 1].sum()

    # Compute the center of each bin
    k_bins_center = 0.5 * (bins[:-1] + bins[1:])

    return k_bins_center, E_k