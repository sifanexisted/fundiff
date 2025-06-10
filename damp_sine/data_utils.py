from functools import partial

import numpy as np
from einops import rearrange, repeat

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import jax
import jax.numpy as jnp

from jax import random, jit

from torch.utils.data import Dataset, DataLoader, Subset


def generate_dataset(num_samples=100, num_sensors=100):
    """Generate family of damped oscillations in [0,1]"""
    x = np.linspace(0, 1, num_sensors)

    y_list = []
    x_list = []

    # Parameter ranges
    A_range = (0.5, 1.0)  # Amplitude
    gamma_range = (2, 4)  # Decay
    omega_range = (6 * np.pi, 8 * np.pi)  # Frequency
    # phi_range = (0, 2 * np.pi)  # Phase
    shift_range = (-0.5, 0.5)  # Shift

    for _ in range(num_samples):
        A = np.random.uniform(*A_range)
        gamma = np.random.uniform(*gamma_range)
        omega = np.random.uniform(*omega_range)
        shift = np.random.uniform(*shift_range)

        y = A * np.exp(-gamma * x) * np.sin(omega * x) + shift

        x_list.append(x)
        y_list.append(y)

    x = np.array(x_list)
    y = np.array(y_list)

    x = np.array(x[..., None])
    y = np.array(y[..., None])
    return x, y


class BaseDataset(Dataset):
    # This dataset class is used for homogenization
    def __init__(
            self,
            x,
            y,
            downsample_factor=1,
            num_samples=None,
            ):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.num_samples = num_samples  # Number of samples to use for training, if None use all samples by default

        self.x = x[:, ::downsample_factor]
        self.y = y[:, ::downsample_factor]

    def __len__(self):
        # Assuming all datasets have the same length, use the length of the first one
        if self.num_samples is not None:
            return self.num_samples
        else:
            return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        batch = np.concatenate([x, y], axis=1)

        return batch


class BatchParser:
    def __init__(self, n_x):

        x_star = jnp.linspace(0, 1, n_x)
        self.coords = x_star[:, None]

    # @partial(jit, static_argnums=(0,))
    def random_query(self, batch, num_sensors, num_queries, rng_key=None):
        batch_inputs = batch
        batch_outputs = batch_inputs[..., 1:2]

        key1, key2 = random.split(rng_key)

        sensor_idx = random.choice(key1, batch_outputs.shape[1], (num_sensors,), replace=False)

        # Sort the sensors
        sensor_idx = jnp.sort(sensor_idx)
        batch_inputs = batch_inputs[:, sensor_idx]

        coords_idx = random.choice(
            key2, batch_outputs.shape[1], (num_queries,), replace=False
            )
        batch_coords = self.coords[coords_idx]
        batch_outputs = batch_outputs[:, coords_idx]

        # Repeat the coords across devices
        batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

        return batch_coords, batch_inputs, batch_outputs

    @partial(jit, static_argnums=(0,))
    def query_all(self, batch):
        batch_inputs = batch

        batch_outputs = batch_inputs[..., 1]
        batch_coords = self.coords

        # Repeat the coords  across devices
        batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

        return batch_coords, batch_inputs, batch_outputs


def fit_damped_sine(t, y):
    """Fit damped sine to discrete data"""

    def damped_sine(t, A, gamma, omega, phi):
        return A * np.exp(-gamma * t) * np.sin(omega * t + phi)

    # Initial parameter estimation
    # Amplitude: max absolute value
    A_guess = np.max(np.abs(y))

    # Frequency: from peak distances
    peaks, _ = find_peaks(y)
    if len(peaks) >= 2:
        T = np.mean(np.diff(t[peaks]))
        omega_guess = 2 * np.pi / T
    else:
        # Fallback: FFT
        freqs = np.fft.fftfreq(len(t), t[1] - t[0])
        fft = np.fft.fft(y)
        omega_guess = 2 * np.pi * abs(freqs[np.argmax(np.abs(fft[1:])) + 1])

    # Decay rate: from peak amplitudes
    if len(peaks) >= 2:
        peak_vals = y[peaks]
        gamma_guess = -np.log(abs(peak_vals[1] / peak_vals[0])) / (t[peaks[1]] - t[peaks[0]])
    else:
        gamma_guess = 1.0

    # Phase: zero initial guess
    phi_guess = 0.0

    # Curve fit
    popt, pcov = curve_fit(damped_sine, t, y,
                           p0=[A_guess, gamma_guess, omega_guess, phi_guess],
                           bounds=([0, 0, 0, -2 * np.pi],
                                   [np.inf, np.inf, np.inf, 2 * np.pi]))

    # Get uncertainties
    perr = np.sqrt(np.diag(pcov))

    return {
        'A': popt[0],
        'gamma': popt[1],
        'omega': popt[2],
        'phi': popt[3],
        'uncertainties': {
            'A_err': perr[0],
            'gamma_err': perr[1],
            'omega_err': perr[2],
            'phi_err': perr[3]
            }
        }
