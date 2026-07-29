from __future__ import annotations

from itertools import cycle

import numpy as np


def draw_bkps(n_samples=100, n_bkps=3, seed=None):
    rng = np.random.default_rng(seed=seed)
    alpha = np.ones(n_bkps + 1) / (n_bkps + 1) * 2000
    bkps = np.cumsum(rng.dirichlet(alpha) * n_samples).astype(int).tolist()
    bkps[-1] = n_samples
    return bkps


def pw_constant(
    n_samples=200,
    n_features=1,
    n_bkps=3,
    noise_std=None,
    delta=(1, 10),
    seed=None,
):
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    signal = np.empty((n_samples, n_features), dtype=float)
    indices = np.arange(n_samples)
    delta_min, delta_max = delta
    center = np.zeros(n_features)
    rng = np.random.default_rng(seed=seed)
    for segment in np.split(indices, bkps):
        if segment.size:
            jump = rng.uniform(delta_min, delta_max, size=n_features)
            spin = rng.choice([-1, 1], n_features)
            center += jump * spin
            signal[segment] = center
    if noise_std is not None:
        signal = signal + rng.normal(size=signal.shape) * noise_std
    return signal, bkps


def pw_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=None, seed=None):
    rng = np.random.default_rng(seed=seed)
    covariates = rng.normal(size=(n_samples, n_features))
    coefficients, bkps = pw_constant(
        n_samples=n_samples,
        n_bkps=n_bkps,
        n_features=n_features,
        noise_std=None,
        seed=seed,
    )
    variable = np.sum(coefficients * covariates, axis=1)
    if noise_std is not None:
        variable += rng.normal(scale=noise_std, size=variable.shape)
    return np.c_[variable, covariates], bkps


def pw_normal(n_samples=200, n_bkps=3, seed=None):
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    signal = np.zeros((n_samples, 2), dtype=float)
    covariance1 = np.array([[1, 0.9], [0.9, 1]])
    covariance2 = np.array([[1, -0.9], [-0.9, 1]])
    rng = np.random.default_rng(seed=seed)
    for segment, covariance in zip(
        np.split(signal, bkps), cycle((covariance1, covariance2))
    ):
        segment += rng.multivariate_normal(
            [0, 0], covariance, size=segment.shape[0]
        )
    return signal, bkps


def pw_wavy(n_samples=200, n_bkps=3, noise_std=None, seed=None):
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    frequencies = np.zeros((n_samples, 2))
    for segment, value in zip(
        np.split(frequencies, bkps[:-1]),
        cycle((np.array([0.075, 0.1]), np.array([0.1, 0.125]))),
    ):
        segment += value
    times = np.arange(n_samples)
    signal = np.sum(
        [np.sin(2 * np.pi * times * frequency) for frequency in frequencies.T],
        axis=0,
    )
    if noise_std is not None:
        signal += np.random.default_rng(seed=seed).normal(
            scale=noise_std, size=signal.shape
        )
    return signal, bkps


__all__ = ["draw_bkps", "pw_constant", "pw_linear", "pw_normal", "pw_wavy"]
