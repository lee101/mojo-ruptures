"""End-to-end benchmarks against ruptures 1.1.10 on identical signals."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import ruptures as upstream

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojoruptures as mojo  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def piecewise_signal(n_samples, n_dims, seed=0):
    rng = np.random.default_rng(seed)
    boundaries = [0, n_samples // 4, n_samples // 2, 3 * n_samples // 4, n_samples]
    signal = np.empty((n_samples, n_dims), dtype=np.float64)
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        signal[start:end] = rng.normal(
            loc=(-2.0, 2.5, -0.5, 3.5)[index], size=(end - start, n_dims)
        )
    return signal


def cost_case():
    signal = piecewise_signal(100_000, 4)
    starts = np.arange(20_000, dtype=np.int64) * 3
    ends = starts + 64
    ours = mojo.CostL2().fit(signal)
    theirs = upstream.costs.CostL2().fit(signal)

    def mojo_run():
        return ours.error_many(starts, ends)

    def upstream_run():
        return np.array(
            [theirs.error(int(start), int(end)) for start, end in zip(starts, ends)]
        )

    assert np.allclose(mojo_run(), upstream_run(), rtol=1e-8, atol=1e-8)
    return mojo_run, upstream_run


def cost_fit_case():
    signal = piecewise_signal(1_000_000, 4, seed=4)

    def mojo_run():
        return mojo.CostL2().fit(signal)

    def upstream_run():
        return upstream.costs.CostL2().fit(signal)

    assert math.isclose(
        mojo_run().error(123, 987_654),
        upstream_run().error(123, 987_654),
        rel_tol=1e-8,
        abs_tol=1e-8,
    )
    return mojo_run, upstream_run


def cost_fit_parallel_case():
    signal = piecewise_signal(250_000, 64, seed=5)

    def mojo_run():
        return mojo.CostL2().fit(signal)

    def upstream_run():
        return upstream.costs.CostL2().fit(signal)

    assert math.isclose(
        mojo_run().error(123, 249_876),
        upstream_run().error(123, 249_876),
        rel_tol=1e-8,
        abs_tol=1e-8,
    )
    return mojo_run, upstream_run


def dynp_case():
    signal = piecewise_signal(600, 3, seed=1)

    def mojo_run():
        return mojo.Dynp(jump=5, min_size=5).fit_predict(signal, n_bkps=4)

    def upstream_run():
        return upstream.Dynp(jump=5, min_size=5).fit_predict(signal, n_bkps=4)

    assert mojo_run() == upstream_run()
    return mojo_run, upstream_run


def pelt_case():
    signal = piecewise_signal(5_000, 3, seed=2)

    def mojo_run():
        return mojo.Pelt(jump=5, min_size=5).fit_predict(signal, pen=20.0)

    def upstream_run():
        return upstream.Pelt(jump=5, min_size=5).fit_predict(signal, pen=20.0)

    assert mojo_run() == upstream_run()
    return mojo_run, upstream_run


def binseg_case():
    signal = piecewise_signal(2_000, 3, seed=3)

    def mojo_run():
        return mojo.Binseg(jump=5, min_size=5).fit_predict(signal, n_bkps=8)

    def upstream_run():
        return upstream.Binseg(jump=5, min_size=5).fit_predict(
            signal, n_bkps=8
        )

    assert mojo_run() == upstream_run()
    return mojo_run, upstream_run


CASES = [
    ("CostL2.fit, n=1m x 4d", cost_fit_case),
    ("CostL2.fit, n=250k x 64d", cost_fit_parallel_case),
    ("CostL2.error_many, 20k x length 64 x 4d", cost_case),
    ("Dynp.fit_predict, n=600 x 3d, k=4", dynp_case),
    ("Pelt.fit_predict, n=5k x 3d", pelt_case),
    ("Binseg.fit_predict, n=2k x 3d, k=8", binseg_case),
]


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| Case | Mojo (ms) | ruptures (ms) | Speedup |")
    print("|---|---:|---:|---:|")
    for name, prepare in CASES:
        mojo_run, upstream_run = prepare()
        mojo_run()
        upstream_run()
        mojo_seconds = timeit(mojo_run)
        upstream_seconds = timeit(upstream_run)
        print(
            f"| {name} | {mojo_seconds * 1e3:.3f} | "
            f"{upstream_seconds * 1e3:.3f} | "
            f"{upstream_seconds / mojo_seconds:.2f}x |"
        )


if __name__ == "__main__":
    main()
