# mojo-ruptures

`mojo-ruptures` is a standalone Mojo port of the compute-heavy L2 change-point
detection core from the Python
[`ruptures`](https://centre-borelli.github.io/ruptures-docs/) package. Its Python
API keeps the upstream class names, constructor signatures, and
`fit`/`predict`/`fit_predict` conventions for the covered subset.

The native implementation builds centered prefix statistics once and executes
the segmentation recurrences in Mojo. It is intended for workloads where many
candidate segments must be scored, not as a replacement for every model and
visualization in upstream ruptures.

## Coverage

Implemented:

- `CostL2`, including `fit`, `error`, and `sum_of_costs`
- `Dynp`, `Pelt`, and `Binseg`, with the upstream constructor and prediction
  signatures
- all three `Binseg` stopping rules: `n_bkps`, `pen`, and `epsilon`
- custom cost objects through a correctness-first Python fallback
- `hausdorff`, `meantime`, `precision_recall`, and `randindex`
- `draw_bkps`, `pw_constant`, `pw_linear`, `pw_normal`, and `pw_wavy`
- `CostL2.error_many`, an additional batched API which amortizes FFI overhead

Not implemented:

- L1, RBF, cosine, normal, linear, autoregressive, rank, and Mahalanobis costs
- `BottomUp`, `Window`, and `KernelCPD`
- plotting and evaluation helpers other than the four metrics above

Passing an unsupported `model` raises `ValueError` instead of silently
substituting L2. A supplied custom cost still works, but its detector recurrence
runs in Python because arbitrary Python callbacks cannot execute inside the
Mojo kernel.

## Install

The repository pins the tested Mojo nightly and installs upstream ruptures for
parity testing:

```bash
pixi install
pixi run build
pixi run test
```

The build emits `dist/libmojo-ruptures.so`. The activated Pixi environment sets
`PYTHONPATH=python`, so no editable installation is required.

## Usage

```python
import mojoruptures as rpt

signal, expected = rpt.pw_constant(
    n_samples=1_000,
    n_features=3,
    n_bkps=3,
    noise_std=1.0,
    seed=7,
)

detector = rpt.Pelt(model="l2", min_size=2, jump=5)
found = detector.fit_predict(signal, pen=10.0)
print(found)

cost = rpt.CostL2().fit(signal)
print(cost.error(0, found[0]))
```

Run that example from the checkout with `pixi run python example.py`, or use
the same imports in an interactive `pixi run python` session.

## Benchmarks

Measured with ruptures 1.1.10 on an Intel Xeon E5-2697 v4 at 2.30GHz, Linux
x86-64. Each output was checked for parity before timing. Times are the best of
three warm runs; detector rows include fitting and prefix construction. The
cost row compares already-fitted cost objects.

| Case | Mojo (ms) | ruptures (ms) | Speedup |
|---|---:|---:|---:|
| CostL2.fit, n=1m x 4d | 54.349 | 0.001 | 0.00x |
| CostL2.error_many, 20k x length 64 x 4d | 0.414 | 517.222 | 1249.79x |
| Dynp.fit_predict, n=600 x 3d, k=4 | 0.426 | 301.253 | 707.45x |
| Pelt.fit_predict, n=5k x 3d | 3.921 | 6221.820 | 1586.99x |
| Binseg.fit_predict, n=2k x 3d, k=8 | 0.226 | 176.425 | 780.78x |

These ratios reflect the specific covered workload: upstream computes each
candidate L2 cost from a NumPy slice and crosses Python for every recurrence
step, while the Mojo implementation uses constant-time prefix queries within
one native call. The fit row exposes the opposite tradeoff: upstream fit is
lazy, while this implementation eagerly builds prefix statistics so later
queries are constant time. Reproduce the table only through:

```bash
pixi run bench
```

The task takes a machine-wide file lock so concurrent benchmark jobs do not
contend with one another.

No GPU path is provided.

## How it works

NumPy inputs are converted once to C-contiguous `float64` row-major arrays.
Python passes their addresses, dimensions, and caller-owned scratch buffers to
the shared library through `ctypes`; the C ABI never receives a Python object
and Mojo performs no cross-boundary allocation.

For an `n x d` signal, `CostL2.fit` makes one allocation holding two
`(n + 1) x d` views for prefix sums and squared prefix sums. The native prefix
loop uses host-width SIMD with a scalar remainder. Large inputs can build the
sum and squared-sum planes concurrently; smaller inputs stay on the fused
serial path to avoid worker setup overhead.
Values are centered on the first sample in each feature before accumulation to
reduce cancellation for signals with large offsets. Any segment sum of squared
deviations can then be obtained in `O(d)` time. `Dynp`, PELT pruning, and binary
segmentation consume those statistics directly in the same compilation unit
and write breakpoint indices into caller-owned contiguous `int64` arrays.

The test suite asserts numerical and behavioral parity with the real upstream
package across one- and multivariate signals, multiple subsampling jumps and
stopping rules, custom costs, datasets, metrics, invalid configurations, and
large-offset inputs.

## License

The port is MIT licensed. The upstream ruptures copyright and BSD 2-Clause
terms are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
