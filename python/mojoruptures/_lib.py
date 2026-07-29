"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "ruptures.mojo")
LIB = os.environ.get("MOJORUPTURES_LIB") or os.path.join(
    ROOT, "dist", "libmojo-ruptures.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mr_l2_prefix": ([I, I, I, I, I], None),
    "mr_l2_error": ([I, I, I, I, I], F),
    "mr_l2_error_many": ([I, I, I, I, I, I, I], None),
    "mr_dynp": ([I, I, I, I, I, I, I, I, I, I], I),
    "mr_pelt": ([I, I, I, I, I, I, I, I, I, I, F], I),
    "mr_binseg": ([I, I, I, I, I, I, I, I, F], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    """Build the shared library when it is absent or older than the source."""
    if os.environ.get("MOJORUPTURES_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not os.path.exists(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no Mojo source at {SRC} and no shared library at {LIB}")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    pixi = shutil.which("pixi")
    if pixi and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        cmd = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
    else:
        mojo = shutil.which("mojo")
        if not mojo:
            raise BuildError("mojo not found; run `pixi run build` in the checkout")
        os.makedirs(os.path.dirname(LIB), exist_ok=True)
        cmd = [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray, dtype: np.dtype) -> int:
    """Return an ABI-safe address after checking the complete buffer contract."""
    expected = np.dtype(dtype)
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if array.dtype != expected:
        raise TypeError(f"native buffer must have dtype {expected}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("native buffer must be C-contiguous")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("native buffer has a null data pointer")
    if address > np.iinfo(np.int64).max:
        raise OverflowError("native buffer address does not fit the Mojo Int ABI")
    return address
