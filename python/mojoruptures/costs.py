from __future__ import annotations

import numpy as np

from ._lib import addr, lib
from .base import BaseCost
from .exceptions import NotEnoughPoints


class CostL2(BaseCost):
    """Least-squares segment cost backed by centered prefix statistics."""

    model = "l2"

    def __init__(self):
        self.signal = None
        self.min_size = 1
        self._prefix = None
        self._sums = None
        self._squares = None

    def fit(self, signal) -> "CostL2":
        array = np.asarray(signal)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim != 2:
            raise ValueError("signal must be one- or two-dimensional")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("signal must contain at least one sample and one feature")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError("complex signals are not supported")
        if np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > 8:
            raise TypeError("floating-point inputs wider than float64 are not supported")
        if np.issubdtype(array.dtype, np.integer) and array.size:
            largest = np.max(np.abs(array.astype(object)))
            if largest > 2**53:
                raise ValueError("integer signal values must be exactly representable as float64")
        self.signal = np.ascontiguousarray(array, dtype=np.float64)
        if not np.all(np.isfinite(self.signal)):
            raise ValueError("signal must contain only finite values")
        samples, dims = self.signal.shape
        self._prefix = np.empty((2, samples + 1, dims), dtype=np.float64)
        self._sums = self._prefix[0]
        self._squares = self._prefix[1]
        lib().mr_l2_prefix(
            addr(self.signal, np.float64),
            addr(self._sums, np.float64),
            addr(self._squares, np.float64),
            samples,
            dims,
        )
        return self

    def _check_fit(self):
        if self.signal is None:
            raise RuntimeError("fit must be called before error")

    def error(self, start, end) -> float:
        self._check_fit()
        start = _index(start, "start")
        end = _index(end, "end")
        if end - start < self.min_size:
            raise NotEnoughPoints
        if start < 0 or end > len(self.signal):
            raise IndexError("segment bounds are outside the fitted signal")
        return float(
            lib().mr_l2_error(
                addr(self._sums, np.float64),
                addr(self._squares, np.float64),
                start,
                end,
                self.signal.shape[1],
            )
        )

    def error_many(self, starts, ends) -> np.ndarray:
        """Evaluate many segments in one FFI call."""
        self._check_fit()
        starts_array = _indices(starts, "starts")
        ends_array = _indices(ends, "ends")
        if starts_array.ndim != 1 or ends_array.shape != starts_array.shape:
            raise ValueError("starts and ends must be one-dimensional arrays of equal size")
        if np.any(ends_array - starts_array < self.min_size):
            raise NotEnoughPoints
        if np.any(starts_array < 0) or np.any(ends_array > len(self.signal)):
            raise IndexError("segment bounds are outside the fitted signal")
        result = np.empty(starts_array.size, dtype=np.float64)
        if result.size:
            lib().mr_l2_error_many(
                addr(self._sums, np.float64),
                addr(self._squares, np.float64),
                addr(starts_array, np.int64),
                addr(ends_array, np.int64),
                addr(result, np.float64),
                result.size,
                self.signal.shape[1],
            )
        return result


def cost_factory(model, *args, **kwargs):
    if model == "l2":
        return CostL2(*args, **kwargs)
    raise ValueError(f"Not such model: {model}")


def _indices(values, name):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must contain integers")
    info = np.iinfo(np.int64)
    if array.size and (np.any(array < info.min) or np.any(array > info.max)):
        raise OverflowError(f"{name} values do not fit int64")
    return np.ascontiguousarray(array, dtype=np.int64)


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < np.iinfo(np.int64).min or value > np.iinfo(np.int64).max:
        raise OverflowError(f"{name} does not fit int64")
    return value


__all__ = ["CostL2", "cost_factory"]
