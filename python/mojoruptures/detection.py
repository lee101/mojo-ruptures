from __future__ import annotations

from functools import lru_cache
from math import floor

import numpy as np

from ._lib import addr, lib
from .base import BaseEstimator
from .costs import CostL2, cost_factory
from .exceptions import BadSegmentationParameters
from .utils import sanity_check


def _make_cost(model, custom_cost, params):
    if custom_cost is not None:
        return custom_cost
    return cost_factory(model=model, **({} if params is None else params))


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_float(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _endpoints(n_samples, jump):
    points = np.arange(0, n_samples, jump, dtype=np.int64)
    if points.size == 0 or points[0] != 0:
        points = np.r_[np.int64(0), points]
    return np.ascontiguousarray(np.r_[points, np.int64(n_samples)])


class Dynp(BaseEstimator):
    """Optimal fixed-count segmentation using dynamic programming."""

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        self.cost = _make_cost(model, custom_cost, params)
        if custom_cost is None:
            self.model_name = model
        self.min_size = max(_positive_int(min_size, "min_size"), self.cost.min_size)
        self.jump = _positive_int(jump, "jump")
        self.n_samples = None

    @lru_cache(maxsize=None)
    def seg(self, start, end, n_bkps):
        if n_bkps == 0:
            return {(start, end): self.cost.error(start, end)}
        admissible = []
        for bkp in range(start, end):
            if bkp % self.jump:
                continue
            if sanity_check(
                bkp - start, n_bkps - 1, self.jump, self.min_size
            ) and end - bkp >= self.min_size:
                admissible.append(bkp)
        if not admissible:
            raise AssertionError(
                f"No admissible last breakpoints found. start, end: ({start},{end}), "
                f"n_bkps: {n_bkps}."
            )
        candidates = []
        for bkp in admissible:
            partition = dict(self.seg(start, bkp, n_bkps - 1))
            partition[(bkp, end)] = self.cost.error(bkp, end)
            candidates.append(partition)
        return min(candidates, key=lambda partition: sum(partition.values()))

    def fit(self, signal) -> "Dynp":
        self.seg.cache_clear()
        self.cost.fit(signal)
        self.n_samples = np.asarray(signal).shape[0]
        return self

    def predict(self, n_bkps):
        if self.n_samples is None:
            raise RuntimeError("fit must be called before predict")
        if isinstance(n_bkps, (bool, np.bool_)) or not isinstance(
            n_bkps, (int, np.integer)
        ):
            raise TypeError("n_bkps must be an integer")
        n_bkps = int(n_bkps)
        if n_bkps < 0:
            raise ValueError("n_bkps must be non-negative")
        if not sanity_check(
            self.n_samples, n_bkps, self.jump, self.min_size
        ):
            raise BadSegmentationParameters
        if not isinstance(self.cost, CostL2):
            partition = self.seg(0, self.n_samples, n_bkps)
            return sorted(end for _, end in partition)

        endpoints = _endpoints(self.n_samples, self.jump)
        segments = n_bkps + 1
        dp = np.empty((segments + 1, endpoints.size), dtype=np.float64)
        back = np.empty(dp.shape, dtype=np.int64)
        result = np.empty(segments, dtype=np.int64)
        count = lib().mr_dynp(
            addr(self.cost._sums, np.float64),
            addr(self.cost._squares, np.float64),
            addr(endpoints, np.int64),
            addr(dp, np.float64),
            addr(back, np.int64),
            addr(result, np.int64),
            endpoints.size,
            self.cost.signal.shape[1],
            self.min_size,
            n_bkps,
        )
        if count == 0:
            raise BadSegmentationParameters
        return result[:count].tolist()

    def fit_predict(self, signal, n_bkps):
        return self.fit(signal).predict(n_bkps)


class Pelt(BaseEstimator):
    """Penalized exact segmentation with PELT pruning for L2 costs."""

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        self.cost = _make_cost(model, custom_cost, params)
        self.min_size = max(_positive_int(min_size, "min_size"), self.cost.min_size)
        self.jump = _positive_int(jump, "jump")
        self.n_samples = None

    def fit(self, signal) -> "Pelt":
        self.cost.fit(signal)
        self.n_samples = np.asarray(signal).shape[0]
        return self

    def _seg_python(self, pen):
        partitions = {0: {(0, 0): 0}}
        admissible = []
        indices = [
            point
            for point in range(0, self.n_samples, self.jump)
            if point >= self.min_size
        ] + [self.n_samples]
        for bkp in indices:
            new_point = floor((bkp - self.min_size) / self.jump) * self.jump
            admissible.append(new_point)
            subproblems = []
            kept_points = []
            for point in admissible:
                if point not in partitions:
                    continue
                partition = partitions[point].copy()
                partition[(point, bkp)] = self.cost.error(point, bkp) + pen
                subproblems.append(partition)
                kept_points.append(point)
            partitions[bkp] = min(subproblems, key=lambda part: sum(part.values()))
            best = sum(partitions[bkp].values())
            admissible = [
                point
                for point, partition in zip(kept_points, subproblems)
                if sum(partition.values()) <= best + pen
            ]
        answer = partitions[self.n_samples]
        del answer[(0, 0)]
        return answer

    def predict(self, pen):
        if self.n_samples is None:
            raise RuntimeError("fit must be called before predict")
        pen = _finite_float(pen, "pen")
        if not sanity_check(self.n_samples, 0, self.jump, self.min_size):
            raise BadSegmentationParameters
        if not isinstance(self.cost, CostL2):
            partition = self._seg_python(pen)
            return sorted(end for _, end in partition)

        first = ((self.min_size + self.jump - 1) // self.jump) * self.jump
        interior = np.arange(first, self.n_samples, self.jump, dtype=np.int64)
        endpoints = np.ascontiguousarray(
            np.r_[np.int64(0), interior, np.int64(self.n_samples)]
        )
        score = np.empty(endpoints.size, dtype=np.float64)
        back = np.empty(endpoints.size, dtype=np.int64)
        active = np.empty(endpoints.size, dtype=np.int64)
        result = np.empty(endpoints.size, dtype=np.int64)
        count = lib().mr_pelt(
            addr(self.cost._sums, np.float64),
            addr(self.cost._squares, np.float64),
            addr(endpoints, np.int64),
            addr(score, np.float64),
            addr(back, np.int64),
            addr(active, np.int64),
            addr(result, np.int64),
            endpoints.size,
            self.cost.signal.shape[1],
            self.min_size,
            pen,
        )
        if count == 0:
            raise BadSegmentationParameters
        return result[:count].tolist()

    def fit_predict(self, signal, pen):
        return self.fit(signal).predict(pen)


class Binseg(BaseEstimator):
    """Greedy binary segmentation with upstream-compatible stopping rules."""

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        self.cost = _make_cost(model, custom_cost, params)
        self.min_size = max(_positive_int(min_size, "min_size"), self.cost.min_size)
        self.jump = _positive_int(jump, "jump")
        self.n_samples = None
        self.signal = None

    @lru_cache(maxsize=None)
    def single_bkp(self, start, end):
        segment_cost = self.cost.error(start, end)
        if np.isneginf(segment_cost):
            return None, 0
        gains = []
        for bkp in range(start, end, self.jump):
            if bkp - start >= self.min_size and end - bkp >= self.min_size:
                gain = (
                    segment_cost
                    - self.cost.error(start, bkp)
                    - self.cost.error(bkp, end)
                )
                gains.append((gain, bkp))
        if not gains:
            return None, 0
        gain, bkp = max(gains)
        return bkp, gain

    def fit(self, signal) -> "Binseg":
        array = np.asarray(signal)
        self.signal = array.reshape(-1, 1) if array.ndim == 1 else array
        self.n_samples = self.signal.shape[0]
        self.cost.fit(signal)
        self.single_bkp.cache_clear()
        return self

    def _seg_python(self, n_bkps=None, pen=None, epsilon=None):
        bkps = [self.n_samples]
        while True:
            candidates = [
                self.single_bkp(start, end)
                for start, end in zip([0] + bkps[:-1], bkps)
            ]
            bkp, gain = max(candidates, key=lambda value: value[1])
            if bkp is None:
                break
            if n_bkps is not None:
                split = len(bkps) - 1 < n_bkps
            elif pen is not None:
                split = gain > pen
            else:
                split = self.cost.sum_of_costs(bkps) > epsilon
            if not split:
                break
            bkps.append(bkp)
            bkps.sort()
        return bkps

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        if not any(value is not None for value in (n_bkps, pen, epsilon)):
            raise AssertionError("Give a parameter.")
        if self.n_samples is None:
            raise RuntimeError("fit must be called before predict")
        if n_bkps is not None and (
            isinstance(n_bkps, (bool, np.bool_))
            or not isinstance(n_bkps, (int, np.integer))
        ):
            raise TypeError("n_bkps must be an integer")
        requested = 0 if n_bkps is None else int(n_bkps)
        if requested < 0:
            raise ValueError("n_bkps must be non-negative")
        if not sanity_check(
            self.n_samples, requested, self.jump, self.min_size
        ):
            raise BadSegmentationParameters
        if not isinstance(self.cost, CostL2):
            return self._seg_python(n_bkps, pen, epsilon)

        if n_bkps is not None:
            mode, target = 0, float(n_bkps)
        elif pen is not None:
            mode, target = 1, _finite_float(pen, "pen")
        else:
            mode, target = 2, _finite_float(epsilon, "epsilon")
        result = np.empty(self.n_samples + 1, dtype=np.int64)
        count = lib().mr_binseg(
            addr(self.cost._sums, np.float64),
            addr(self.cost._squares, np.float64),
            addr(result, np.int64),
            self.n_samples,
            self.cost.signal.shape[1],
            self.min_size,
            self.jump,
            mode,
            target,
        )
        return result[:count].tolist()

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        return self.fit(signal).predict(
            n_bkps=n_bkps, pen=pen, epsilon=epsilon
        )


__all__ = ["Dynp", "Pelt", "Binseg"]
