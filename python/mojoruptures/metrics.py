from __future__ import annotations

from itertools import product

import numpy as np


class BadPartitions(Exception):
    pass


def _check_partitions(bkps1, bkps2):
    for name, bkps in zip(("first", "second"), (bkps1, bkps2)):
        if len(bkps) == 0:
            raise BadPartitions(f"The {name} partition is empty.")
    if max(bkps1) != max(bkps2):
        raise BadPartitions(
            "The end of the last regime is not the same for each of the "
            f"partitions:\n{bkps1}\n{bkps2}"
        )
    for bkps in (bkps1, bkps2):
        if len(set(bkps)) != len(bkps):
            raise BadPartitions(f"Some indexes are repeated: {bkps}")


def _distances(first, second):
    return np.abs(
        np.asarray(first[:-1], dtype=float)[:, None]
        - np.asarray(second[:-1], dtype=float)[None, :]
    )


def hausdorff(bkps1, bkps2):
    _check_partitions(bkps1, bkps2)
    distances = _distances(bkps1, bkps2)
    return max(distances.min(axis=0).max(), distances.min(axis=1).max())


def meantime(true_bkps, my_bkps):
    _check_partitions(true_bkps, my_bkps)
    return _distances(true_bkps, my_bkps).min(axis=0).mean()


def precision_recall(true_bkps, my_bkps, margin=10):
    _check_partitions(true_bkps, my_bkps)
    assert margin > 0, f"Margin of error must be positive (margin = {margin})"
    if len(my_bkps) == 1:
        return 0, 0
    used = set()
    true_positive = set(
        true_b
        for true_b, my_b in product(true_bkps[:-1], my_bkps[:-1])
        if my_b - margin < true_b < my_b + margin
        and not (my_b in used or used.add(my_b))
    )
    count = len(true_positive)
    return count / (len(my_bkps) - 1), count / (len(true_bkps) - 1)


def randindex(bkps1, bkps2):
    _check_partitions(bkps1, bkps2)
    n_samples = bkps1[-1]
    first = [0] + list(bkps1)
    second = [0] + list(bkps2)
    disagreement = 0
    begin_second = 0
    for first_idx in range(len(bkps1)):
        start1, end1 = first[first_idx : first_idx + 2]
        for second_idx in range(begin_second, len(bkps2)):
            start2, end2 = second[second_idx : second_idx + 2]
            overlap = max(min(end1, end2) - max(start1, start2), 0)
            disagreement += overlap * abs(end1 - end2)
            if end1 < end2:
                break
            begin_second = second_idx + 1
    disagreement /= n_samples * (n_samples - 1) / 2
    return 1.0 - disagreement


__all__ = ["hausdorff", "meantime", "precision_recall", "randindex"]
