from __future__ import annotations

import math


def sanity_check(n_samples, n_bkps, jump, min_size):
    n_adm_bkps = n_samples // jump
    if n_bkps > n_adm_bkps:
        return False
    if n_bkps * math.ceil(min_size / jump) * jump + min_size > n_samples:
        return False
    return True

