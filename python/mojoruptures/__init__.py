"""Change-point detection with compute-heavy L2 kernels implemented in Mojo."""

from . import costs, datasets, detection, metrics
from ._lib import build
from .base import BaseCost, BaseEstimator
from .costs import CostL2
from .datasets import draw_bkps, pw_constant, pw_linear, pw_normal, pw_wavy
from .detection import Binseg, Dynp, Pelt
from .metrics import hausdorff, meantime, precision_recall, randindex

__version__ = "0.1.0"

__all__ = [
    "BaseCost",
    "BaseEstimator",
    "CostL2",
    "Dynp",
    "Pelt",
    "Binseg",
    "draw_bkps",
    "pw_constant",
    "pw_linear",
    "pw_normal",
    "pw_wavy",
    "hausdorff",
    "meantime",
    "precision_recall",
    "randindex",
    "build",
    "costs",
    "datasets",
    "detection",
    "metrics",
]
