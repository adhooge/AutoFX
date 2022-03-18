"""
Functionals to be used for simpler representation of time changing-features.
"""

import numpy as np
from numpy.typing import ArrayLike


def f_max(arr: ArrayLike) -> float:
    return np.max(arr)


def f_min(arr: ArrayLike) -> float:
    return np.min(arr)


def f_avg(arr: ArrayLike):
    return np.mean(arr)


def f_std(arr: ArrayLike):
    return np.std(arr)

