"""Auto-scaling outcome metrics.

Paper mapping: Section IV-C.

- Under-Provisioning Rate: fraction of periods in which the allocated
  resources fall short of the actual workload demand (w_t / c_t > theta).
- Over-Provisioning Rate: fraction of periods in which the allocated
  nodes exceed the minimum number required for the actual workload.
"""

from __future__ import annotations

import numpy as np


def min_nodes_required(workload: np.ndarray, theta: float) -> np.ndarray:
    """Minimum integer node count satisfying workload / c < theta.

    This is the closed-form per-step optimum of the scaling objective
    (paper Eq. 6 / Algorithm 1): c = floor(w / theta) + 1.
    """
    workload = np.asarray(workload, dtype=float)
    if theta <= 0:
        raise ValueError("theta must be positive")
    # A (forecasted) workload below zero carries no demand; keep one node
    # as the minimum allocation so node counts stay positive.
    workload = np.clip(workload, 0.0, None)
    return np.floor(workload / theta).astype(int) + 1


def under_provisioning_rate(
    allocations: np.ndarray, actual: np.ndarray, theta: float
) -> float:
    """Fraction of periods with insufficient resources: actual / c > theta."""
    allocations = np.asarray(allocations, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if allocations.shape != actual.shape:
        raise ValueError("allocations and actual must have the same shape")
    return float(np.mean(actual / allocations > theta))


def over_provisioning_rate(
    allocations: np.ndarray, actual: np.ndarray, theta: float
) -> float:
    """Fraction of periods where more nodes than necessary are allocated."""
    allocations = np.asarray(allocations, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if allocations.shape != actual.shape:
        raise ValueError("allocations and actual must have the same shape")
    return float(np.mean(allocations > min_nodes_required(actual, theta)))
