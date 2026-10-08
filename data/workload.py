"""Workload trace construction.

Paper mapping: Section IV-A.1 (Datasets and Workloads).
Workload traces are aggregated at 10-minute intervals. A synthetic trace
generator with typical workload characteristics (daily / weekly seasonality,
spikes, heteroscedastic noise) is provided for quick end-to-end runs; real
traces (Alibaba / Google cluster data) can be plugged in through the
`TraceLoader` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

STEPS_PER_DAY = 144  # 10-minute intervals


class TraceLoader(ABC):
    """Interface for real workload traces (paper Section IV-A.1)."""

    @abstractmethod
    def load(self) -> np.ndarray:
        """Return the workload series aggregated at 10-minute intervals."""


class AlibabaTraceLoader(TraceLoader):
    """Alibaba Cluster Trace (https://github.com/alibaba/clusterdata).

    Construct a resource usage trace (CPU, memory, disk) by sampling a
    subset of machines and aggregating the resource usage.
    """

    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir

    def load(self) -> np.ndarray:
        raise NotImplementedError(
            "provide the preprocessed trace or use generate_synthetic_workload()"
        )


class GoogleTraceLoader(TraceLoader):
    """Google Cluster Trace (https://github.com/google/cluster-data).

    Construct a resource usage trace (CPU, memory) by sampling a subset of
    tasks and aggregating the resource usage.
    """

    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir

    def load(self) -> np.ndarray:
        raise NotImplementedError(
            "provide the preprocessed trace or use generate_synthetic_workload()"
        )


def generate_synthetic_workload(
    n_days: int = 60,
    base_level: float = 100.0,
    spike_prob: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic workload with daily/weekly seasonality, spikes and noise.

    The unit is abstract "workload"; with a per-node threshold theta
    (e.g. 70), the demand translates to roughly ``workload / theta`` nodes.

    Args:
        n_days: length of the series in days (144 steps per day).
        base_level: average workload level.
        spike_prob: probability of a spike starting at any time step.
        seed: RNG seed for reproducibility.

    Returns:
        Workload series of length ``n_days * 144``, strictly positive.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days * STEPS_PER_DAY)

    minute_of_day = (t % STEPS_PER_DAY) / STEPS_PER_DAY
    day_of_week = (t // STEPS_PER_DAY) % 7

    daily = 0.55 + 0.45 * np.sin(2 * np.pi * (minute_of_day - 0.25))
    weekly = np.where(day_of_week < 5, 1.0, 0.6)

    series = base_level * daily * weekly

    # Random spikes with exponential decay.
    spike_starts = rng.random(len(t)) < spike_prob
    spike_magnitudes = rng.uniform(1.5, 4.0, len(t)) * spike_starts
    decay = 0.5 ** (np.arange(12) / 2.0)  # half-life of 20 minutes
    spikes = np.convolve(spike_magnitudes, decay)[: len(t)]
    series = series * (1.0 + spikes)

    # Heteroscedastic noise: louder when the level is high.
    noise = rng.normal(0.0, 0.1 * series + 1.0)
    series = series + noise

    return np.clip(series, a_min=1.0, a_max=None)


def train_test_split(
    series: np.ndarray, context_length: int, prediction_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split into train / test so that the last full forecasting window of
    the test part is predictable from `context_length` steps before it."""
    test_len = max(prediction_length, len(series) // 5)
    train = series[: len(series) - test_len]
    test = series[len(series) - test_len - context_length :]
    return train, test
