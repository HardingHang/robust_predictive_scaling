"""Unit tests for the scalers (paper Section III-C, Algorithm 1)."""

import numpy as np
import pytest

from scaler_manager.adaptive import AdaptiveScaler, uncertainty_level
from scaler_manager.point import PointScaler
from scaler_manager.reactive import ReactiveAvg, ReactiveMax
from scaler_manager.robust import RobustScaler

THETA = 70.0


def forecasts(**kwargs):
    return {tau: np.asarray([v], dtype=float) for tau, v in kwargs.items()}


class TestRobustScaler:
    def test_closed_form_allocation(self):
        scaler = RobustScaler(theta=THETA, tau=0.9)
        q = {0.9: np.array([140.0, 139.0, 70.0, 10.0])}
        assert list(scaler.allocate(q)) == [3, 2, 2, 1]

    def test_missing_quantile_raises(self):
        with pytest.raises(KeyError):
            RobustScaler(theta=THETA, tau=0.9).allocate({0.5: np.array([1.0])})

    def test_invalid_tau(self):
        with pytest.raises(ValueError):
            RobustScaler(theta=THETA, tau=1.5)


class TestPointScaler:
    def test_matches_robust_at_median(self):
        q = {0.5: np.array([100.0, 200.0]), 0.9: np.array([150.0, 250.0])}
        point = PointScaler(theta=THETA).allocate(q)
        robust = RobustScaler(theta=THETA, tau=0.5).allocate(q)
        assert list(point) == list(robust)


class TestUncertaintyLevel:
    def test_zero_when_all_equal_median(self):
        assert uncertainty_level({0.1: 5.0, 0.5: 5.0, 0.9: 5.0}) == 0.0

    def test_hand_computed(self):
        # {0.1:4, 0.5:5, 0.9:7}: (5-4)*(0.1-0) + (5-7)*(0.9-1) = 0.1 + 0.2
        assert uncertainty_level({0.1: 4.0, 0.5: 5.0, 0.9: 7.0}) == pytest.approx(0.3)

    def test_grows_with_spread(self):
        narrow = uncertainty_level({0.1: 4.5, 0.5: 5.0, 0.9: 5.5})
        wide = uncertainty_level({0.1: 3.0, 0.5: 5.0, 0.9: 9.0})
        assert 0 <= narrow < wide

    def test_missing_median_raises(self):
        with pytest.raises(KeyError):
            uncertainty_level({0.1: 1.0, 0.9: 2.0})


class TestAdaptiveScaler:
    def _forecasts(self):
        # Step 0: narrow interval (low U). Step 1: wide interval (high U).
        return {
            0.1: np.array([95.0, 20.0]),
            0.5: np.array([100.0, 100.0]),
            0.9: np.array([105.0, 250.0]),
        }

    def test_picks_quantile_by_uncertainty(self):
        # U_0 = (100-95)*0.1 + (100-105)*(0.9-1) = 0.5 + 0.5 = 1.0
        # U_1 = (100-20)*0.1 + (100-250)*(0.9-1) = 8 + 15 = 23.0
        scaler = AdaptiveScaler(theta=THETA, tau1=0.5, tau2=0.9, rho_tau=5.0)
        alloc = scaler.allocate(self._forecasts())
        # step 0: U=1 < 5 -> optimistic 0.5: floor(100/70)+1 = 2
        # step 1: U=23 >= 5 -> conservative 0.9: floor(250/70)+1 = 4
        assert list(alloc) == [2, 4]

    def test_between_fixed_quantile_bounds(self):
        rng = np.random.default_rng(1)
        median = rng.uniform(50, 200, 48)
        qs = {
            0.1: median * rng.uniform(0.5, 0.9, 48),
            0.5: median,
            0.9: median * rng.uniform(1.1, 2.5, 48),
        }
        adaptive = AdaptiveScaler(theta=THETA, tau1=0.5, tau2=0.9, rho_tau=3.0).allocate(qs)
        low = RobustScaler(theta=THETA, tau=0.5).allocate(qs)
        high = RobustScaler(theta=THETA, tau=0.9).allocate(qs)
        assert np.all(low <= adaptive) and np.all(adaptive <= high)

    def test_std_forecast_variant(self):
        # Parametric forecaster variant: U is the predicted std directly.
        qs = {0.5: np.array([100.0, 100.0]), 0.9: np.array([105.0, 250.0])}
        scaler = AdaptiveScaler(
            theta=THETA, tau1=0.5, tau2=0.9, rho_tau=10.0,
            std_forecast=np.array([1.0, 50.0]),
        )
        assert list(scaler.allocate(qs)) == [2, 4]

    def test_invalid_quantile_order(self):
        with pytest.raises(ValueError):
            AdaptiveScaler(theta=THETA, tau1=0.9, tau2=0.5, rho_tau=1.0)


class TestReactiveScalers:
    def test_reactive_max_uses_history_only(self):
        series = np.array([100.0, 100.0, 500.0, 100.0])
        alloc = ReactiveMax(theta=THETA, window=2).allocate_series(series)
        # t=0: no history -> current value; t=1: max(100)=100 -> 2
        # t=2: max(100,100)=100 -> 2 (spike not seen yet, reactive lag)
        # t=3: max(100,500)=500 -> 8
        assert list(alloc) == [2, 2, 2, 8]

    def test_reactive_avg_shape(self):
        series = np.full(20, 100.0)
        alloc = ReactiveAvg(theta=THETA, window=6, half_life=6.0).allocate_series(series)
        assert alloc.shape == (20,)
        # constant series -> average 100 -> 2 nodes everywhere
        assert set(alloc) == {2}
