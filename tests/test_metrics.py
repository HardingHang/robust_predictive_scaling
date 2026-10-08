"""Unit tests for metrics (paper Section IV-B/IV-C) with hand-computed cases."""

import numpy as np
import pytest

from metrics.forecasting import (
    coverage,
    mean_weighted_quantile_loss,
    mse_from_quantiles,
    quantile_loss,
    weighted_quantile_loss,
)
from metrics.scaling import (
    min_nodes_required,
    over_provisioning_rate,
    under_provisioning_rate,
)


class TestQuantileLoss:
    def test_underestimation_weighted_by_tau(self):
        # y=10, y_hat=8, tau=0.9 -> (10-8) * 0.9 = 1.8
        assert quantile_loss(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)

    def test_overestimation_weighted_by_one_minus_tau(self):
        # y=8, y_hat=10, tau=0.9 -> (8-10) * (0.9-1) = 0.2
        assert quantile_loss(np.array([8.0]), np.array([10.0]), 0.9) == pytest.approx(0.2)

    def test_tau_half_is_half_l1(self):
        # tau=0.5 reduces to |y - y_hat| * 0.5
        assert quantile_loss(np.array([10.0]), np.array([8.0]), 0.5) == pytest.approx(1.0)
        assert quantile_loss(np.array([8.0]), np.array([10.0]), 0.5) == pytest.approx(1.0)

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 100, 50)
        for tau in [0.1, 0.5, 0.9]:
            assert quantile_loss(y, y + rng.normal(0, 10, 50), tau) >= 0


class TestWeightedQuantileLoss:
    def test_hand_computed(self):
        y = np.array([10.0, 20.0])
        q = np.array([8.0, 18.0])
        # QL_0.5 = 0.5*(2+2) = 2 ; wQL = 2*2/30
        assert weighted_quantile_loss(y, q, 0.5) == pytest.approx(4 / 30)

    def test_zero_target_raises(self):
        with pytest.raises(ValueError):
            weighted_quantile_loss(np.array([0.0]), np.array([1.0]), 0.5)


class TestCoverage:
    def test_hand_computed(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        q = np.array([2.0, 2.0, 2.0, 2.0])
        assert coverage(y, q) == pytest.approx(0.5)

    def test_perfect_calibration_bounds(self):
        y = np.arange(100, dtype=float)
        assert coverage(y, y + 1) == 1.0
        assert coverage(y, y - 1) == 0.0


class TestMeanWeightedQuantileLoss:
    def test_is_mean_of_wql(self):
        y = np.array([10.0, 20.0])
        qs = {0.5: np.array([9.0, 19.0]), 0.9: np.array([12.0, 25.0])}
        expected = np.mean(
            [weighted_quantile_loss(y, qs[0.5], 0.5), weighted_quantile_loss(y, qs[0.9], 0.9)]
        )
        assert mean_weighted_quantile_loss(y, qs) == pytest.approx(expected)


class TestMseFromQuantiles:
    def test_mean_of_quantiles_as_point(self):
        qs = {0.1: np.array([1.0]), 0.9: np.array([3.0])}
        assert mse_from_quantiles(np.array([2.0]), qs) == pytest.approx(0.0)
        assert mse_from_quantiles(np.array([4.0]), qs) == pytest.approx(4.0)


class TestMinNodesRequired:
    def test_strict_inequality(self):
        # w/theta exactly integer -> one more node (w/c must be < theta)
        assert min_nodes_required(np.array([140.0]), 70.0)[0] == 3
        assert min_nodes_required(np.array([139.0]), 70.0)[0] == 2
        assert min_nodes_required(np.array([70.0]), 70.0)[0] == 2
        assert min_nodes_required(np.array([69.0]), 70.0)[0] == 1

    def test_invalid_theta(self):
        with pytest.raises(ValueError):
            min_nodes_required(np.array([1.0]), 0.0)

    def test_negative_forecast_keeps_one_node(self):
        # negative quantile forecasts (possible from neural models) carry no
        # demand; the allocation stays at the minimum of one node
        assert min_nodes_required(np.array([-5.0]), 70.0)[0] == 1


class TestProvisioningRates:
    def test_under_provisioning(self):
        # c=[2,1], w=[100,100], theta=70 -> w/c = [50, 100] -> [F, T]
        rate = under_provisioning_rate(np.array([2, 1]), np.array([100.0, 100.0]), 70.0)
        assert rate == pytest.approx(0.5)

    def test_over_provisioning(self):
        # actual=100, theta=70 -> min nodes = floor(100/70)+1 = 2
        assert over_provisioning_rate(np.array([3]), np.array([100.0]), 70.0) == 1.0
        assert over_provisioning_rate(np.array([2]), np.array([100.0]), 70.0) == 0.0

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            under_provisioning_rate(np.array([1, 2]), np.array([1.0]), 70.0)
