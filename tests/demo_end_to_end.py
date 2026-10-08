"""End-to-end demo of the robust predictive auto-scaling workflow.

Pipeline (paper Figure 2):
    workload data -> Probabilistic Workload Forecaster (quantile forecasts)
    -> Robust Auto-Scaling Manager (fixed-quantile / uncertainty-aware
    adaptive strategies) -> under-/over-provisioning evaluation.

Runs on a synthetic workload trace so the full pipeline can be exercised
without downloading the cluster traces, and sanity-checks the expected
behavior of the strategies (robustness increasing with the quantile level;
the adaptive strategy bounded by its two fixed-quantile endpoints).

Run from the project root:
    .venv/bin/python -m tests.demo_end_to_end --model both --epochs 5
"""

from __future__ import annotations

import argparse

import numpy as np

from data.workload import generate_synthetic_workload, train_test_split
from metrics.forecasting import (
    coverage,
    mean_weighted_quantile_loss,
    mse_from_quantiles,
    weighted_quantile_loss,
)
from metrics.scaling import over_provisioning_rate, under_provisioning_rate
from scaler_manager.adaptive import AdaptiveScaler
from scaler_manager.point import PointScaler
from scaler_manager.reactive import ReactiveAvg, ReactiveMax
from scaler_manager.robust import RobustScaler

THETA = 70.0  # per-node workload threshold (paper: e.g. percentage CPU)
CONTEXT_LENGTH = 72  # 12 hours at 10-minute intervals
PREDICTION_LENGTH = 72
SCALING_QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9)  # paper Fig. 9/10 range
ADAPTIVE_TAU1, ADAPTIVE_TAU2 = 0.5, 0.9  # paper Algorithm 1 pair


def build_forecaster(name: str, epochs: int):
    from forecaster.base import QuantileForecaster  # noqa: F401
    if name == "deepar":
        from forecaster.deepar import DeepARForecaster

        return DeepARForecaster(
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
            epochs=epochs,
        )
    if name == "tft":
        from forecaster.tft import TFTForecaster

        return TFTForecaster(
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
            epochs=epochs,
        )
    raise ValueError(name)


def evaluate_forecasts(all_quantiles, all_actuals):
    """Forecasting metrics (paper Section IV-B.1), averaged over windows."""
    taus = sorted(all_quantiles[0].keys())
    rows = {}
    for tau in taus:
        rows[f"wQL[{tau}]"] = np.mean(
            [weighted_quantile_loss(y, q[tau], tau) for q, y in zip(all_quantiles, all_actuals)]
        )
        rows[f"cov[{tau}]"] = np.mean(
            [coverage(y, q[tau]) for q, y in zip(all_quantiles, all_actuals)]
        )
    rows["mean wQL"] = np.mean(
        [mean_weighted_quantile_loss(y, q) for q, y in zip(all_quantiles, all_actuals)]
    )
    rows["MSE"] = np.mean(
        [mse_from_quantiles(y, q) for q, y in zip(all_quantiles, all_actuals)]
    )
    return rows


def run_model(name: str, epochs: int, n_days: int, n_windows: int, stride: int):
    print(f"\n{'=' * 64}\nModel: {name.upper()}\n{'=' * 64}")
    series = generate_synthetic_workload(n_days=n_days)
    train, test_region = train_test_split(series, CONTEXT_LENGTH, PREDICTION_LENGTH)

    forecaster = build_forecaster(name, epochs)
    print(f"training on {len(train)} steps ({n_days} days synthetic, 10-min granularity)...")
    forecaster.fit(train)

    # Rolling forecasting windows over the test region (paper: decisions for
    # the future 72 steps / 12 hours).
    starts = [
        i
        for i in range(0, len(test_region) - CONTEXT_LENGTH - PREDICTION_LENGTH + 1, stride)
    ][:n_windows]
    assert starts, "test region too short for one forecasting window"

    all_quantiles, all_actuals, all_stds = [], [], []
    allocations: dict[str, list[np.ndarray]] = {}
    scalers = {
        "point (tau=0.5)": PointScaler(theta=THETA),
        **{f"robust-{tau}": RobustScaler(theta=THETA, tau=tau) for tau in SCALING_QUANTILES},
    }

    for s in starts:
        context = test_region[s : s + CONTEXT_LENGTH]
        actual = test_region[
            s + CONTEXT_LENGTH : s + CONTEXT_LENGTH + PREDICTION_LENGTH
        ]
        quantiles = forecaster.predict_quantiles(context)
        all_quantiles.append(quantiles)
        all_actuals.append(actual)
        std = forecaster.predict_std(context)
        all_stds.append(std)
        for label, scaler in scalers.items():
            allocations.setdefault(label, []).append(scaler.allocate(quantiles))

    # Adaptive scaler: rho_tau set from the median uncertainty over the test
    # windows (paper Section III-C.2: choose the threshold from the model's
    # historical uncertainty-accuracy relationship).
    adaptive_allocs = []
    rho_candidates = []
    probe = AdaptiveScaler(theta=THETA, tau1=ADAPTIVE_TAU1, tau2=ADAPTIVE_TAU2, rho_tau=0.0)
    for quantiles, std in zip(all_quantiles, all_stds):
        probe.std_forecast = std
        rho_candidates.append(probe._uncertainties(quantiles, PREDICTION_LENGTH))
    rho_tau = float(np.median(np.concatenate(rho_candidates)))
    for quantiles, std in zip(all_quantiles, all_stds):
        adaptive = AdaptiveScaler(
            theta=THETA, tau1=ADAPTIVE_TAU1, tau2=ADAPTIVE_TAU2,
            rho_tau=rho_tau, std_forecast=std,
        )
        adaptive_allocs.append(adaptive.allocate(quantiles))
    allocations[f"adaptive({ADAPTIVE_TAU1},{ADAPTIVE_TAU2})"] = adaptive_allocs

    # Reactive baselines over the same periods (step-by-step simulation).
    flat_actual = np.concatenate(all_actuals)
    offsets = np.arange(len(starts)) * PREDICTION_LENGTH
    for label, scaler in {
        "reactive-max": ReactiveMax(theta=THETA, window=6),
        "reactive-avg": ReactiveAvg(theta=THETA, window=6, half_life=6.0),
    }.items():
        alloc = scaler.allocate_series(flat_actual)
        allocations[label] = [alloc[o : o + PREDICTION_LENGTH] for o in offsets]

    # --- report ---
    fc = evaluate_forecasts(all_quantiles, all_actuals)
    print("\nForecasting metrics (avg over windows):")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in fc.items()))

    print(f"\nScaling results (theta={THETA}, adaptive rho_tau={rho_tau:.3f}):")
    print(f"  {'strategy':<24}{'under-prov':>12}{'over-prov':>12}{'avg nodes':>11}")
    rates = {}
    for label, allocs in allocations.items():
        alloc = np.concatenate(allocs)
        under = under_provisioning_rate(alloc, flat_actual, THETA)
        over = over_provisioning_rate(alloc, flat_actual, THETA)
        rates[label] = (under, over)
        print(f"  {label:<24}{under:>12.3f}{over:>12.3f}{alloc.mean():>11.2f}")

    # --- behavioral sanity checks ---
    eps = 1.0 / len(flat_actual)
    robust_taus = SCALING_QUANTILES
    unders = [rates[f"robust-{t}"][0] for t in robust_taus]
    overs = [rates[f"robust-{t}"][1] for t in robust_taus]
    monotone_under = all(a <= b + eps for a, b in zip(unders[1:], unders))
    monotone_over = all(a + eps >= b for a, b in zip(overs[1:], overs))
    robust_beats_point = rates["robust-0.9"][0] <= rates["point (tau=0.5)"][0] + eps

    adaptive_alloc = np.concatenate(allocations[f"adaptive({ADAPTIVE_TAU1},{ADAPTIVE_TAU2})"])
    low = np.concatenate(allocations[f"robust-{ADAPTIVE_TAU1}"])
    high = np.concatenate(allocations[f"robust-{ADAPTIVE_TAU2}"])
    adaptive_bounded = bool(np.all(low <= adaptive_alloc) and np.all(adaptive_alloc <= high))

    print("\nBehavioral checks:")
    print(f"  under-provisioning non-increasing in tau: {monotone_under} {unders}")
    print(f"  over-provisioning non-decreasing in tau:  {monotone_over} {overs}")
    print(f"  robust-0.9 <= point under-provisioning:   {robust_beats_point}")
    print(f"  adaptive bounded by fixed-tau endpoints:  {adaptive_bounded}")

    ok = monotone_under and monotone_over and robust_beats_point and adaptive_bounded
    print(f"\n[{name}] FUNCTIONAL CHECKS {'PASSED' if ok else 'FAILED'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["deepar", "tft", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--stride", type=int, default=24)
    args = parser.parse_args()

    models = ["deepar", "tft"] if args.model == "both" else [args.model]
    results = {
        m: run_model(m, args.epochs, args.days, args.windows, args.stride) for m in models
    }
    if not all(results.values()):
        raise SystemExit(1)
    print("\nALL FUNCTIONAL CHECKS PASSED")


if __name__ == "__main__":
    main()
