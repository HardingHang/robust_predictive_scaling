# Robust Auto-Scaling with Probabilistic Workload Forecasting for Cloud Databases

This repository contains the core code implementation of the research paper
"Robust Auto-Scaling with Probabilistic Workload Forecasting for Cloud
Databases" (ICDE 2024). The project addresses auto-scaling of cloud databases
under workload uncertainty: probabilistic forecasting models generate quantile
forecasts of future workloads, and the auto-scaling manager formulates
resource scaling as a robust optimization problem, enabling strategies with
customizable levels of robustness and adaptive conservatism control.

## Folder Structure

```
.
├── data               # workload traces (10-minute granularity)
├── forecaster         # probabilistic workload forecasting models
├── scaler_manager     # auto-scaling strategies
├── metrics            # forecasting / scaling evaluation metrics
├── tests              # unit tests and the end-to-end demo
└── README.md
```

## Requirements

- `GluonTS` (torch backend)
- `scipy`
- `numpy`, `pandas`, `matplotlib`

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install "gluonts[torch]==0.17.0" scipy matplotlib pytest
```

## Usage

To run, you need workload trace data prepared (`data/`). For easy usage,
refer to the code in `tests/`:

```bash
# unit tests for metrics and scaling strategies
.venv/bin/python -m pytest tests/ -q

# end-to-end demo on a synthetic workload trace:
# data -> quantile forecasting -> robust/adaptive scaling -> evaluation
.venv/bin/python -m tests.demo_end_to_end --model both --epochs 5
```

Real workload traces (e.g., Alibaba / Google cluster data, aggregated at
10-minute intervals) can be plugged in through the `TraceLoader` interface
in `data/workload.py`.

## Method Overview

| Component | Description |
|---|---|
| `forecaster/base.py` | Workload forecasting & quantile workload forecasting interface (Def. 1-2) |
| `forecaster/deepar.py` | DeepAR: learns a parametric distribution (Student-t); quantiles obtained by sampling |
| `forecaster/tft.py` | TFT: learns a pre-specified grid of quantiles, trained with quantile loss |
| `metrics/forecasting.py` | wQL[τ], Coverage[τ], mean wQL, MSE |
| `metrics/scaling.py` | Under-/Over-Provisioning Rate |
| `scaler_manager/reactive.py` | Reactive baselines: max / exponentially-weighted average over a recent window |
| `scaler_manager/point.py` | Point-forecast scaler (0.5-quantile; non-robust) |
| `scaler_manager/robust.py` | Robust scaler: allocate from a fixed quantile level τ (Def. 4, Eq. 4-6) |
| `scaler_manager/adaptive.py` | Uncertainty-aware adaptive scaler (Def. 5, Eq. 7-8, Alg. 1) |
| `tests/demo_end_to_end.py` | End-to-end workflow: forecaster → scaling manager → evaluation |

## Implementation Notes

- The robust scaling objective (Eq. 6) decomposes across time steps and
  admits the closed-form optimum `c_t = floor(ŵ^τ_t / θ) + 1`, the smallest
  node count satisfying `ŵ^τ_t / c_t < θ`; the scalers use this directly.
- The uncertainty level U (Eq. 8) is computed as the pinball-style deviation
  of each quantile forecast from the median (0.5-quantile) forecast; for
  parametric forecasters (DeepAR) the predicted standard deviation can be
  used as U instead.
- `tests/demo_end_to_end.py` ships with a synthetic workload generator
  (daily/weekly seasonality, spikes, heteroscedastic noise) so the full
  pipeline runs without downloading the cluster traces.
