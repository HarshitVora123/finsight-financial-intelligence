"""Lightweight input drift, missingness, and delayed-outcome monitoring."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_share = np.clip(ref_counts / len(ref), 1e-6, None)
    cur_share = np.clip(cur_counts / len(cur), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def monitor_batch(reference: pd.DataFrame, current: pd.DataFrame, predictions: pd.Series | np.ndarray,
                  output_path: str | Path, actual_labels: pd.Series | np.ndarray | None = None,
                  prediction_latency_ms: float | None = None) -> dict:
    shared = sorted(set(reference.columns) & set(current.columns))
    drift = {column: population_stability_index(reference[column], current[column])
             for column in shared if pd.api.types.is_numeric_dtype(reference[column])}
    pred = np.asarray(predictions, dtype=float)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "n_current": int(len(current)), "missing_rate": {str(c): float(current[c].isna().mean()) for c in current},
              "prediction_summary": {"mean": float(np.mean(pred)), "std": float(np.std(pred)),
                                     "p05": float(np.quantile(pred, .05)), "p95": float(np.quantile(pred, .95))},
              "feature_psi": drift, "prediction_latency_ms": prediction_latency_ms,
              "drift_flag_psi_gt_0_2": [c for c, v in drift.items() if np.isfinite(v) and v > .2],
              "delayed_label_metrics": None}
    if actual_labels is not None:
        y = np.asarray(actual_labels, dtype=int)
        report["delayed_label_metrics"] = {"pr_auc": float(average_precision_score(y, pred)),
                                            "roc_auc": float(roc_auc_score(y, pred)),
                                            "brier_score": float(brier_score_loss(y, pred))}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, allow_nan=False) + "\n")
    return report
