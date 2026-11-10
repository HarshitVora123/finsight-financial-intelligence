"""Optional loader for UCI Default of Credit Card Clients (dataset 350)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROTECTED_OR_PROXY = {"SEX", "AGE", "MARRIAGE", "EDUCATION"}


def load_uci_credit_default(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load a cached UCI CSV or fetch dataset 350 with ``ucimlrepo``.

    Demographic fields and the record identifier are excluded from model features.
    The UCI label is next-month default, not a real-world consumer credit decision.
    """
    frame = None
    if path is not None and Path(path).exists():
        frame = pd.read_csv(path)
    if frame is not None:
        frame.columns = [str(c).strip().upper() for c in frame.columns]
        target_candidates = [c for c in frame.columns if "DEFAULT" in c and "NEXT" in c]
        if not target_candidates:
            target_candidates = [c for c in frame.columns if c in {"Y", "TARGET", "DEFAULT"}]
        if not target_candidates:
            raise ValueError("Could not identify the UCI next-month default target in cached CSV")
        target = frame.pop(target_candidates[0]).astype(int).rename("default_payment_next_month")
        frame = frame.drop(columns=[c for c in ["ID", *PROTECTED_OR_PROXY] if c in frame.columns])
        return frame.apply(pd.to_numeric, errors="coerce"), target
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError("Install ucimlrepo or pass a cached UCI CSV path to load dataset 350") from exc
    dataset = fetch_ucirepo(id=350)
    features = dataset.data.features.copy()
    target_frame = dataset.data.targets.copy()
    features.columns = [str(c).strip().upper() for c in features.columns]
    y = target_frame.iloc[:, 0].astype(int).rename("default_payment_next_month")
    features = features.drop(columns=[c for c in ["ID", *PROTECTED_OR_PROXY] if c in features.columns])
    return features.apply(pd.to_numeric, errors="coerce"), y
