"""Optional loader for UCI Default of Credit Card Clients (dataset 350)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROTECTED_OR_PROXY = {"SEX", "AGE", "MARRIAGE", "EDUCATION"}
UCI_FEATURE_NAMES = {
    **{f"X{i}": name for i, name in enumerate([
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ], start=1)},
}


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().upper() for column in result.columns]
    return result.rename(columns=UCI_FEATURE_NAMES)


def load_uci_credit_default_with_audit(
        path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load UCI model features, the target, and held-out fairness audit columns.

    UCI's API exposes generic X1...X23 names, so map them before excluding
    demographics. Audit columns are returned separately and must never be model
    predictors. The target is historical next-month default, not a current
    consumer credit decision.
    """
    frame = None
    if path is not None and Path(path).exists():
        frame = pd.read_csv(path)
    if frame is not None:
        frame = _canonicalize_columns(frame)
        target_candidates = [c for c in frame.columns if "DEFAULT" in c and "NEXT" in c]
        if not target_candidates:
            target_candidates = [c for c in frame.columns if c in {"Y", "TARGET", "DEFAULT"}]
        if not target_candidates:
            raise ValueError("Could not identify the UCI next-month default target in cached CSV")
        target = frame.pop(target_candidates[0]).astype(int).rename("default_payment_next_month")
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise RuntimeError("Install ucimlrepo or pass a cached UCI CSV path to load dataset 350") from exc
        dataset = fetch_ucirepo(id=350)
        frame = _canonicalize_columns(dataset.data.features)
        target_frame = _canonicalize_columns(dataset.data.targets)
        target = target_frame.iloc[:, 0].astype(int).rename("default_payment_next_month")

    audit_columns = [column for column in sorted(PROTECTED_OR_PROXY) if column in frame.columns]
    audit = frame[audit_columns].apply(pd.to_numeric, errors="coerce").copy()
    features = frame.drop(columns=[column for column in ["ID", *PROTECTED_OR_PROXY] if column in frame.columns])
    return features.apply(pd.to_numeric, errors="coerce"), target, audit


def load_uci_credit_default(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset 350 while excluding ID and demographic/proxy columns."""
    features, target, _ = load_uci_credit_default_with_audit(path)
    return features, target
