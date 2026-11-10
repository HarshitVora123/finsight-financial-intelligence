"""Schema and quality checks for BrightPath source tables."""

from __future__ import annotations

import pandas as pd


SCHEMAS = {
    "customers": {"customer_id", "monthly_income", "credit_limit", "current_credit_balance", "existing_debt"},
    "transactions": {"transaction_id", "customer_id", "transaction_date", "category", "amount", "transaction_type"},
    "debts": {"debt_id", "customer_id", "principal", "annual_rate", "minimum_payment", "due_day"},
    "repayments": {"repayment_id", "customer_id", "period", "amount_due", "amount_paid", "on_time"},
}


def quality_report(name: str, frame: pd.DataFrame) -> dict:
    if name not in SCHEMAS:
        raise ValueError(f"Unknown dataset schema: {name}")
    missing_columns = sorted(SCHEMAS[name].difference(frame.columns))
    numeric = frame.select_dtypes(include="number")
    outliers = {}
    for column in numeric:
        q1, q3 = numeric[column].quantile([0.25, 0.75])
        spread = q3 - q1
        outliers[column] = int(((numeric[column] < q1 - 1.5 * spread) | (numeric[column] > q3 + 1.5 * spread)).sum())
    key = {"customers": "customer_id", "transactions": "transaction_id", "debts": "debt_id", "repayments": "repayment_id"}[name]
    return {
        "dataset": name, "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "missing_required_columns": missing_columns, "duplicate_key_rows": int(frame[key].duplicated().sum()) if key in frame else None,
        "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items() if v},
        "numeric_iqr_outliers": outliers,
    }


def validate_bundle(bundle) -> list[dict]:
    reports = [quality_report(name, getattr(bundle, name)) for name in SCHEMAS]
    for report in reports:
        if report["missing_required_columns"]:
            raise ValueError(f"{report['dataset']} lacks required columns: {report['missing_required_columns']}")
        if report["duplicate_key_rows"]:
            raise ValueError(f"{report['dataset']} contains duplicate primary keys")
    customers = bundle.customers
    if (customers.monthly_income <= 0).any() or (customers.credit_limit <= 0).any():
        raise ValueError("Monthly income and credit limit must be positive")
    if (customers.current_credit_balance < 0).any() or (customers.current_credit_balance > customers.credit_limit * 1.5).any():
        raise ValueError("Current credit balance is outside the accepted consistency range")
    if (customers.existing_debt < customers.current_credit_balance).any():
        raise ValueError("Existing debt cannot be below the revolving balance")
    if (bundle.debts[["principal", "minimum_payment"]] < 0).any().any():
        raise ValueError("Debt principal and minimum payment cannot be negative")
    if (bundle.debts.annual_rate < 0).any() or not bundle.debts.due_day.between(1, 31).all():
        raise ValueError("Debt annual rates and due days are invalid")
    if (bundle.transactions.amount < 0).any() or (bundle.repayments[["amount_due", "amount_paid"]] < 0).any().any():
        raise ValueError("Transaction and repayment amounts cannot be negative")
    if not bundle.repayments.on_time.isin([0, 1]).all():
        raise ValueError("Repayment on_time must contain only 0/1 values")
    if not bundle.transactions.transaction_type.isin(["credit", "debit"]).all():
        raise ValueError("transaction_type must be credit or debit")
    customer_ids = set(customers.customer_id.astype(str))
    for name in ("transactions", "debts", "repayments"):
        if not set(getattr(bundle, name).customer_id.astype(str)).issubset(customer_ids):
            raise ValueError(f"{name} contains customer IDs missing from customer profiles")
    return reports


def normalize_categories(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    out = frame.copy()
    out[column] = out[column].astype("string").str.strip().str.lower().replace({"": "unknown", "nan": "unknown"})
    return out
