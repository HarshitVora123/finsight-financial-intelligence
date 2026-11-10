"""Point-in-time customer feature aggregation from synthetic source tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_FEATURES = [
    "monthly_income", "credit_limit", "credit_utilization_ratio",
    "existing_debt", "active_credit_accounts", "savings_balance",
    "monthly_debt_payment", "debt_to_income_ratio", "monthly_spending",
    "expense_to_income_ratio", "monthly_savings_ratio", "spending_volatility",
    "income_stability_score", "missed_payment_count", "recent_delinquency_frequency",
    "payment_consistency_score", "emergency_savings_coverage", "month_over_month_spend_change",
]


def build_customer_features(customers: pd.DataFrame, transactions: pd.DataFrame,
                            repayments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate history available before the synthetic next-month outcome."""
    required = {"customer_id", "monthly_income", "employment_months", "credit_limit",
                "current_credit_balance", "existing_debt", "active_credit_accounts",
                "credit_history_months", "savings_balance", "minimum_monthly_payment"}
    missing = required.difference(customers.columns)
    if missing:
        raise ValueError(f"Customer profile is missing required columns: {sorted(missing)}")
    if customers.customer_id.duplicated().any():
        raise ValueError("customer_id must be unique in customer profiles")

    tx = transactions.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
    if tx.transaction_date.isna().any():
        raise ValueError("transaction_date contains invalid values")
    tx["month"] = tx.transaction_date.dt.to_period("M").astype(str)
    tx["is_expense"] = tx.transaction_type.eq("debit")
    tx["is_income"] = tx.transaction_type.eq("credit")
    tx_month = tx.groupby(["customer_id", "month"], as_index=False).agg(
        income=("amount", lambda x: float(x[tx.loc[x.index, "is_income"]].sum())),
        expenses=("amount", lambda x: float(x[tx.loc[x.index, "is_expense"]].sum())),
    )
    monthly = tx_month.groupby("customer_id").agg(
        avg_income_tx=("income", "mean"), income_tx_sd=("income", "std"),
        monthly_spending=("expenses", "mean"), spending_sd=("expenses", "std"),
        months_observed=("month", "nunique"),
    )
    monthly["spending_volatility"] = monthly.spending_sd.fillna(0) / monthly.monthly_spending.clip(lower=1)
    monthly["income_stability_score"] = (1 - monthly.income_tx_sd.fillna(0) / monthly.avg_income_tx.clip(lower=1)).clip(0, 1)
    spend_by_month = tx_month.pivot(index="customer_id", columns="month", values="expenses").sort_index(axis=1)
    if spend_by_month.shape[1] >= 2:
        monthly["month_over_month_spend_change"] = (
            spend_by_month.iloc[:, -1] - spend_by_month.iloc[:, -2]
        ) / spend_by_month.iloc[:, -2].clip(lower=1)
    else:
        monthly["month_over_month_spend_change"] = 0.0

    rep = repayments.copy()
    rep["period"] = rep.period.astype(str)
    rep = rep.sort_values("period")
    rep_summary = rep.groupby("customer_id").agg(
        missed_payment_count=("on_time", lambda x: int((1 - x.astype(int)).sum())),
        payment_consistency_score=("on_time", "mean"),
    )
    recent = rep.groupby("customer_id").tail(3).groupby("customer_id").agg(
        recent_delinquency_frequency=("on_time", lambda x: float((1 - x.astype(int)).mean()))
    )
    out = customers.merge(monthly, left_on="customer_id", right_index=True, how="left")
    out = out.merge(rep_summary, left_on="customer_id", right_index=True, how="left")
    out = out.merge(recent, left_on="customer_id", right_index=True, how="left")
    out["monthly_debt_payment"] = out.minimum_monthly_payment.astype(float)
    out["credit_utilization_ratio"] = out.current_credit_balance / out.credit_limit.clip(lower=1)
    out["debt_to_income_ratio"] = out.monthly_debt_payment / out.monthly_income.clip(lower=1)
    out["expense_to_income_ratio"] = out.monthly_spending / out.monthly_income.clip(lower=1)
    out["monthly_savings_ratio"] = 1 - out.expense_to_income_ratio
    out["emergency_savings_coverage"] = out.savings_balance / out.monthly_spending.clip(lower=1)
    out["recent_delinquency_frequency"] = out.recent_delinquency_frequency.fillna(0)
    out["missed_payment_count"] = out.missed_payment_count.fillna(0)
    out["payment_consistency_score"] = out.payment_consistency_score.fillna(1)
    out[MODEL_FEATURES] = out[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    # Keep label and IDs as metadata; downstream modeling explicitly selects MODEL_FEATURES.
    return out


def feature_dictionary() -> dict[str, str]:
    return {
        "monthly_income": "Profile monthly income; synthetic or user-entered amount.",
        "credit_limit": "Total revolving credit limit.",
        "credit_utilization_ratio": "Current revolving balance divided by credit limit.",
        "existing_debt": "Total outstanding principal across active debts.",
        "active_credit_accounts": "Number of open credit accounts.",
        "savings_balance": "Reported liquid savings balance.",
        "monthly_debt_payment": "Current required monthly debt payment.",
        "debt_to_income_ratio": "Required monthly debt payment divided by monthly gross income.",
        "monthly_spending": "Mean monthly debit transaction outflow over the observed window.",
        "expense_to_income_ratio": "Mean monthly debit outflow divided by monthly income.",
        "monthly_savings_ratio": "One minus the observed expense-to-income ratio; may be negative.",
        "spending_volatility": "Standard deviation of monthly debit outflow divided by its mean.",
        "income_stability_score": "One minus income deposit coefficient of variation, clipped to [0, 1].",
        "missed_payment_count": "Number of late or missed payments in observed repayment history.",
        "recent_delinquency_frequency": "Missed payment share in the latest three observed periods.",
        "payment_consistency_score": "Share of observed payments made on time.",
        "emergency_savings_coverage": "Liquid savings divided by mean monthly spending, in months.",
        "month_over_month_spend_change": "Relative change in debit outflow between the latest two months.",
    }


def source_table_dictionary() -> dict[str, dict[str, str]]:
    """Definitions for source fields stored in the relational demo tables."""
    return {
        "customers": {
            "customer_id": "Synthetic customer key.", "monthly_income": "Monthly income in generated currency units.",
            "employment_months": "Employment tenure in months.", "credit_limit": "Total revolving credit limit.",
            "current_credit_balance": "Current revolving balance.", "existing_debt": "Total debt principal across modeled accounts.",
            "active_credit_accounts": "Number of active modeled credit accounts.", "credit_history_months": "Observed history length in months.",
            "savings_balance": "Liquid savings balance.", "minimum_monthly_payment": "Aggregate required debt minimum per month.",
            "monthly_budget": "Stated modeled monthly debt-repayment budget.",
            "simulated_missed_payment_next_month": "Synthetic binary next-period target; never included as a model input.",
            "data_origin": "Provenance label (synthetic_demo).",
        },
        "transactions": {
            "transaction_id": "Synthetic transaction key.", "customer_id": "Foreign key to customers.",
            "transaction_date": "Transaction date; generated monthly history.", "category": "Normalized expense or income category.",
            "amount": "Non-negative transaction amount in generated currency units.",
            "transaction_type": "credit for income or debit for outflow.", "merchant_category": "Synthetic merchant grouping.",
        },
        "debts": {
            "debt_id": "Synthetic debt account key.", "customer_id": "Foreign key to customers.",
            "principal": "Current outstanding principal.", "annual_rate": "Annual percentage rate as decimal (0.18 means 18%).",
            "minimum_payment": "Required minimum amount per month.", "due_day": "Synthetic day of month from 1 to 31.",
            "debt_type": "Synthetic account category.",
        },
        "repayments": {
            "repayment_id": "Synthetic repayment observation key.", "customer_id": "Foreign key to customers.",
            "period": "Observed repayment month in YYYY-MM format.", "due_date": "Synthetic due date.",
            "amount_due": "Synthetic scheduled amount due.", "amount_paid": "Synthetic amount paid; zero for generated miss events.",
            "on_time": "1 for generated on-time payment, 0 for late or missed.",
        },
        "engineered_model_features": feature_dictionary(),
    }
