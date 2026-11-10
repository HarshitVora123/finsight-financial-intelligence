"""Deterministic, clearly synthetic household finance data for the demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticBundle:
    customers: pd.DataFrame
    transactions: pd.DataFrame
    debts: pd.DataFrame
    repayments: pd.DataFrame


def generate_synthetic_data(n_customers: int = 1_200, seed: int = 42) -> SyntheticBundle:
    """Generate linked profile, transaction, debt and historical repayment tables.

    The next-month missed-payment label is a simulated outcome. It is created from
    pre-outcome affordability and behaviour signals and is never used as a feature.
    """
    if n_customers < 100:
        raise ValueError("n_customers must be at least 100 for the demo dataset")
    rng = np.random.default_rng(seed)
    ids = np.array([f"BP{i:06d}" for i in range(1, n_customers + 1)])
    income = np.clip(rng.lognormal(np.log(4_600), 0.48, n_customers), 1_250, 18_000)
    employment_months = np.clip(rng.gamma(3.2, 24, n_customers), 1, 420).astype(int)
    limit = np.clip(income * rng.lognormal(1.05, 0.42, n_customers), 1_000, 65_000)
    utilization = np.clip(rng.beta(2.0, 3.5, n_customers) + rng.normal(0, 0.08, n_customers), 0.01, 0.99)
    balance = limit * utilization
    accounts = np.clip(rng.poisson(2.2, n_customers) + 1, 1, 8)
    history = np.clip(rng.gamma(3.0, 25, n_customers), 3, 300).astype(int)
    savings = np.clip(income * rng.lognormal(0.55, 1.05, n_customers), 0, 120_000)
    expense_ratio = np.clip(rng.normal(0.69 + 0.11 * (utilization > 0.7), 0.13, n_customers), 0.38, 1.12)
    living_expenses = income * expense_ratio
    existing_debt = np.clip(balance + income * rng.lognormal(0.7, 0.7, n_customers), balance, 180_000)
    debt_payment = np.clip(existing_debt * rng.uniform(0.018, 0.055, n_customers), 25, income * 0.42)
    budget = np.maximum(debt_payment, np.clip(income - living_expenses, 0, income * 0.42))

    # Six months of transaction history. Categories and merchant groups are synthetic.
    months = pd.date_range(end=pd.Timestamp("2026-10-01"), periods=6, freq="MS")
    categories = ["housing", "groceries", "utilities", "transport", "health", "discretionary", "debt_payment"]
    base_weights = np.array([0.31, 0.19, 0.08, 0.09, 0.06, 0.17, 0.10])
    tx_rows: list[dict] = []
    monthly_spend = np.zeros((n_customers, len(months)))
    for j, month in enumerate(months):
        month_factor = rng.normal(1.0, 0.075, n_customers)
        spend = living_expenses * month_factor
        monthly_spend[:, j] = spend
        # An income deposit plus seven categorized outflows per customer/month.
        for i, cid in enumerate(ids):
            tx_rows.append({"transaction_id": f"T{j:02d}-{i:06d}-I", "customer_id": cid,
                            "transaction_date": month + pd.Timedelta(days=1), "category": "income",
                            "amount": round(float(income[i] * rng.normal(1.0, 0.025)), 2),
                            "transaction_type": "credit", "merchant_category": "payroll"})
            alloc = rng.dirichlet(base_weights * 75) * (spend[i] + debt_payment[i])
            for k, cat in enumerate(categories):
                tx_rows.append({"transaction_id": f"T{j:02d}-{i:06d}-{k}", "customer_id": cid,
                                "transaction_date": month + pd.Timedelta(days=int(rng.integers(2, 27))),
                                "category": cat, "amount": round(float(alloc[k]), 2),
                                "transaction_type": "debit", "merchant_category": cat})
    transactions = pd.DataFrame(tx_rows)

    spending_cv = monthly_spend.std(axis=1) / np.maximum(monthly_spend.mean(axis=1), 1)
    repayment_burden = debt_payment / np.maximum(income, 1)
    dti = debt_payment / np.maximum(income, 1)
    cashflow_gap = np.maximum(expense_ratio - 0.82, 0)
    # Historical and next-month outcomes come from an explicitly simulated scenario.
    logits = (-3.15 + 2.3 * utilization + 4.0 * dti + 1.55 * spending_cv
              + 1.1 * cashflow_gap + 2.0 * (savings < income).astype(float))
    risk_prob = 1 / (1 + np.exp(-logits))
    prior_prob = np.clip(risk_prob * 0.82, 0.02, 0.65)
    repayment_rows: list[dict] = []
    repayment_start = rng.integers(0, len(months), size=n_customers)
    for j, month in enumerate(months):
        missed = rng.binomial(1, prior_prob, n_customers)
        for i, cid in enumerate(ids):
            if j < repayment_start[i]:
                continue
            repayment_rows.append({"repayment_id": f"R{j:02d}-{i:06d}", "customer_id": cid,
                                   "period": month.strftime("%Y-%m"), "due_date": month + pd.Timedelta(days=25),
                                   "amount_due": round(float(debt_payment[i]), 2),
                                   "amount_paid": round(float(debt_payment[i] * (0.0 if missed[i] else rng.uniform(0.92, 1.08))), 2),
                                   "on_time": int(not missed[i])})
    repayments = pd.DataFrame(repayment_rows)
    future_missed = rng.binomial(1, risk_prob)
    customers = pd.DataFrame({
        "customer_id": ids, "monthly_income": income.round(2), "employment_months": employment_months,
        "credit_limit": limit.round(2), "current_credit_balance": balance.round(2),
        "existing_debt": existing_debt.round(2), "active_credit_accounts": accounts,
        "credit_history_months": history, "savings_balance": savings.round(2),
        "minimum_monthly_payment": debt_payment.round(2), "monthly_budget": budget.round(2),
        "simulated_missed_payment_next_month": future_missed,
        "data_origin": "synthetic_demo",
    })

    debt_rows: list[dict] = []
    for i, cid in enumerate(ids):
        count = int(np.clip(accounts[i], 1, 4))
        weights = rng.dirichlet(np.ones(count) * 2)
        principals = existing_debt[i] * weights
        rates = rng.uniform(0.045, 0.32, count)
        mins = np.maximum(principals * rng.uniform(0.02, 0.045, count), 15)
        for k in range(count):
            debt_rows.append({"debt_id": f"{cid}-D{k+1}", "customer_id": cid,
                              "principal": round(float(principals[k]), 2), "annual_rate": round(float(rates[k]), 5),
                              "minimum_payment": round(float(mins[k]), 2), "due_day": int(rng.integers(1, 29)),
                              "debt_type": ["credit_card", "personal_loan", "retail_credit"][k % 3]})
    debts = pd.DataFrame(debt_rows)
    return SyntheticBundle(customers, transactions, debts, repayments)
