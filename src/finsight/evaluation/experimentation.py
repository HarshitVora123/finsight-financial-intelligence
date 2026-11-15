"""Scenario-based A/B experiment simulator, explicitly not causal evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def simulate_experiment(features: pd.DataFrame, output_dir: str | Path,
                        seed: int = 42, assumed_max_absolute_uplift: float = .04) -> dict:
    if len(features) < 100:
        raise ValueError("At least 100 customers are required for the illustrative experiment")
    required = {"credit_utilization_ratio", "debt_to_income_ratio", "spending_volatility",
                "monthly_budget", "existing_debt"}
    if not required.issubset(features.columns):
        raise ValueError(f"Experiment data missing: {sorted(required.difference(features.columns))}")
    rng = np.random.default_rng(seed)
    frame = features.reset_index(drop=True).copy()
    n = len(frame)
    risk_logit = (-3.1 + 2.0 * frame.credit_utilization_ratio.to_numpy()
                  + 4.0 * frame.debt_to_income_ratio.to_numpy()
                  + 1.25 * frame.spending_volatility.to_numpy())
    baseline_miss = 1 / (1 + np.exp(-risk_logit))
    assignment = np.zeros(n, dtype=int)
    assignment[rng.permutation(n)[:n // 2]] = 1
    fit = np.clip(.35 + .45 * (frame.monthly_budget.to_numpy() > 0)
                  + .2 * (frame.debt_to_income_ratio.to_numpy() < .25), 0, 1)
    acceptance_probability = np.where(assignment == 1, .30 + .4 * fit, .25)
    accepted = rng.binomial(1, acceptance_probability)
    assigned_uplift = np.where(assignment == 1, assumed_max_absolute_uplift, .01) * fit * accepted
    miss_prob = np.clip(baseline_miss - assigned_uplift, .01, .95)
    missed = rng.binomial(1, miss_prob)
    on_time = 1 - missed
    budget = frame.monthly_budget.to_numpy()
    debt = frame.existing_debt.to_numpy()
    control_reduction = np.minimum(debt, budget * rng.uniform(.55, .78, n) * on_time)
    # Accepted personalized action adds a simulated extra principal reduction.
    extra = np.minimum(np.maximum(debt - control_reduction, 0),
                       budget * np.where(assignment == 1, .12, .04) * accepted)
    debt_reduction = control_reduction + extra
    interest_saved = extra * .12 * 3 / 12  # assumes 12% annual APR and three months of earlier principal reduction

    treatment = assignment == 1
    control = ~treatment
    n_treat, n_ctrl = int(treatment.sum()), int(control.sum())
    rate_treat, rate_ctrl = float(on_time[treatment].mean()), float(on_time[control].mean())
    diff = rate_treat - rate_ctrl
    pooled = (on_time[treatment].sum() + on_time[control].sum()) / n
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_treat + 1 / n_ctrl))
    z = diff / se if se > 0 else 0.0
    pvalue = float(2 * stats.norm.sf(abs(z)))
    ci = [float(diff - 1.96 * se), float(diff + 1.96 * se)]
    cohens_h = float(2 * np.arcsin(np.sqrt(rate_treat)) - 2 * np.arcsin(np.sqrt(rate_ctrl)))
    t_reduction, c_reduction = debt_reduction[treatment], debt_reduction[control]
    reduction_diff = float(t_reduction.mean() - c_reduction.mean())
    welch = stats.ttest_ind(t_reduction, c_reduction, equal_var=False)
    # Welch confidence interval for mean debt-reduction difference.
    se_w = np.sqrt(t_reduction.var(ddof=1) / n_treat + c_reduction.var(ddof=1) / n_ctrl)
    df_w = (t_reduction.var(ddof=1) / n_treat + c_reduction.var(ddof=1) / n_ctrl) ** 2 / (
        (t_reduction.var(ddof=1) / n_treat) ** 2 / (n_treat - 1) + (c_reduction.var(ddof=1) / n_ctrl) ** 2 / (n_ctrl - 1))
    crit = stats.t.ppf(.975, df_w)
    results = {
        "design": "Individual-level 1:1 randomized scenario; treatment effect is an explicit simulator assumption.",
        "assumptions": {"maximum absolute on_time uplift": assumed_max_absolute_uplift,
                        "generic recommendation acceptance": 0.25,
                        "personalized acceptance probability": "0.30 + 0.40 * recommendation_fit",
                        "accepted recommendation on_time effect": "1 percentage point for generic guidance; up to configured maximum for personalized guidance, scaled by fit",
                        "interest savings": "12% APR, 3-month horizon, based only on simulated extra principal reduction"},
        "groups": {"control_n": n_ctrl, "treatment_n": n_treat,
                   "control_on_time_rate": rate_ctrl, "treatment_on_time_rate": rate_treat,
                   "control_missed_rate": float(1 - rate_ctrl), "treatment_missed_rate": float(1 - rate_treat),
                   "control_total_debt_reduced": float(c_reduction.sum()),
                   "treatment_total_debt_reduced": float(t_reduction.sum()),
                   "control_acceptance_rate": float(accepted[control].mean()),
                   "treatment_acceptance_rate": float(accepted[treatment].mean()),
                   "control_interest_saved_estimate": float(interest_saved[control].sum()),
                   "treatment_interest_saved_estimate": float(interest_saved[treatment].sum())},
        "on_time_rate_test": {"null": "Control and treatment on-time rates are equal",
                              "difference_treatment_minus_control": diff, "95pct_ci": ci,
                              "z_statistic": float(z), "p_value": pvalue, "cohens_h": cohens_h,
                              "interpretation": "This is the result of the configured simulation, not measured customer impact."},
        "debt_reduction_test": {"difference_mean_treatment_minus_control": reduction_diff,
                                "95pct_ci": [float(reduction_diff - crit * se_w), float(reduction_diff + crit * se_w)],
                                "welch_t": float(welch.statistic), "p_value": float(welch.pvalue)},
        "production_test_notes": ["Randomize eligible customers after consent and stratify by baseline risk and affordability.",
                                  "Pre-register a primary outcome, guardrails, sample size, duration, and stopping rules.",
                                  "Measure on-time payment and debt balance from verified follow-up data; analyze intent-to-treat.",
                                  "Monitor disparate impacts and customer harm; do not use this simulation as causal evidence."],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
