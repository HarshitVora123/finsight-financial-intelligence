"""Reproducible descriptive and inferential analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def univariate_summary(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include="number")
    result = {}
    for column in numeric:
        x = numeric[column].dropna()
        if x.empty:
            continue
        result[column] = {
            "count": int(x.size), "mean": float(x.mean()), "median": float(x.median()),
            "mode": float(x.mode().iloc[0]), "variance": float(x.var()), "std": float(x.std()),
            "q1": float(x.quantile(.25)), "q3": float(x.quantile(.75)),
            "p05": float(x.quantile(.05)), "p95": float(x.quantile(.95)),
            "skewness": float(x.skew()), "kurtosis": float(x.kurt()),
        }
    return result


def hypothesis_tests(features: pd.DataFrame, label: str = "simulated_missed_payment_next_month") -> dict:
    if label not in features:
        raise ValueError(f"Outcome column {label!r} is not present")
    y = features[label].astype(int)
    groups = {"missed": features.loc[y.eq(1)], "on_time": features.loc[y.eq(0)]}
    result: dict[str, dict] = {"outcome_prevalence": {"rate": float(y.mean()), "n": int(y.size)}}
    tests = [
        ("credit_utilization_ratio", "Credit utilization is higher among missed-payment cases"),
        ("spending_volatility", "Spending volatility is higher among missed-payment cases"),
        ("debt_to_income_ratio", "Debt-payment-to-income ratio is higher among missed-payment cases"),
    ]
    for column, hypothesis in tests:
        a = groups["missed"][column].dropna().astype(float)
        b = groups["on_time"][column].dropna().astype(float)
        test = stats.ttest_ind(a, b, equal_var=False)
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
            (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
        crit = stats.t.ppf(.975, df)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        difference = float(a.mean() - b.mean())
        pooled_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
        cohen_d = difference / pooled_sd if pooled_sd > 0 else 0.0
        magnitude = "small" if abs(cohen_d) < .2 else ("moderate" if abs(cohen_d) < .5 else "large")
        result[column] = {
            "null": f"Mean {column} is equal for missed and on-time customers",
            "alternative": f"{hypothesis} (two-sided equality test; direction is read from the group means)",
            "test_assumptions": "Independent observations; Welch's test does not assume equal variances and is reasonably robust with these sample sizes. Mann-Whitney compares rank distributions and is not automatically a test of medians.",
            "missed_mean": float(a.mean()), "on_time_mean": float(b.mean()),
            "mean_difference": difference, "welch_t": float(test.statistic), "welch_p_value": float(test.pvalue),
            "difference_95pct_ci": [float(difference - crit * se), float(difference + crit * se)],
            "cohens_d": float(cohen_d), "standardized_effect_size": magnitude,
            "mann_whitney_u": float(mw.statistic), "mann_whitney_p_value": float(mw.pvalue),
            "interpretation": ("The groups differ in this generated sample at alpha=0.05; this is association, not causation."
                               if test.pvalue < .05 else "The sample does not provide evidence of a mean difference at alpha=0.05."),
            "practical_significance_note": f"The standardized difference is {magnitude} by the Cohen's d rule of thumb (d={cohen_d:.2f}); domain experts must decide whether the raw-unit difference is operationally material.",
        }
    high_util = features.credit_utilization_ratio.ge(.7)
    contingency = pd.crosstab(high_util, y)
    chi = stats.chi2_contingency(contingency, correction=False)
    cramers_v = float(np.sqrt(chi.statistic / (len(features) * min(contingency.shape[0] - 1, contingency.shape[1] - 1))))
    result["utilization_chi_square"] = {
        "null": "High utilization (>=70%) and next-month simulated missed payment are independent",
        "test_assumptions": "Independent observations and expected cell counts generally at least five; inspect expected counts for small samples.",
        "chi2": float(chi.statistic), "degrees_of_freedom": int(chi.dof), "p_value": float(chi.pvalue),
        "cramers_v": cramers_v,
        "contingency_table": contingency.to_dict(),
        "interpretation": "Association in the sample; the 70% boundary is a descriptive cut, not a lending rule.",
    }
    corr = stats.pearsonr(features.debt_to_income_ratio, y)
    result["dti_point_biserial_correlation"] = {
        "correlation": float(corr.statistic), "p_value": float(corr.pvalue),
        "interpretation": "A correlation describes direction and strength; it does not establish a causal mechanism.",
    }
    return result


def create_eda_report(features: pd.DataFrame, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": {"rows": int(len(features)), "columns": int(len(features.columns)),
                    "origin": str(features.get("data_origin", pd.Series(["unknown"])).iloc[0])},
        "univariate": univariate_summary(features), "hypothesis_tests": hypothesis_tests(features),
        "correlation_matrix": features.select_dtypes(include="number").corr().fillna(0).to_dict(),
        "business_readout": "These findings are descriptive of the selected dataset. Synthetic rows illustrate a pipeline and do not estimate real consumer outcomes.",
    }
    (output / "eda_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
