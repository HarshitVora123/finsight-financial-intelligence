"""Generate data and build the reproducible BrightPath demo artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brightpath.data.storage import write_bundle_sqlite, write_segments_sqlite
from brightpath.data.synthetic import generate_synthetic_data
from brightpath.data.validation import normalize_categories, validate_bundle
from brightpath.evaluation.eda import create_eda_report
from brightpath.evaluation.experimentation import simulate_experiment
from brightpath.features.engineering import build_customer_features, source_table_dictionary
from brightpath.models.risk import load_model, train_risk_model
from brightpath.models.segmentation import segment_customers
from brightpath.monitoring.monitor import monitor_batch


def _render_readme_summary(summary: dict, model: dict, segmentation: dict,
                           experiment: dict, eda: dict) -> str:
    metrics = model["evaluation"]
    prevalence = eda["hypothesis_tests"]["outcome_prevalence"]["rate"]
    cv = model["cross_validation"]
    confusion = metrics["confusion_matrix"]
    util = eda["hypothesis_tests"]["credit_utilization_ratio"]
    volatility = eda["hypothesis_tests"]["spending_volatility"]
    dti = eda["hypothesis_tests"]["debt_to_income_ratio"]
    chi = eda["hypothesis_tests"]["utilization_chi_square"]
    on_time = experiment["on_time_rate_test"]
    debt_test = experiment["debt_reduction_test"]
    groups = experiment["groups"]
    parameters = cv[model["selected_model"]].get("best_params", {})
    parameter_summary = ", ".join(f"{key.split('__')[-1]}={value}" for key, value in parameters.items())
    if not parameter_summary:
        parameter_summary = "default parameters"
    cv_lines = []
    for name, result in cv.items():
        cv_lines.append(f"{name.replace('_', ' ').title()} **{result['cv_pr_auc_mean']:.3f} / {result.get('cv_recall_mean', 0):.3f}**")
    segment_lines = []
    for name, count in segmentation["cluster_sizes"].items():
        segment = next(value for value in segmentation["segment_definitions"].values() if value["name"] == name)
        characteristics = segment["financial_characteristics"]
        segment_lines.append(f"{count} `{name}` (on-time share {characteristics['payment_consistency_score']:.1%}, "
                             f"utilization {characteristics['credit_utilization_ratio']:.1%}, "
                             f"savings coverage {characteristics['emergency_savings_coverage']:.1f} months)")
    return "\n\n".join([
        f"Seed {summary['seed']} generated **{summary['customers']:,} customer profiles**, **{summary['transactions']:,} transactions**, "
        f"**{summary['debts']:,} debt accounts**, and **{summary['repayment_records']:,} repayment observations**. "
        f"The synthetic target's positive rate was **{prevalence:.1%}**.",
        "| Risk-model result | Value |\n|---|---:|\n"
        f"| Selected model | {model['selected_model'].replace('_', ' ').title()} ({parameter_summary}) |\n"
        f"| Test split | {model['test_rows']} stratified customers; threshold {model['decision_threshold']:.3f} |\n"
        f"| PR-AUC / ROC-AUC | {metrics['pr_auc']:.3f} / {metrics['roc_auc']:.3f} |\n"
        f"| Accuracy / precision / recall / F1 | {metrics['accuracy']:.3f} / {metrics['precision']:.3f} / {metrics['recall']:.3f} / {metrics['f1']:.3f} |\n"
        f"| Specificity / false positive rate / false negative rate | {metrics['specificity']:.3f} / {metrics['false_positive_rate']:.3f} / {metrics['false_negative_rate']:.3f} |\n"
        f"| Brier score | {metrics['brier_score']:.3f} |\n"
        f"| Confusion matrix | TN {confusion['tn']}, FP {confusion['fp']}, FN {confusion['fn']}, TP {confusion['tp']} |",
        "Cross-validation PR-AUC / recall at threshold 0.5: " + ", ".join(cv_lines) + ". "
        f"Selection used a {model['selection_policy']['minimum_recall_at_0_5']:.2f} recall floor, then CV PR-AUC. "
        "The operating threshold uses training-only out-of-fold probabilities under a 3:1 false-negative cost assumption. "
        f"The holdout false-positive rate is {metrics['false_positive_rate']:.1%}; this is an illustrative support setting. "
        "Accuracy alone can obscure missed cases on an imbalanced target, so PR-AUC, recall, calibration and the error trade-off are reported.",
        f"EDA found mean utilization of {util['missed_mean']:.1%} for simulated missed-payment cases versus {util['on_time_mean']:.1%} "
        f"for on-time cases (difference {util['mean_difference'] * 100:.1f} percentage points, 95% CI "
        f"{util['difference_95pct_ci'][0] * 100:.1f} to {util['difference_95pct_ci'][1] * 100:.1f} points, "
        f"Welch p={util['welch_p_value']:.3g}, Cohen's d={util['cohens_d']:.2f}). "
        f"Debt-payment-to-income ratio was {dti['missed_mean']:.1%} versus {dti['on_time_mean']:.1%} "
        f"(difference {dti['mean_difference'] * 100:.1f} percentage points, 95% CI "
        f"{dti['difference_95pct_ci'][0] * 100:.1f} to {dti['difference_95pct_ci'][1] * 100:.1f} points, "
        f"p={dti['welch_p_value']:.3g}). "
        f"Spending volatility was {volatility['missed_mean']:.2%} versus {volatility['on_time_mean']:.2%} "
        f"(p={volatility['welch_p_value']:.3f}), so the example volatility hypothesis was not supported. "
        f"The >=70% utilization chi-square test had p={chi['p_value']:.3g}, Cramer's V={chi['cramers_v']:.3f}. "
        "These associations describe the generated target mechanism and are not evidence about real consumers.",
        f"K-means selected **{segmentation['selected_k']} clusters** (silhouette {segmentation['silhouette_score']:.3f}; "
        f"second-seed ARI {segmentation['cluster_stability_ari']:.3f}): " + "; ".join(segment_lines) + ". "
        "Names are assigned from the observed generated cluster profiles.",
        f"In the randomized **simulation**, treatment on-time rate was {groups['treatment_on_time_rate']:.1%} versus "
        f"{groups['control_on_time_rate']:.1%} in control, a {on_time['difference_treatment_minus_control'] * 100:.1f}-point difference "
        f"(95% CI {on_time['95pct_ci'][0] * 100:.1f} to {on_time['95pct_ci'][1] * 100:.1f} percentage points, p={on_time['p_value']:.3f}); "
        f"this simulated on-time difference is {'statistically significant' if on_time['p_value'] < .05 else 'not statistically significant'} at 5%. "
        f"Simulated mean debt reduction differed by {debt_test['difference_mean_treatment_minus_control']:.2f} units "
        f"(95% CI {debt_test['95pct_ci'][0]:.2f} to {debt_test['95pct_ci'][1]:.2f}, Welch p={debt_test['p_value']:.3g}); "
        f"this difference is {'statistically significant' if debt_test['p_value'] < .05 else 'not statistically significant'} at 5%. "
        f"Generic/personalized acceptance was {groups['control_acceptance_rate']:.1%}/{groups['treatment_acceptance_rate']:.1%}. "
        f"Estimated interest saved was {groups['control_interest_saved_estimate']:,.0f}/{groups['treatment_interest_saved_estimate']:,.0f} units "
        "for control/treatment under the stated simulator rules. "
        "These are not real customer outcomes or causal evidence.",
    ])


def run(customers: int = 1_200, seed: int = 42) -> dict:
    bundle = generate_synthetic_data(customers, seed)
    for category in ("category", "merchant_category", "transaction_type"):
        bundle.transactions = normalize_categories(bundle.transactions, category)
    bundle.debts = normalize_categories(bundle.debts, "debt_type")
    quality = validate_bundle(bundle)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "synthetic").mkdir(parents=True, exist_ok=True)
    for table_name in ("customers", "transactions", "debts", "repayments"):
        getattr(bundle, table_name).to_csv(ROOT / "data" / "synthetic" / f"{table_name}.csv", index=False)
    write_bundle_sqlite(bundle, ROOT / "data" / "brightpath.sqlite")
    (ROOT / "reports" / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    features = build_customer_features(bundle.customers, bundle.transactions, bundle.repayments)
    features.to_csv(ROOT / "data" / "processed" / "customer_features.csv", index=False)
    (ROOT / "reports" / "data_dictionary.json").write_text(json.dumps(source_table_dictionary(), indent=2), encoding="utf-8")
    eda = create_eda_report(features, ROOT / "reports")
    model_metadata = train_risk_model(features, ROOT / "models", seed=seed)
    segmentation = segment_customers(features, ROOT / "reports", seed=seed)
    write_segments_sqlite(pd.read_csv(ROOT / "reports" / "customer_segments.csv"),
                          ROOT / "data" / "brightpath.sqlite")
    experiment = simulate_experiment(features, ROOT / "reports", seed=seed)
    artifact = load_model(ROOT / "models" / "risk_model.joblib")
    numeric = features[artifact["feature_columns"]]
    import time
    started = time.perf_counter()
    probability = artifact["model"].predict_proba(numeric)[:, 1]
    prediction_latency_ms = (time.perf_counter() - started) * 1000 / len(numeric)
    midpoint = len(numeric) // 2
    monitor_batch(numeric.iloc[:midpoint], numeric.iloc[midpoint:], probability[midpoint:],
                  ROOT / "reports" / "monitoring.jsonl",
                  actual_labels=features.iloc[midpoint:]["simulated_missed_payment_next_month"],
                  prediction_latency_ms=prediction_latency_ms)
    summary = {
        "source": "synthetic_demo", "seed": seed, "customers": len(bundle.customers),
        "transactions": len(bundle.transactions), "debts": len(bundle.debts), "repayment_records": len(bundle.repayments),
        "default_model": model_metadata["selected_model"],
        "test_pr_auc": model_metadata["evaluation"]["pr_auc"],
        "test_roc_auc": model_metadata["evaluation"]["roc_auc"],
        "segment_count": segmentation["selected_k"], "segment_silhouette": segmentation["silhouette_score"],
        "simulated_treatment_on_time": experiment["groups"]["treatment_on_time_rate"],
        "simulated_control_on_time": experiment["groups"]["control_on_time_rate"],
        "eda_outcome_rate": eda["hypothesis_tests"]["outcome_prevalence"]["rate"],
        "disclaimer": "Generated demonstration data only; not real-world consumer or causal results.",
    }
    (ROOT / "reports" / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    start_marker, end_marker = "<!-- GENERATED_RUN_SUMMARY_START -->", "<!-- GENERATED_RUN_SUMMARY_END -->"
    if start_marker not in readme or end_marker not in readme:
        raise ValueError("README generated-summary markers are missing")
    before, remainder = readme.split(start_marker, 1)
    _, after = remainder.split(end_marker, 1)
    readme_path.write_text(before + start_marker + "\n" + _render_readme_summary(
        summary, model_metadata, segmentation, experiment, eda) + "\n" + end_marker + after, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.customers, args.seed), indent=2))
