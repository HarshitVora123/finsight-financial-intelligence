"""Train an independent baseline on UCI dataset 350 (CC BY 4.0)."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.data.uci import load_uci_credit_default_with_audit
from finsight.models.risk import train_risk_model


def _render_readme_summary(metrics: dict) -> str:
    evaluation = metrics["evaluation"]
    ci = evaluation.get("bootstrap_95pct_ci", {})
    pr_ci = ci["pr_auc"]
    roc_ci = ci["roc_auc"]
    matrix = evaluation["confusion_matrix"]
    best = metrics["cross_validation"][metrics["selected_model"]]
    return (
        f"The selected model was **{metrics['selected_model'].replace('_', ' ').title()}** on a stratified "
        f"{metrics['test_rows']:,}-record holdout from {metrics['source']} ({metrics['training_rows'] + metrics['test_rows']:,} "
        f"records total; {metrics['positive_rate']:.1%} positive across the full dataset). Test PR-AUC was "
        f"**{evaluation['pr_auc']:.3f}** (stratified-bootstrap 95% CI {pr_ci[0]:.3f}–{pr_ci[1]:.3f}; "
        f"test prevalence baseline {evaluation['positive_prevalence']:.3f}) and ROC-AUC was "
        f"**{evaluation['roc_auc']:.3f}** (95% CI {roc_ci[0]:.3f}–{roc_ci[1]:.3f}). At threshold "
        f"{evaluation['threshold']:.3f}, recall was {evaluation['recall']:.1%}, specificity "
        f"{evaluation['specificity']:.1%}, and Brier score {evaluation['brier_score']:.3f}; confusion matrix "
        f"TN/FP/FN/TP = {matrix['tn']}/{matrix['fp']}/{matrix['fn']}/{matrix['tp']}. The threshold uses an "
        "illustrative 3:1 false-negative to false-positive cost assumption, not a measured business cost. "
        f"Three-fold CV PR-AUC for the selected model was {best['cv_pr_auc_mean']:.3f} "
        f"(SD {best['cv_pr_auc_std']:.3f}). Demographic fields were excluded from training features and are used "
        "only for exploratory held-out subgroup checks. This random holdout estimates performance within this "
        "historical dataset; it is not a time-forward or external-population validation."
    )


def _update_readme(metrics: dict) -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    start_marker, end_marker = "<!-- UCI_MODEL_SUMMARY_START -->", "<!-- UCI_MODEL_SUMMARY_END -->"
    start, end = text.index(start_marker) + len(start_marker), text.index(end_marker)
    summary = "\n" + _render_readme_summary(metrics) + "\n"
    path.write_text(text[:start] + summary + text[end:], encoding="utf-8")


if __name__ == "__main__":
    cache = ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
    if cache.exists():
        x, y, audit = load_uci_credit_default_with_audit(cache)
    else:
        x, y, audit = load_uci_credit_default_with_audit()
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([x, audit, y], axis=1).to_csv(cache, index=False)
    frame = x.copy()
    frame["default_payment_next_month"] = y
    metrics = train_risk_model(frame, ROOT / "models" / "uci", feature_columns=list(x.columns),
                               target_column="default_payment_next_month", source="UCI_dataset_350",
                               audit_features=audit)
    _update_readme(metrics)
    print(json.dumps(metrics, indent=2))
