"""Transparent evaluation view for the public historical UCI model."""

import pandas as pd
import streamlit as st

from app.common import api_call, show_api_warning


def render(frame: pd.DataFrame | None = None):
    st.title("Public Data Model Evaluation")
    st.caption("Measured results from a real, public historical credit-default dataset")
    report, error = api_call("GET", "/uci-model-summary")
    if error and "503" not in error:
        show_api_warning(error)
    if not report:
        st.info("Train the public-data model first: `python scripts/train_uci.py`. The first run downloads and caches the UCI dataset.")
        return

    evaluation = report["evaluation"]
    ci = evaluation.get("bootstrap_95pct_ci", {})
    st.markdown(
        "**Dataset:** [UCI Default of Credit Card Clients (dataset 350)]"
        "(https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients), "
        "30,000 records from Taiwan; licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
        "The target is next-month default. Sensitive demographic fields are excluded from model predictors; "
        "available fields are used separately for exploratory test-set subgroup checks."
    )
    st.warning(
        "This is a historical benchmark, not evidence of performance on current customers or other populations. "
        "It does not score the synthetic customer profiles shown on other pages."
    )
    st.caption("The shown operating threshold uses an illustrative 3:1 false-negative to false-positive cost assumption; it is not a validated business cost.")

    first_row = st.columns(4)
    first_row[0].metric("Test PR-AUC", f"{evaluation['pr_auc']:.3f}",
                        delta=f"Baseline prevalence {evaluation['positive_prevalence']:.3f}", delta_color="off")
    first_row[1].metric("Test ROC-AUC", f"{evaluation['roc_auc']:.3f}")
    first_row[2].metric("Recall", f"{evaluation['recall']:.1%}",
                        delta=f"Threshold {evaluation['threshold']:.3f}", delta_color="off")
    first_row[3].metric("Brier score", f"{evaluation['brier_score']:.3f}")

    ci_cols = st.columns(2)
    if "pr_auc" in ci:
        ci_cols[0].caption(f"PR-AUC 95% stratified-bootstrap CI: {ci['pr_auc'][0]:.3f}–{ci['pr_auc'][1]:.3f}")
    if "roc_auc" in ci:
        ci_cols[1].caption(f"ROC-AUC 95% stratified-bootstrap CI: {ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f}")

    matrix = evaluation["confusion_matrix"]
    st.subheader("Held-out test results")
    st.dataframe(pd.DataFrame([
        {"Metric": "Accuracy", "Value": evaluation["accuracy"]},
        {"Metric": "Precision", "Value": evaluation["precision"]},
        {"Metric": "Specificity", "Value": evaluation["specificity"]},
        {"Metric": "False positive rate", "Value": evaluation["false_positive_rate"]},
        {"Metric": "False negative rate", "Value": evaluation["false_negative_rate"]},
        {"Metric": "TN / FP / FN / TP", "Value": f"{matrix['tn']} / {matrix['fp']} / {matrix['fn']} / {matrix['tp']}"},
    ]), hide_index=True, use_container_width=True)

    comparison = report.get("cross_validation", {})
    if comparison:
        st.subheader("Model comparison (training-fold cross-validation)")
        comparison_frame = pd.DataFrame([
            {"Model": name.replace("_", " ").title(),
             "Mean PR-AUC": values.get("cv_pr_auc_mean"),
             "PR-AUC SD": values.get("cv_pr_auc_std"),
             "Mean recall": values.get("cv_recall_mean")}
            for name, values in comparison.items()
        ]).sort_values("Mean PR-AUC", ascending=False)
        st.dataframe(comparison_frame, hide_index=True, use_container_width=True)

    calibration = evaluation.get("calibration_curve", {})
    if calibration.get("mean_predicted"):
        st.subheader("Calibration on held-out records")
        calibration_frame = pd.DataFrame({
            "Mean predicted probability": calibration["mean_predicted"],
            "Observed default rate": calibration["observed_rate"],
        })
        st.line_chart(calibration_frame, x="Mean predicted probability", y="Observed default rate")

    importance = pd.DataFrame(report.get("feature_importance", []))
    if not importance.empty:
        st.subheader("Permutation importance")
        st.caption("Importance measures changes in test PR-AUC when one feature is shuffled; it does not establish causation.")
        st.bar_chart(importance.head(10), x="feature", y="importance_mean")

    subgroup = evaluation.get("subgroup_evaluation", {})
    subgroup_rows = [
        {"Audit field": field, "Group": group, **values}
        for field, groups in subgroup.items() for group, values in groups.items()
    ]
    if subgroup_rows:
        st.subheader("Exploratory held-out subgroup checks")
        st.caption("Protected attributes were used for auditing only, never as predictors. Slice differences require expert review and are not a complete fairness certification.")
        st.dataframe(pd.DataFrame(subgroup_rows), hide_index=True, use_container_width=True)

    st.caption(
        f"Source: {report.get('source')}; selected model: {report.get('selected_model')}; "
        f"training rows: {report.get('training_rows'):,}; test rows: {report.get('test_rows'):,}. "
        "The split is stratified and random, not a time-forward validation."
    )
