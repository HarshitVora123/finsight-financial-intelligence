"""Imbalance-aware, leakage-conscious repayment risk modeling."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from brightpath.features.engineering import MODEL_FEATURES

LOGGER = logging.getLogger(__name__)
MODEL_VERSION = "brightpath-risk-0.1.0"


def _pipeline(classifier) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                     ("scale", StandardScaler()), ("classifier", classifier)])


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray,
                      false_negative_cost: float = 3.0, false_positive_cost: float = 1.0) -> float:
    thresholds = np.linspace(.05, .85, 161)
    costs = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        fn = np.sum((y_true == 1) & ~predicted)
        fp = np.sum((y_true == 0) & predicted)
        costs.append(false_negative_cost * fn + false_positive_cost * fp)
    return float(thresholds[int(np.argmin(costs))])


def _metrics(y_true, probabilities, threshold) -> dict:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    curve_prob, curve_true = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration_curve": {"mean_predicted": [float(v) for v in curve_prob],
                              "observed_rate": [float(v) for v in curve_true]},
    }


def train_risk_model(frame: pd.DataFrame, output_dir: str | Path,
                     feature_columns: list[str] | None = None,
                     target_column: str = "simulated_missed_payment_next_month",
                     source: str = "synthetic_demo", seed: int = 42) -> dict:
    """Train and compare models, tune RF, calibrate the selected model, persist artifact.

    All preprocessing is inside sklearn pipelines. The holdout is never used for
    selection; threshold selection uses out-of-fold predictions on the train set.
    """
    if target_column not in frame:
        raise ValueError(f"Target column {target_column!r} not found")
    columns = feature_columns or [c for c in MODEL_FEATURES if c in frame.columns]
    if len(columns) < 2 or not set(columns).issubset(frame.columns):
        raise ValueError("At least two present model feature columns are required")
    x = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    y = frame[target_column].astype(int).copy()
    if y.nunique() != 2:
        raise ValueError("Risk outcome must have both positive and negative examples")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.25, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    candidates = {
        "logistic_regression": _pipeline(LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)),
        "decision_tree": _pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=12,
                                                            class_weight="balanced", random_state=seed)),
        "random_forest": _pipeline(RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=8,
                                                            class_weight="balanced_subsample", n_jobs=-1,
                                                            random_state=seed)),
        "gradient_boosting": _pipeline(GradientBoostingClassifier(n_estimators=100, learning_rate=.05,
                                                                    max_depth=2, random_state=seed)),
    }
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall", "f1": "f1"}
    comparisons: dict[str, dict] = {}
    fitted_candidates = {}
    for name, estimator in candidates.items():
        cv_scores = cross_validate(estimator, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        comparisons[name] = {f"cv_{metric}_mean": float(np.mean(cv_scores[f"test_{metric}"])) for metric in scoring}
        comparisons[name]["cv_pr_auc_std"] = float(np.std(cv_scores["test_pr_auc"]))
        fitted_candidates[name] = estimator
    # Small, reproducible RF hyperparameter search; score is independent of holdout.
    grid = GridSearchCV(candidates["random_forest"], {
        "classifier__n_estimators": [120, 200], "classifier__max_depth": [5, 9]},
        scoring={"pr_auc": "average_precision", "recall": "recall"},
        cv=cv, n_jobs=1, refit="pr_auc")
    grid.fit(x_train, y_train)
    fitted_candidates["tuned_random_forest"] = grid.best_estimator_
    best_index = grid.best_index_
    comparisons["tuned_random_forest"] = {"cv_pr_auc_mean": float(grid.cv_results_["mean_test_pr_auc"][best_index]),
                                            "cv_recall_mean": float(grid.cv_results_["mean_test_recall"][best_index]),
                                            "cv_pr_auc_std": float(grid.cv_results_["std_test_pr_auc"][best_index]),
                                            "best_params": grid.best_params_}
    # First enforce a predeclared supportive-intervention recall floor, then
    # choose the highest PR-AUC. If no model reaches it at 0.5, use best recall
    # and PR-AUC as the tie-break; the final operational threshold is tuned later.
    recall_floor = .45
    eligible = {name: score["cv_pr_auc_mean"] for name, score in comparisons.items()
                if score.get("cv_recall_mean", 0) >= recall_floor}
    if eligible:
        selected_name = max(eligible, key=eligible.get)
        selection_fallback = None
    else:
        selected_name = max(comparisons, key=lambda name: (comparisons[name].get("cv_recall_mean", 0),
                                                            comparisons[name]["cv_pr_auc_mean"]))
        selection_fallback = "No candidate met the 0.45 cross-validated recall floor; best recall then PR-AUC selected."
    base_estimator = fitted_candidates[selected_name]
    # Cross-fitted calibration and threshold selection avoid reusing holdout labels.
    calibrated_template = CalibratedClassifierCV(estimator=base_estimator, method="sigmoid", cv=3)
    oof_prob = cross_val_predict(calibrated_template, x_train, y_train, cv=cv,
                                 method="predict_proba", n_jobs=1)[:, 1]
    threshold = _select_threshold(y_train.to_numpy(), oof_prob)
    final_model = CalibratedClassifierCV(estimator=base_estimator, method="sigmoid", cv=3)
    final_model.fit(x_train, y_train)
    test_prob = final_model.predict_proba(x_test)[:, 1]
    test_metrics = _metrics(y_test.to_numpy(), test_prob, threshold)
    # Report a transparent intervention trade-off in count form as well as metrics.
    test_metrics["decision_cost_assumption"] = {"false_negative_relative_cost": 3, "false_positive_relative_cost": 1}
    importance = permutation_importance(final_model, x_test, y_test, scoring="average_precision",
                                        n_repeats=5, random_state=seed, n_jobs=1)
    feature_importance = sorted(
        [{"feature": name, "importance_mean": float(mean), "importance_std": float(std)}
         for name, mean, std in zip(columns, importance.importances_mean, importance.importances_std)],
        key=lambda item: item["importance_mean"], reverse=True)
    metadata = {
        "model_version": MODEL_VERSION, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source, "target": target_column, "features": columns,
        "protected_fields_excluded": ["sex", "age", "marital_status", "education"],
        "proxy_prone_profile_fields_excluded": ["employment_months", "credit_history_months"],
        "selected_model": selected_name, "decision_threshold": threshold,
        "training_rows": int(len(x_train)), "test_rows": int(len(x_test)),
        "positive_rate": float(y.mean()), "evaluation": test_metrics,
        "cross_validation": comparisons, "feature_importance": feature_importance,
        "selection_policy": {"primary_metric": "cross-validated PR-AUC",
                             "minimum_recall_at_0_5": recall_floor,
                             "selected_candidate_fallback": selection_fallback,
                             "threshold_policy": "Minimize 3*false negatives + 1*false positives using out-of-fold calibrated train probabilities."},
        "limitations": ["Decision support only; not an automated credit approval or denial.",
                        "Synthetic labels represent a simulation and are not real customer outcomes." if source == "synthetic_demo" else
                        "UCI target is historical Taiwan next-month default; population/time transfer is unvalidated."],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_medians = {str(c): float(x_train[c].median()) for c in columns}
    joblib.dump({"model": final_model, "feature_columns": columns, "feature_medians": feature_medians,
                 "metadata": metadata}, output / "risk_model.joblib")
    (output / "model_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Trained %s with PR-AUC %.3f", selected_name, test_metrics["pr_auc"])
    return metadata


def load_model(path: str | Path) -> dict:
    return joblib.load(path)


def predict_risk(artifact: dict, features: dict | pd.DataFrame) -> dict:
    row = pd.DataFrame([features]) if isinstance(features, dict) else features.copy()
    columns = artifact["feature_columns"]
    missing = set(columns).difference(row.columns)
    if missing:
        raise ValueError(f"Prediction input is missing required features: {sorted(missing)}")
    probability = float(artifact["model"].predict_proba(row[columns])[:, 1][0])
    threshold = float(artifact["metadata"]["decision_threshold"])
    risk_category = "elevated" if probability >= threshold else "lower"
    # Local perturbation explanation: one feature at a time to its training median.
    baseline = artifact.get("feature_medians", {})
    deltas = []
    base_row = row[columns].copy()
    for feature in columns:
        if feature not in baseline:
            continue
        changed = base_row.copy()
        changed.loc[changed.index[0], feature] = baseline[feature]
        changed_prob = float(artifact["model"].predict_proba(changed)[:, 1][0])
        deltas.append({"feature": feature, "observed_value": float(base_row.iloc[0][feature]),
                       "reference_value": float(baseline[feature]), "risk_change_if_at_reference": changed_prob - probability})
    deltas.sort(key=lambda item: abs(item["risk_change_if_at_reference"]), reverse=True)
    return {"missed_payment_probability": probability, "risk_category": risk_category,
            "decision_threshold": threshold,
            "explanation": deltas[:5],
            "caveat": "An estimated likelihood under the trained sample; it is not a guaranteed outcome."}
