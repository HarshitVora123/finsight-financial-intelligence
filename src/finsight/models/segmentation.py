"""Behavioural clustering with data-derived, human-readable labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


SEGMENT_FEATURES = ["monthly_income", "credit_utilization_ratio", "debt_to_income_ratio",
                    "spending_volatility", "monthly_savings_ratio", "emergency_savings_coverage",
                    "recent_delinquency_frequency", "payment_consistency_score", "active_credit_accounts"]


def _label_segment(row: pd.Series, baseline: pd.Series) -> tuple[str, str, str]:
    if row.credit_utilization_ratio > max(.65, baseline.credit_utilization_ratio * 1.35):
        return "High Credit Utilization", "Revolving balances consume a large share of available limit.", "Offer utilization coaching and a payoff plan; review consolidation affordability."
    if row.debt_to_income_ratio > max(.22, baseline.debt_to_income_ratio * 1.35):
        return "High Debt Burden", "Required monthly debt payments are high relative to income.", "Prioritize minimum-payment safety and compare lower-cost payoff scenarios."
    if row.spending_volatility > max(.16, baseline.spending_volatility * 1.35):
        return "Irregular Cash Flow", "Observed monthly outflows vary more than the portfolio midpoint.", "Offer cash-flow calendar tools and due-date alignment."
    if row.emergency_savings_coverage < max(.5, baseline.emergency_savings_coverage * .55):
        return "Limited Emergency Buffer", "Liquid savings cover few observed months of spending.", "Suggest an affordable starter buffer alongside required debt minimums."
    if row.payment_consistency_score >= baseline.payment_consistency_score and row.credit_utilization_ratio <= baseline.credit_utilization_ratio:
        return "Consistent Repayers", "Observed on-time payment share is at or above the portfolio midpoint.", "Support continued payment consistency and optional savings goals."
    return "Financially Stable", "Observed affordability and cash-flow signals are near or better than portfolio medians.", "Offer self-serve progress tracking and optional balance optimization."


def segment_customers(frame: pd.DataFrame, output_dir: str | Path,
                      features: list[str] | None = None, seed: int = 42) -> dict:
    columns = features or SEGMENT_FEATURES
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Segmentation inputs missing: {sorted(missing)}")
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required for customer segmentation")
    x = SimpleImputer(strategy="median").fit_transform(frame[columns])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    candidates = {}
    for k in range(2, min(7, len(frame) - 1)):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(scaled)
        candidates[k] = (silhouette_score(scaled, labels, sample_size=min(3000, len(frame)), random_state=seed), km, labels)
    best_k = max(candidates, key=lambda k: candidates[k][0])
    score, model, labels = candidates[best_k]
    rerun = KMeans(n_clusters=best_k, n_init=10, random_state=seed + 11).fit_predict(scaled)
    stability = float(adjusted_rand_score(labels, rerun))
    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(scaled)
    enriched = frame.copy()
    enriched["cluster_id"] = labels
    enriched["pca_1"], enriched["pca_2"] = coordinates[:, 0], coordinates[:, 1]
    stats = enriched.groupby("cluster_id")[columns].mean()
    baseline = enriched[columns].median()
    names, definitions = {}, {}
    used = set()
    for cluster_id, row in stats.iterrows():
        name, challenge, intervention = _label_segment(row, baseline)
        if name in used:
            name = f"{name} {int(cluster_id) + 1}"
        used.add(name)
        names[int(cluster_id)] = name
        definitions[int(cluster_id)] = {"name": name, "financial_characteristics": {c: float(row[c]) for c in columns},
                                        "main_challenge": challenge, "potential_intervention": intervention,
                                        "business_opportunity": "Measure engagement and sustained repayment outcomes in a randomized pilot."}
    enriched["segment"] = enriched.cluster_id.map(names)
    # Persist coordinates for the dashboard and metadata for reproducible interpretation.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enriched[[c for c in ["customer_id", "cluster_id", "segment", "pca_1", "pca_2", *columns] if c in enriched]].to_csv(
        output / "customer_segments.csv", index=False)
    result = {"selected_k": int(best_k), "silhouette_score": float(score), "cluster_stability_ari": stability,
              "cluster_sizes": {names[int(i)]: int(n) for i, n in enriched.cluster_id.value_counts().items()},
              "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
              "segment_definitions": definitions, "features": columns,
              "caveat": "Cluster labels summarize this sample; they are exploratory and should not determine eligibility."}
    (output / "segmentation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
