"""Unit and integration coverage, runnable with pytest or stdlib unittest."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / ".test-tmp"
TEST_TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TEST_TMP)
sys.path.insert(0, str(ROOT / "src"))

from brightpath.data.storage import write_bundle_sqlite, write_segments_sqlite
from brightpath.data.synthetic import generate_synthetic_data
from brightpath.data.uci import load_uci_credit_default
from brightpath.data.validation import validate_bundle
from brightpath.evaluation.experimentation import simulate_experiment
from brightpath.features.engineering import MODEL_FEATURES, build_customer_features
from brightpath.models.risk import predict_risk, load_model
from brightpath.models.segmentation import segment_customers
from brightpath.optimization.debt import compare_strategies, simulate_strategy
from brightpath.recommendations.products import match_products
import backend.main as api


class ConstantRiskModel:
    def predict_proba(self, frame):
        return np.tile([0.7, 0.3], (len(frame), 1))


class TestDataAndFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_synthetic_data(120, seed=17)
        cls.features = build_customer_features(cls.bundle.customers, cls.bundle.transactions, cls.bundle.repayments)

    def test_schema_validation_and_transaction_keys(self):
        report = validate_bundle(self.bundle)
        self.assertEqual(len(report), 4)
        self.assertEqual(sum(item["duplicate_key_rows"] for item in report), 0)

    def test_engineered_features_are_finite_and_label_excluded(self):
        self.assertTrue(set(MODEL_FEATURES).issubset(self.features.columns))
        self.assertNotIn("simulated_missed_payment_next_month", MODEL_FEATURES)
        self.assertNotIn("employment_months", MODEL_FEATURES)
        self.assertNotIn("credit_history_months", MODEL_FEATURES)
        self.assertTrue(np.isfinite(self.features[MODEL_FEATURES].to_numpy()).all())
        self.assertTrue(self.features.credit_utilization_ratio.between(0, 1.1).all())

    def test_invalid_customer_key_rejected(self):
        broken = self.bundle.customers.copy()
        broken.loc[1, "customer_id"] = broken.loc[0, "customer_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            build_customer_features(broken, self.bundle.transactions, self.bundle.repayments)

    def test_cached_uci_loader_excludes_sensitive_and_identifier_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "uci.csv"
            pd.DataFrame({"ID": [1, 2, 3, 4], "LIMIT_BAL": [1000, 2000, 3000, 4000],
                          "SEX": [1, 2, 1, 2], "AGE": [25, 30, 40, 50],
                          "PAY_0": [0, 1, 0, 2], "default.payment.next.month": [0, 1, 0, 1]}).to_csv(path, index=False)
            x, y = load_uci_credit_default(path)
            self.assertEqual(list(x.columns), ["LIMIT_BAL", "PAY_0"])
            self.assertEqual(y.name, "default_payment_next_month")
            self.assertEqual(list(y), [0, 1, 0, 1])


class TestDebtOptimization(unittest.TestCase):
    def test_avalanche_and_snowball_payoff(self):
        debts = [{"debt_id": "high_rate", "principal": 1_800, "annual_rate": .28, "minimum_payment": 50},
                 {"debt_id": "small_balance", "principal": 450, "annual_rate": .05, "minimum_payment": 25}]
        result = compare_strategies(debts, 450)
        self.assertTrue(result["avalanche"]["paid_off"])
        self.assertTrue(result["snowball"]["paid_off"])
        self.assertLessEqual(result["avalanche"]["total_interest"], result["snowball"]["total_interest"])
        self.assertLessEqual(max(row["payment"] for row in result["avalanche"]["schedule"]), 450.01)
        expected = result["snowball"]["total_interest"] - result["avalanche"]["total_interest"]
        self.assertEqual(result["interest_savings_vs_alternative"], round(expected, 2))

    def test_zero_interest_and_already_paid(self):
        result = simulate_strategy([{"debt_id": "zero", "principal": 200, "annual_rate": 0,
                                     "minimum_payment": 20}], 100, "snowball")
        self.assertEqual(result["total_interest"], 0)
        self.assertEqual(result["debt_free_month"], 2)
        paid = simulate_strategy([{"debt_id": "paid", "principal": 0, "annual_rate": .2,
                                   "minimum_payment": 0}], 100, "avalanche")
        self.assertEqual(paid["debt_free_month"], 0)
        small_payoff = simulate_strategy([{"debt_id": "tiny", "principal": 10, "annual_rate": .12,
                                           "minimum_payment": 100}], 50, "avalanche")
        self.assertTrue(small_payoff["paid_off"])

    def test_insufficient_budget_and_invalid_debt(self):
        with self.assertRaisesRegex(ValueError, "below required minimum"):
            compare_strategies([{"debt_id": "d1", "principal": 500, "annual_rate": .1,
                                 "minimum_payment": 70}], 50)
        with self.assertRaises(ValueError):
            simulate_strategy([{"principal": -1, "annual_rate": .1}], 100, "avalanche")


class TestDecisionSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_synthetic_data(120, seed=19)
        cls.features = build_customer_features(cls.bundle.customers, cls.bundle.transactions, cls.bundle.repayments)

    def test_product_match_transparent_and_ranked(self):
        row = self.features.iloc[0].to_dict()
        debts = self.bundle.debts.loc[self.bundle.debts.customer_id.eq(row["customer_id"])].to_dict("records")
        result = match_products(row, debts)
        self.assertEqual(len(result["products"]), 4)
        self.assertGreaterEqual(result["products"][0]["suitability_score"], result["products"][-1]["suitability_score"])
        self.assertIn("hypothetical", result["caveat"])

    def test_segmentation_and_experiment_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            segments = segment_customers(self.features, temp, seed=9)
            self.assertGreaterEqual(segments["selected_k"], 2)
            self.assertTrue((Path(temp) / "customer_segments.csv").exists())
            experiment = simulate_experiment(self.features, temp, seed=9)
            self.assertEqual(experiment["groups"]["control_n"] + experiment["groups"]["treatment_n"], len(self.features))
            self.assertIn("not measured customer impact", experiment["on_time_rate_test"]["interpretation"])

    def test_risk_prediction_explanation_and_model_load(self):
        artifact = {"model": ConstantRiskModel(), "feature_columns": ["monthly_income", "credit_utilization_ratio"],
                    "feature_medians": {"monthly_income": 3_000, "credit_utilization_ratio": .4},
                    "metadata": {"decision_threshold": .25, "model_version": "test"}}
        customer = {"monthly_income": 4_000, "credit_utilization_ratio": .8}
        result = predict_risk(artifact, customer)
        self.assertEqual(result["risk_category"], "elevated")
        self.assertEqual(result["missed_payment_probability"], .3)
        self.assertTrue(result["explanation"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.joblib"
            joblib.dump(artifact, path)
            self.assertEqual(load_model(path)["metadata"]["model_version"], "test")


class TestApiIntegration(unittest.TestCase):
    def test_routes_and_request_validation(self):
        bundle = generate_synthetic_data(100, seed=23)
        features = build_customer_features(bundle.customers, bundle.transactions, bundle.repayments)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data" / "processed").mkdir(parents=True)
            (root / "models").mkdir()
            (root / "reports").mkdir()
            features.to_csv(root / "data" / "processed" / "customer_features.csv", index=False)
            write_bundle_sqlite(bundle, root / "data" / "brightpath.sqlite")
            segments = pd.DataFrame({"customer_id": features.customer_id, "cluster_id": 0,
                                     "segment": "Financially Stable", "pca_1": 0.0, "pca_2": 0.0})
            segments.to_csv(root / "reports" / "customer_segments.csv", index=False)
            write_segments_sqlite(segments, root / "data" / "brightpath.sqlite")
            with closing(sqlite3.connect(root / "data" / "brightpath.sqlite")) as connection:
                foreign_keys = connection.execute("PRAGMA foreign_key_list(customer_segments)").fetchall()
                self.assertEqual(foreign_keys[0][2], "customers")
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            artifact = {"model": ConstantRiskModel(),
                        "feature_columns": ["monthly_income", "credit_utilization_ratio"],
                        "feature_medians": {"monthly_income": 3000.0, "credit_utilization_ratio": .4},
                        "metadata": {"decision_threshold": .25, "model_version": "test-model"}}
            joblib.dump(artifact, root / "models" / "risk_model.joblib")
            customer_id = str(features.iloc[0].customer_id)
            with patch.object(api, "ROOT", root):
                self.assertEqual(api.health()["status"], "ok")
                self.assertTrue(api.app.openapi()["paths"]["/predict-risk"]["post"]["responses"]["200"]["content"]["application/json"]["schema"])
                TypeAdapter(api.HealthResponse).validate_python(api.health())
                summary = api.dashboard_summary()
                self.assertEqual(len(summary["monthly_trends"]), 6)
                TypeAdapter(api.DashboardSummaryResponse).validate_python(summary)
                profile = api.customer_profile(customer_id)
                self.assertEqual(profile["data_origin"], "synthetic_demo")
                TypeAdapter(api.CustomerProfileResponse).validate_python(profile)
                risk = api.predict_risk_endpoint(api.RiskRequest(customer_id=customer_id))
                self.assertIn("missed_payment_probability", risk)
                TypeAdapter(api.RiskResponse).validate_python(risk)
                segment = api.segment_customer_endpoint(api.SegmentRequest(customer_id=customer_id))
                self.assertEqual(segment["segment"], "Financially Stable")
                TypeAdapter(api.SegmentResponse).validate_python(segment)
                payoff = api.optimize_debt_endpoint(api.OptimizationRequest(monthly_budget=200, debts=[
                    api.DebtInput(debt_id="d1", principal=300, annual_rate=.1, minimum_payment=20)]))
                self.assertIn("snowball", payoff)
                TypeAdapter(api.DebtOptimizationResponse).validate_python(payoff)
                with self.assertRaises(HTTPException) as invalid:
                    api.optimize_debt_endpoint(api.OptimizationRequest(monthly_budget=10, debts=[
                        api.DebtInput(debt_id="d1", principal=300, annual_rate=.1, minimum_payment=20)]))
                self.assertEqual(invalid.exception.status_code, 422)
                product = api.match_products_endpoint(api.ProductRequest(customer_id=customer_id))
                self.assertIn("products", product)
                TypeAdapter(api.ProductMatchResponse).validate_python(product)
                with self.assertRaises(HTTPException) as unknown:
                    api.predict_risk_endpoint(api.RiskRequest(customer_id="absent"))
                self.assertEqual(unknown.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
