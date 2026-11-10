"""FastAPI decision-support service. Run `uvicorn backend.main:app --reload`."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
import sys
from contextlib import closing

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brightpath.models.risk import load_model, predict_risk
from brightpath.optimization.debt import compare_strategies
from brightpath.recommendations.products import match_products

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brightpath.api")
app = FastAPI(title="BrightPath AI", version="0.1.0",
              description="Portfolio decision-support API. Not a credit approval or denial service.")


class DebtInput(BaseModel):
    debt_id: str = Field(min_length=1, max_length=80)
    principal: float = Field(ge=0)
    annual_rate: float = Field(ge=0, le=5, description="Annual interest rate as decimal, e.g. 0.18 for 18%")
    minimum_payment: float = Field(default=0, ge=0)
    due_day: int = Field(default=1, ge=1, le=31)


class RiskRequest(BaseModel):
    customer_id: str = Field(min_length=1)


class SegmentRequest(BaseModel):
    customer_id: str = Field(min_length=1)


class OptimizationRequest(BaseModel):
    debts: list[DebtInput] = Field(min_length=1)
    monthly_budget: float = Field(ge=0)


class ProductRequest(BaseModel):
    customer_id: str | None = None
    profile: dict | None = None
    debts: list[DebtInput] | None = None

    @model_validator(mode="after")
    def has_customer_or_profile(self):
        if not self.customer_id and self.profile is None:
            raise ValueError("Provide customer_id or a financial profile")
        return self


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    database_ready: bool
    data_source: str


class DashboardSummaryResponse(BaseModel):
    customers: int
    average_outstanding_debt: float
    average_credit_utilization: float
    estimated_high_risk_share: float | None
    average_monthly_income: float
    average_monthly_spending: float
    segment_distribution: dict[str, int]
    monthly_trends: list[dict]
    data_origin: str


class RiskExplanation(BaseModel):
    feature: str
    observed_value: float
    reference_value: float
    risk_change_if_at_reference: float


class RiskResponse(BaseModel):
    customer_id: str
    model_version: str
    missed_payment_probability: float
    risk_category: str
    decision_threshold: float
    explanation: list[RiskExplanation]
    caveat: str


class SegmentResponse(BaseModel):
    customer_id: str
    segment: str
    cluster_id: int
    pca_coordinates: list[float]
    caveat: str


class ProductMatchResponse(BaseModel):
    products: list[dict]
    profile_rules: dict[str, float]
    caveat: str


class StrategyProjection(BaseModel):
    strategy: str
    total_interest: float
    total_repaid: float
    debt_free_month: int | None
    paid_off: bool
    schedule: list[dict]
    remaining_balances: dict[str, float]
    assumptions: str | None = None


class DebtOptimizationResponse(BaseModel):
    monthly_budget: float
    avalanche: StrategyProjection
    snowball: StrategyProjection
    personalized_strategy: str
    personalized: StrategyProjection
    interest_savings_vs_alternative: float
    personalization_rule: str


class CustomerProfileResponse(BaseModel):
    customer_id: str
    profile: dict
    repayment_history: list[dict]
    debts: list[dict]
    data_origin: str
    caveat: str


def _feature_frame() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "customer_features.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Demo artifacts not found. Run `python scripts/run_pipeline.py` first.")
    return pd.read_csv(path)


def _find_customer(customer_id: str) -> pd.Series:
    frame = _feature_frame()
    matched = frame.loc[frame.customer_id.astype(str).eq(customer_id)]
    if matched.empty:
        raise HTTPException(status_code=404, detail="Customer ID not found in the synthetic demo data")
    return matched.iloc[0]


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_ready": (ROOT / "models" / "risk_model.joblib").exists(),
            "database_ready": (ROOT / "data" / "brightpath.sqlite").exists(),
            "data_source": "synthetic_demo"}


@app.get("/dashboard-summary", response_model=DashboardSummaryResponse)
def dashboard_summary():
    frame = _feature_frame()
    try:
        artifact = load_model(ROOT / "models" / "risk_model.joblib")
        probabilities = artifact["model"].predict_proba(frame[artifact["feature_columns"]])[:, 1]
        high_risk_rate = float((probabilities >= artifact["metadata"]["decision_threshold"]).mean())
    except FileNotFoundError:
        high_risk_rate = None
    segments_path = ROOT / "reports" / "customer_segments.csv"
    distribution = {}
    if segments_path.exists():
        distribution = pd.read_csv(segments_path).segment.value_counts().to_dict()
    monthly_trends = []
    try:
        with closing(sqlite3.connect(ROOT / "data" / "brightpath.sqlite")) as connection:
            monthly_trends = pd.read_sql_query("""
                WITH by_customer_month AS (
                    SELECT customer_id, substr(transaction_date, 1, 7) AS month,
                           SUM(CASE WHEN transaction_type='credit' THEN amount ELSE 0 END) AS income,
                           SUM(CASE WHEN transaction_type='debit' THEN amount ELSE 0 END) AS spending
                    FROM transactions GROUP BY customer_id, substr(transaction_date, 1, 7)
                )
                SELECT month, AVG(income) AS avg_income, AVG(spending) AS avg_spending
                FROM by_customer_month GROUP BY month ORDER BY month
            """, connection).to_dict(orient="records")
    except sqlite3.Error:
        logger.exception("Monthly transaction trend query failed")
    return {"customers": int(len(frame)), "average_outstanding_debt": float(frame.existing_debt.mean()),
            "average_credit_utilization": float(frame.credit_utilization_ratio.mean()),
            "estimated_high_risk_share": high_risk_rate,
            "average_monthly_income": float(frame.monthly_income.mean()),
            "average_monthly_spending": float(frame.monthly_spending.mean()),
            "segment_distribution": distribution, "monthly_trends": monthly_trends,
            "data_origin": "synthetic_demo"}


@app.post("/predict-risk", response_model=RiskResponse)
def predict_risk_endpoint(request: RiskRequest):
    try:
        artifact = load_model(ROOT / "models" / "risk_model.joblib")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Risk model artifact missing. Run the data pipeline first.")
    row = _find_customer(request.customer_id)
    result = predict_risk(artifact, row.to_dict())
    return {"customer_id": request.customer_id, "model_version": artifact["metadata"]["model_version"], **result}


@app.post("/segment-customer", response_model=SegmentResponse)
def segment_customer_endpoint(request: SegmentRequest):
    path = ROOT / "reports" / "customer_segments.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Segmentation results missing. Run the data pipeline first.")
    frame = pd.read_csv(path)
    rows = frame.loc[frame.customer_id.astype(str).eq(request.customer_id)]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Customer ID not found in the synthetic demo data")
    row = rows.iloc[0]
    return {"customer_id": request.customer_id, "segment": row.segment,
            "cluster_id": int(row.cluster_id), "pca_coordinates": [float(row.pca_1), float(row.pca_2)],
            "caveat": "Exploratory cluster label derived from this synthetic sample."}


@app.post("/optimize-debt", response_model=DebtOptimizationResponse)
def optimize_debt_endpoint(request: OptimizationRequest):
    try:
        return compare_strategies([debt.model_dump() for debt in request.debts], request.monthly_budget)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/match-products", response_model=ProductMatchResponse)
def match_products_endpoint(request: ProductRequest):
    if request.profile is not None:
        profile = request.profile
        debts = [debt.model_dump() for debt in (request.debts or [])]
    else:
        row = _find_customer(request.customer_id or "")
        profile = row.to_dict()
        with closing(sqlite3.connect(ROOT / "data" / "brightpath.sqlite")) as connection:
            debts = pd.read_sql_query("SELECT * FROM debts WHERE customer_id = ?", connection,
                                      params=(request.customer_id,)).to_dict(orient="records")
    try:
        return match_products(profile, debts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/customer/{customer_id}/profile", response_model=CustomerProfileResponse)
def customer_profile(customer_id: str):
    row = _find_customer(customer_id)
    database = ROOT / "data" / "brightpath.sqlite"
    try:
        with closing(sqlite3.connect(database)) as connection:
            repayments = pd.read_sql_query("SELECT period, amount_due, amount_paid, on_time FROM repayments WHERE customer_id = ? ORDER BY period",
                                           connection, params=(customer_id,)).to_dict(orient="records")
            debts = pd.read_sql_query("SELECT debt_id, principal, annual_rate, minimum_payment, due_day, debt_type FROM debts WHERE customer_id = ?",
                                      connection, params=(customer_id,)).to_dict(orient="records")
    except sqlite3.Error as exc:
        logger.exception("Database read failed")
        raise HTTPException(status_code=503, detail="Customer database is unavailable") from exc
    values = row.to_dict()
    return {"customer_id": customer_id,
            "profile": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                        for k, v in values.items() if k not in {"simulated_missed_payment_next_month"}},
            "repayment_history": repayments, "debts": debts,
            "data_origin": "synthetic_demo",
            "caveat": "Sample profile and financial events are generated demonstration data."}
