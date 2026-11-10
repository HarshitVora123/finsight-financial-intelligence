"""Hypothetical financial product matching with explicit non-discriminatory rules."""

from __future__ import annotations

import math


PRODUCT_CATALOGUE = [
    {"product_id": "builder_01", "name": "Bright Builder Secured", "type": "secured credit-building",
     "apr": 0.049, "fee": 0.0, "term_months": 12, "deposit": 200.0,
     "description": "Illustrative secured instalment product; requires a refundable deposit."},
    {"product_id": "consolidation_01", "name": "BrightPath Lower-Rate Consolidation Pilot", "type": "debt consolidation",
     "apr": 0.129, "fee": 49.0, "term_months": 36, "deposit": 0.0,
     "description": "Hypothetical fixed-term consolidation scenario. No lender approval is predicted."},
    {"product_id": "saver_01", "name": "BrightPath Rainy Day Saver", "type": "savings",
     "apy": 0.035, "fee": 0.0, "term_months": 0, "deposit": 0.0,
     "description": "Illustrative no-fee savings account with a sample APY assumption."},
    {"product_id": "coach_01", "name": "BrightPath Payment Calendar", "type": "financial wellness tool",
     "apr": 0.0, "fee": 0.0, "term_months": 0, "deposit": 0.0,
     "description": "Free due-date and budget planning tool; not a credit product."},
]


def _payment(principal: float, annual_rate: float, months: int) -> float:
    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        return principal / months
    return principal * monthly_rate / (1 - (1 + monthly_rate) ** -months)


def match_products(profile: dict, debts: list[dict] | None = None,
                   catalogue: list[dict] | None = None) -> dict:
    """Rank illustrative options using affordability, eligibility, fees and benefit."""
    income = float(profile.get("monthly_income", 0))
    savings = float(profile.get("savings_balance", 0))
    monthly_spending = float(profile.get("monthly_spending", profile.get("monthly_expenses", 0)))
    monthly_budget = float(profile.get("monthly_budget", 0))
    debt_payment = float(profile.get("monthly_debt_payment", profile.get("minimum_monthly_payment", 0)))
    total_debt = float(profile.get("existing_debt", 0))
    payment_consistency = float(profile.get("payment_consistency_score", 1))
    if min(income, savings, monthly_spending, monthly_budget, debt_payment, total_debt) < 0:
        raise ValueError("Financial profile values cannot be negative")
    if income <= 0:
        raise ValueError("monthly_income must be positive")
    dti = debt_payment / income
    surplus = income - monthly_spending - debt_payment
    coverage = savings / max(monthly_spending, 1)
    active_catalogue = catalogue or PRODUCT_CATALOGUE
    ranked = []
    for product in active_catalogue:
        reasons, blockers = [], []
        eligibility = 1.0
        affordability = 1.0
        benefit = 0.5
        cost = float(product.get("fee", 0))
        if product["type"] == "debt consolidation":
            if total_debt < 1_000:
                eligibility = 0.0
                blockers.append("Outstanding debt is below the illustrative $1,000 minimum.")
            weighted_rate = 0.0
            if debts:
                principal_sum = sum(float(d.get("principal", 0)) for d in debts)
                weighted_rate = (sum(float(d.get("principal", 0)) * float(d.get("annual_rate", 0)) for d in debts)
                                 / principal_sum) if principal_sum else 0.0
            if weighted_rate <= float(product.get("apr", 1)) + .02:
                eligibility = 0.0
                blockers.append("Current weighted debt APR is not at least 2 percentage points above the sample APR.")
            if dti > .35:
                eligibility = 0.0
                blockers.append("Debt-payment-to-income ratio exceeds the sample affordability ceiling of 35%.")
            payment = _payment(max(total_debt, 0), float(product["apr"]), int(product["term_months"])) if total_debt else 0
            affordability = min(1.0, monthly_budget / max(payment, 1))
            if payment > monthly_budget:
                blockers.append("Illustrative monthly payment exceeds the stated monthly budget.")
            if eligibility:
                reasons.append(f"Sample APR is lower than current weighted APR; modeled payment is {payment:,.0f} units/month.")
                benefit = min(1, max(0, (weighted_rate - float(product["apr"])) / .15))
        elif product["type"] == "secured credit-building":
            if payment_consistency < .70:
                eligibility = 0.0
                blockers.append("Illustrative payment-history rule requires at least a 70% observed on-time share.")
            if dti > .35:
                eligibility = 0.0
                blockers.append("Debt-payment-to-income ratio exceeds the sample affordability ceiling of 35%.")
            if savings < float(product.get("deposit", 0)) or surplus < 0:
                affordability = .1
                blockers.append("Refundable deposit or positive monthly surplus is not available in this profile.")
            else:
                reasons.append("Observed payment consistency and deposit affordability meet the sample rules for an optional credit-building product.")
            benefit = .55 if eligibility > 0 else 0
        elif product["type"] == "savings":
            if surplus <= 0:
                affordability = .1
                blockers.append("No positive monthly cash-flow surplus is observed for recurring savings.")
            if coverage < 3:
                reasons.append(f"Observed savings cover about {coverage:.1f} months of spending; building a buffer may help.")
                benefit = min(1, (3 - coverage) / 3)
            else:
                benefit = .25
                reasons.append("Provides a liquid place for optional savings; current buffer is already near the sample goal.")
        else:
            reasons.append("Offers payment-date organization without adding borrowing or fees.")
            benefit = .7 if payment_consistency < .9 or dti > .2 else .45
        # Eligibility is a hard gate for credit offers; support tools remain discoverable.
        eligible = not blockers and affordability >= .5
        score = 100 * (0.4 * benefit + 0.35 * affordability + 0.25 * eligibility) - min(cost, 100) * .05
        ranked.append({**product, "eligible_under_sample_rules": bool(eligible),
                       "suitability_score": round(float(max(0, min(score, 100))), 1),
                       "estimated_monthly_payment": round(float(payment), 2) if product["type"] == "debt consolidation" else None,
                       "reasons": reasons, "rule_blockers": blockers})
    ranked.sort(key=lambda p: (p["eligible_under_sample_rules"], p["suitability_score"]), reverse=True)
    return {"products": ranked, "profile_rules": {"monthly_debt_payment_to_income": round(dti, 4),
                                                     "cash_flow_surplus": round(surplus, 2),
                                                     "savings_months_coverage": round(coverage, 2)},
            "caveat": "Catalogue, rates, fees, eligibility and scores are hypothetical. This does not represent a real offer, approval, or lender underwriting decision."}
