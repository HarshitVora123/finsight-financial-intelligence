"""Month-by-month debt avalanche and snowball payoff simulations."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal


def _validated_debts(debts: list[dict]) -> list[dict]:
    if not isinstance(debts, list):
        raise ValueError("debts must be a list")
    seen = set()
    result = []
    for index, debt in enumerate(debts):
        if not isinstance(debt, dict):
            raise ValueError(f"Debt at index {index} must be an object")
        principal = float(debt.get("principal", debt.get("balance", -1)))
        annual_rate = float(debt.get("annual_rate", debt.get("interest_rate", -1)))
        minimum = float(debt.get("minimum_payment", 0))
        if principal < 0 or annual_rate < 0 or minimum < 0:
            raise ValueError("Principal, annual_rate, and minimum_payment must be non-negative")
        debt_id = str(debt.get("debt_id", f"debt_{index + 1}"))
        if debt_id in seen:
            raise ValueError(f"Duplicate debt_id: {debt_id}")
        seen.add(debt_id)
        result.append({"debt_id": debt_id, "principal": principal, "annual_rate": annual_rate,
                       "minimum_payment": minimum, "due_day": int(debt.get("due_day", 1))})
    return result


def simulate_strategy(debts: list[dict], monthly_budget: float,
                      strategy: Literal["avalanche", "snowball"], max_months: int = 600) -> dict:
    items = _validated_debts(debts)
    budget = float(monthly_budget)
    if budget < 0 or max_months < 1:
        raise ValueError("monthly_budget must be non-negative and max_months must be positive")
    if strategy not in {"avalanche", "snowball"}:
        raise ValueError("strategy must be avalanche or snowball")
    total_principal = sum(d["principal"] for d in items)
    if total_principal <= 1e-8:
        return {"strategy": strategy, "total_interest": 0.0, "total_repaid": 0.0,
                "debt_free_month": 0, "paid_off": True, "schedule": [], "remaining_balances": {}}
    minimums = sum(min(d["minimum_payment"], d["principal"]) for d in items if d["principal"] > 1e-8)
    if budget + 1e-8 < minimums:
        raise ValueError(f"Monthly budget ({budget:.2f}) is below required minimum payments ({minimums:.2f})")

    balances = {d["debt_id"]: d["principal"] for d in items}
    debt_by_id = {d["debt_id"]: d for d in items}
    total_interest = total_repaid = 0.0
    schedule = []
    for month in range(1, max_months + 1):
        month_interest = {}
        for debt_id, balance in list(balances.items()):
            interest = balance * debt_by_id[debt_id]["annual_rate"] / 12
            month_interest[debt_id] = interest
            balances[debt_id] += interest
        required = {debt_id: min(debt_by_id[debt_id]["minimum_payment"], balance)
                    for debt_id, balance in balances.items() if balance > 1e-8}
        required_total = sum(required.values())
        if budget + 1e-8 < required_total:
            raise ValueError(f"Monthly budget ({budget:.2f}) is below current required minimums ({required_total:.2f})")
        payments = {debt_id: amount for debt_id, amount in required.items()}
        remaining_budget = budget - required_total
        active = [debt_id for debt_id, balance in balances.items() if balance > 1e-8]
        if strategy == "avalanche":
            active.sort(key=lambda debt_id: (-debt_by_id[debt_id]["annual_rate"], balances[debt_id], debt_id))
        else:
            active.sort(key=lambda debt_id: (balances[debt_id], -debt_by_id[debt_id]["annual_rate"], debt_id))
        for debt_id in active:
            if remaining_budget <= 1e-8:
                break
            extra = min(remaining_budget, max(0.0, balances[debt_id] - payments.get(debt_id, 0.0)))
            payments[debt_id] = payments.get(debt_id, 0.0) + extra
            remaining_budget -= extra
        month_interest_total = sum(month_interest.values())
        total_interest += month_interest_total
        month_repaid = sum(payments.values())
        total_repaid += month_repaid
        for debt_id, payment in payments.items():
            balances[debt_id] = max(0.0, balances[debt_id] - payment)
        # The schedule records after-payment balances and never budgets above the input.
        schedule.append({"month": month, "payment": round(month_repaid, 2),
                         "interest": round(month_interest_total, 2),
                         "balances": {key: round(value, 2) for key, value in balances.items()}})
        if all(balance <= 1e-8 for balance in balances.values()):
            break
    paid_off = all(balance <= 1e-8 for balance in balances.values())
    return {"strategy": strategy, "total_interest": round(total_interest, 2),
            "total_repaid": round(total_repaid, 2), "debt_free_month": len(schedule) if paid_off else None,
            "paid_off": paid_off, "schedule": schedule,
            "remaining_balances": {key: round(value, 2) for key, value in balances.items()},
            "assumptions": "Monthly interest is balance * APR/12, added before payment; rates, fees, and payments are assumed constant."}


def compare_strategies(debts: list[dict], monthly_budget: float) -> dict:
    validated = _validated_debts(debts)
    avalanche = simulate_strategy(deepcopy(validated), monthly_budget, "avalanche")
    snowball = simulate_strategy(deepcopy(validated), monthly_budget, "snowball")
    if avalanche["paid_off"] and snowball["paid_off"]:
        preferred = "avalanche" if avalanche["total_interest"] <= snowball["total_interest"] else "snowball"
    elif avalanche["paid_off"]:
        preferred = "avalanche"
    elif snowball["paid_off"]:
        preferred = "snowball"
    else:
        preferred = "avalanche" if avalanche["remaining_balances"] and sum(avalanche["remaining_balances"].values()) <= sum(snowball["remaining_balances"].values()) else "snowball"
    chosen = avalanche if preferred == "avalanche" else snowball
    other = snowball if preferred == "avalanche" else avalanche
    return {"monthly_budget": float(monthly_budget), "avalanche": avalanche, "snowball": snowball,
            "personalized_strategy": preferred, "personalized": chosen,
            "interest_savings_vs_alternative": round(other["total_interest"] - chosen["total_interest"], 2),
            "personalization_rule": "Choose the strategy with lower modeled interest when both pay off; otherwise choose the strategy with lower remaining balance at the simulation horizon. Snowball breaks exact interest ties."}
