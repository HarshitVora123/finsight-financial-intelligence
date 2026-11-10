import pandas as pd
import streamlit as st

from app.common import api_call, show_api_warning


def render(frame: pd.DataFrame):
    st.title("Debt Payoff Optimizer")
    st.caption("Compare fixed-budget snowball and avalanche payoff projections.")
    with st.form("debt_form"):
        budget = st.number_input("Affordable monthly budget", min_value=0.0, value=650.0, step=25.0)
        count = st.number_input("Number of debts", min_value=1, max_value=8, value=3, step=1)
        debts = []
        for i in range(int(count)):
            cols = st.columns(4)
            debt_id = cols[0].text_input("Debt name", value=f"Debt {i + 1}", key=f"debt_id_{i}")
            principal = cols[1].number_input("Balance", min_value=0.0, value=[2400., 5200., 1800.][i % 3], step=100.0, key=f"principal_{i}")
            rate_pct = cols[2].number_input("APR (%)", min_value=0.0, value=[19.9, 11.0, 27.0][i % 3], step=.25, key=f"rate_{i}")
            minimum = cols[3].number_input("Minimum payment", min_value=0.0, value=60.0, step=10.0, key=f"minimum_{i}")
            debts.append({"debt_id": debt_id, "principal": principal, "annual_rate": rate_pct / 100,
                          "minimum_payment": minimum, "due_day": 15})
        submitted = st.form_submit_button("Compare strategies", type="primary")
    if submitted:
        result, error = api_call("POST", "/optimize-debt", {"debts": debts, "monthly_budget": budget})
        show_api_warning(error)
        if result:
            st.session_state["debt_result"] = result
    result = st.session_state.get("debt_result")
    if result:
        comparison = pd.DataFrame([result["avalanche"], result["snowball"]])
        st.dataframe(comparison[["strategy", "total_interest", "total_repaid", "debt_free_month", "paid_off"]],
                     hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Personalized strategy", result["personalized_strategy"].title())
        interest_difference = result["interest_savings_vs_alternative"]
        interest_label = "Interest saved vs alternative" if interest_difference >= 0 else "Additional interest vs alternative"
        c2.metric(interest_label, f"{abs(interest_difference):,.2f} units")
        schedules = []
        for name in ("avalanche", "snowball"):
            for month in result[name]["schedule"]:
                schedules.append({"month": month["month"], "strategy": name.title(),
                                  "remaining balance": sum(month["balances"].values())})
        if schedules:
            st.line_chart(pd.DataFrame(schedules), x="month", y="remaining balance", color="strategy",
                          use_container_width=True)
        st.subheader("Personalized month-by-month schedule")
        chosen_schedule = pd.DataFrame(result["personalized"]["schedule"])
        if not chosen_schedule.empty:
            monthly_summary = chosen_schedule[["month", "payment", "interest"]].copy()
            monthly_summary["total remaining"] = chosen_schedule.balances.map(lambda values: sum(values.values()))
            st.dataframe(monthly_summary, hide_index=True, use_container_width=True)
            per_debt = pd.DataFrame([
                {"month": row.month, "debt": debt_id, "remaining_balance": balance}
                for row in chosen_schedule.itertuples()
                for debt_id, balance in row.balances.items()
            ])
            st.subheader("Remaining balance by debt")
            st.line_chart(per_debt, x="month", y="remaining_balance", color="debt", use_container_width=True)
        st.caption(result["personalized"]["assumptions"] + " These are estimates, not guarantees.")
