import pandas as pd
import streamlit as st

from app.common import api_call, customer_picker, money, show_api_warning


def render(frame: pd.DataFrame):
    st.title("Customer Financial Profile")
    customer_id = customer_picker(frame, "profile_customer")
    if not customer_id:
        st.info("Generate the demo dataset first.")
        return
    profile, error = api_call("GET", f"/customer/{customer_id}/profile")
    show_api_warning(error)
    if not profile:
        return
    row = profile["profile"]
    st.caption(f"{customer_id} · Synthetic profile · values are demonstration currency units")
    columns = st.columns(4)
    for col, label, key in zip(columns, ["Monthly income", "Monthly spending", "Outstanding debt", "Liquid savings"],
                               ["monthly_income", "monthly_spending", "existing_debt", "savings_balance"]):
        col.metric(label, money(float(row.get(key, 0))))
    c1, c2, c3 = st.columns(3)
    c1.metric("Credit utilization", f"{float(row.get('credit_utilization_ratio', 0)):.0%}")
    c2.metric("Debt payment / income", f"{float(row.get('debt_to_income_ratio', 0)):.1%}")
    c3.metric("Savings coverage", f"{float(row.get('emergency_savings_coverage', 0)):.1f} months")
    st.subheader("Recent repayments")
    history = pd.DataFrame(profile.get("repayment_history", []))
    if not history.empty:
        history["status"] = history.on_time.map({1: "On time", 0: "Missed / late"})
        st.dataframe(history, hide_index=True, use_container_width=True)
    else:
        st.info("No repayment history found for this sample customer.")
    st.subheader("Active debts")
    st.dataframe(pd.DataFrame(profile.get("debts", [])), hide_index=True, use_container_width=True)
    st.caption(profile["caveat"])
