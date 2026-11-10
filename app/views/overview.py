import pandas as pd
import streamlit as st

from app.common import api_call, money, show_api_warning


def render(frame: pd.DataFrame):
    st.title("Executive Overview")
    st.caption("Portfolio signals from a reproducible synthetic demo dataset")
    summary, error = api_call("GET", "/dashboard-summary")
    show_api_warning(error)
    if summary:
        cols = st.columns(4)
        cols[0].metric("Customers analyzed", f"{summary['customers']:,}")
        cols[1].metric("Average debt", money(summary["average_outstanding_debt"]))
        cols[2].metric("Credit utilization", f"{summary['average_credit_utilization']:.0%}")
        risk = summary["estimated_high_risk_share"]
        cols[3].metric("Elevated estimated risk", f"{risk:.1%}" if risk is not None else "Unavailable")
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Customer segments")
            dist = pd.DataFrame({"segment": list(summary["segment_distribution"]),
                                 "customers": list(summary["segment_distribution"].values())})
            if not dist.empty:
                st.bar_chart(dist, x="segment", y="customers", color="segment", use_container_width=True)
        with right:
            st.subheader("Income and spending")
            if not frame.empty:
                chart = frame[["monthly_income", "monthly_spending"]].rename(
                    columns={"monthly_income": "Income", "monthly_spending": "Spending"})
                st.scatter_chart(chart.sample(min(500, len(chart)), random_state=42), x="Income", y="Spending",
                                 use_container_width=True)
        st.info("Risk estimates support proactive financial assistance. They are not credit approval decisions.")
    if not frame.empty:
        st.subheader("Monthly cash-flow trend")
        if summary and summary.get("monthly_trends"):
            trend = pd.DataFrame(summary["monthly_trends"]).rename(
                columns={"avg_income": "Average income", "avg_spending": "Average outflow"})
            st.line_chart(trend, x="month", y=["Average income", "Average outflow"], use_container_width=True)
