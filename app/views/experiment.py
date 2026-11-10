import json

import pandas as pd
import streamlit as st

from app.common import ROOT


def render(frame: pd.DataFrame):
    st.title("Experimentation & Business Impact")
    path = ROOT / "reports" / "experiment_summary.json"
    if not path.exists():
        st.info("Run the pipeline to create the scenario-based experiment.")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    groups = report["groups"]
    st.caption(f"Control: generic recommendation (n={groups['control_n']:,}) · "
               f"Treatment: personalized recommendation (n={groups['treatment_n']:,})")
    data = pd.DataFrame({"group": ["Control", "Personalized"],
                         "on-time payment rate": [groups["control_on_time_rate"], groups["treatment_on_time_rate"]],
                         "missed payment rate": [groups["control_missed_rate"], groups["treatment_missed_rate"]],
                         "debt reduced": [groups["control_total_debt_reduced"], groups["treatment_total_debt_reduced"]]})
    st.bar_chart(data, x="group", y="on-time payment rate", color="group", use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("On-time rate difference", f"{report['on_time_rate_test']['difference_treatment_minus_control']:+.1%}")
    c2.metric("95% confidence interval", "[" + ", ".join(f"{v:+.1%}" for v in report["on_time_rate_test"]["95pct_ci"]) + "]")
    c3.metric("Scenario p-value", f"{report['on_time_rate_test']['p_value']:.3f}")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Generic recommendation accepted", f"{groups['control_acceptance_rate']:.1%}")
    d2.metric("Personalized recommendation accepted", f"{groups['treatment_acceptance_rate']:.1%}")
    d3.metric("Treatment debt reduced", f"{groups['treatment_total_debt_reduced']:,.0f} units")
    d4.metric("Estimated treatment interest saved", f"{groups['treatment_interest_saved_estimate']:,.0f} units")
    st.dataframe(data, hide_index=True, use_container_width=True)
    st.write("**Assumptions:**")
    st.json(report["assumptions"])
    st.warning("This is a simulated scenario with an assumed treatment effect. It is not real-world evidence that an intervention improves repayment.")
    st.subheader("Production test design")
    for note in report["production_test_notes"]:
        st.write("• " + note)
