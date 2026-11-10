import pandas as pd
import streamlit as st

from app.common import api_call, customer_picker, show_api_warning


def render(frame: pd.DataFrame):
    st.title("Repayment Risk Estimate")
    st.caption("Probability of the simulated customer missing a future payment; use for support planning only.")
    customer_id = customer_picker(frame, "risk_customer")
    if st.button("Estimate risk", type="primary", disabled=not customer_id):
        result, error = api_call("POST", "/predict-risk", {"customer_id": customer_id})
        show_api_warning(error)
        if result:
            st.session_state["risk_result"] = result
    result = st.session_state.get("risk_result")
    if result and result.get("customer_id") == customer_id:
        st.metric("Estimated missed-payment probability", f"{result['missed_payment_probability']:.1%}",
                  delta=f"{result['risk_category']} · threshold {result['decision_threshold']:.2f}", delta_color="off")
        st.caption(f"Model version: {result['model_version']}")
        st.subheader("Largest local sensitivity checks")
        st.dataframe(pd.DataFrame(result["explanation"]), hide_index=True, use_container_width=True)
        st.info(result["caveat"] + " Sensitivity checks compare one feature with its training median; they are not causal explanations.")
