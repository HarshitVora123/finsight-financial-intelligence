import pandas as pd
import streamlit as st

from app.common import api_call, customer_picker, show_api_warning


def render(frame: pd.DataFrame):
    st.title("Hypothetical Product Matching")
    st.caption("Transparent sample catalogue. No actual lender approval or offer is predicted.")
    customer_id = customer_picker(frame, "product_customer")
    if st.button("Find suitable options", type="primary", disabled=not customer_id):
        result, error = api_call("POST", "/match-products", {"customer_id": customer_id})
        show_api_warning(error)
        if result:
            st.session_state["product_result"] = result
    result = st.session_state.get("product_result")
    if result:
        st.caption(f"Sample profile rules: debt payment / income {result['profile_rules']['monthly_debt_payment_to_income']:.1%} · "
                   f"cash flow surplus {result['profile_rules']['cash_flow_surplus']:,.0f} units · "
                   f"buffer {result['profile_rules']['savings_months_coverage']:.1f} months")
        for product in result["products"]:
            with st.container(border=True):
                st.subheader(product["name"])
                st.progress(product["suitability_score"] / 100, text=f"Suitability {product['suitability_score']:.0f}/100 · "
                            f"{'Suitable under sample rules' if product['eligible_under_sample_rules'] else 'Not currently eligible'}")
                st.write(product["description"])
                for reason in product["reasons"]:
                    st.write("• " + reason)
                for blocker in product["rule_blockers"]:
                    st.caption("Rule check: " + blocker)
        st.info(result["caveat"])
