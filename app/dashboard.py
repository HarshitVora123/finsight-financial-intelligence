"""BrightPath AI's interactive Streamlit dashboard."""

import streamlit as st

from app.common import load_features
from app.views import debt, experiment, overview, products, profile, risk, segmentation

st.set_page_config(page_title="BrightPath AI", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .stApp { background: #f5f8fa; color: #12242d; }
  [data-testid="stSidebar"] { background: #102f3b; }
  [data-testid="stSidebar"] * { color: #edf7f5 !important; }
  h1, h2, h3 { color: #123b45; letter-spacing: -0.02em; }
  [data-testid="stMetric"] { background: white; border: 1px solid #e0e9eb; border-radius: 14px; padding: 16px; }
  [data-testid="stMetricLabel"] p { color: #607680 !important; opacity: 1 !important; }
  [data-testid="stMetricValue"] div { color: #123b45 !important; }
  [data-testid="stMetricDelta"] { color: #607680 !important; }
  div[data-testid="stProgress"] > div > div { background: #36a98b; }
  .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# ✦ BrightPath AI")
    st.caption("Financial decision intelligence")
    page = st.radio("Workspace", ["Executive Overview", "Customer Profile", "Risk Prediction",
                                   "Customer Segmentation", "Debt Optimization", "Product Matching",
                                   "Experimentation"], label_visibility="collapsed")
    st.divider()
    st.caption("Portfolio simulation · synthetic demo data")

frame = load_features()
views = {"Executive Overview": overview, "Customer Profile": profile, "Risk Prediction": risk,
         "Customer Segmentation": segmentation, "Debt Optimization": debt,
         "Product Matching": products, "Experimentation": experiment}
views[page].render(frame)
st.markdown("---")
st.caption("BrightPath AI is an educational portfolio simulation. Model estimates and product examples are not financial advice or actual lender offers.")
