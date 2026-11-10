import json

import pandas as pd
import streamlit as st

from app.common import ROOT


def render(frame: pd.DataFrame):
    st.title("Customer Segmentation")
    path = ROOT / "reports" / "customer_segments.csv"
    summary_path = ROOT / "reports" / "segmentation_summary.json"
    if not path.exists() or not summary_path.exists():
        st.info("Run the data pipeline to create segment results.")
        return
    data = pd.read_csv(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected clusters", summary["selected_k"])
    c2.metric("Silhouette score", f"{summary['silhouette_score']:.3f}")
    c3.metric("Seed stability (ARI)", f"{summary['cluster_stability_ari']:.3f}")
    choice = st.selectbox("Highlight segment", ["All segments", *sorted(data.segment.unique())])
    show = data if choice == "All segments" else data.loc[data.segment.eq(choice)]
    st.subheader("PCA projection of standardized behaviour")
    st.scatter_chart(show, x="pca_1", y="pca_2", color="segment", size="monthly_income",
                     use_container_width=True)
    definitions = summary["segment_definitions"]
    tabs = st.tabs([definition["name"] for definition in definitions.values()])
    for tab, definition in zip(tabs, definitions.values()):
        with tab:
            st.write(definition["main_challenge"])
            st.write("**Potential intervention:** " + definition["potential_intervention"])
            st.write("**Business opportunity:** " + definition["business_opportunity"])
            st.json(definition["financial_characteristics"])
    st.caption(summary["caveat"])
