from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
API_BASE = os.getenv("BRIGHTPATH_API_URL", "http://127.0.0.1:8000").rstrip("/")


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "customer_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def api_call(method: str, route: str, payload: dict | None = None):
    try:
        response = requests.request(method, f"{API_BASE}{route}", json=payload, timeout=20)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, f"Backend unavailable: {exc}. Start it with `uvicorn backend.main:app --reload`."


def customer_picker(frame: pd.DataFrame, key: str = "customer_id") -> str:
    if frame.empty:
        return ""
    selected = st.selectbox("Sample customer", frame.customer_id.astype(str).tolist(), key=key)
    return str(selected)


def money(value: float) -> str:
    return f"{value:,.0f} units"


def show_api_warning(message: str | None) -> None:
    if message:
        st.warning(message)
