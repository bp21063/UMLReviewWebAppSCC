import os

import streamlit as st


def get_api_key(name: str) -> str:
    if not name:
        return ""
    env_value = os.getenv(name)
    if env_value:
        return env_value
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""

def get_password() -> str:
    env_value = os.getenv("PASSWORD")
    if env_value:
        return env_value
    try:
        return st.secrets.get("PASSWORD", "")
    except Exception:
        return ""