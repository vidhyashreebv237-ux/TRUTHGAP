"""
Configuration module for TRUTHGAP.
Loads environment variables and Streamlit secrets.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# Read API key from local .env or Streamlit Cloud secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    try:
        import streamlit as st
        GEMINI_API_KEY = str(
            st.secrets.get("GEMINI_API_KEY", "")
        ).strip()
    except Exception:
        GEMINI_API_KEY = ""

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2

GAP_THRESHOLDS = {
    "LOW": (0, 30),
    "MODERATE": (31, 60),
    "HIGH": (61, 100),
}

def validate_config() -> bool:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_actual_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Please configure your Gemini API key."
        )
    return True