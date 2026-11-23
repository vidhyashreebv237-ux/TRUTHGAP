"""
Configuration module for TRUTHGAP.
Dynamically resolves environment variables and Streamlit Cloud secrets.
Includes automatic key sanitization for accidental quotes or whitespace.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local environment variables from .env if present
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)

# Default Gemini model configuration
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TEMPERATURE = 0.2

# Expectation gap classification thresholds
GAP_THRESHOLDS = {
    "LOW": (0, 30),
    "MODERATE": (31, 60),
    "HIGH": (61, 100),
}


def sanitize_api_key(val: Any) -> str:
    """
    Cleans an API key by stripping:
    - Leading and trailing whitespace / newlines
    - Accidental surrounding quotes (' " ` “ ” ‘ ’) commonly caused by copy-pasting
    """
    if not val:
        return ""
    cleaned = str(val).strip()
    # Strip any accidental wrapping quotes (single, double, backtick, smart quotes)
    cleaned = cleaned.strip("'\"`“”‘’ \t\n\r")
    return cleaned


def get_gemini_api_key() -> str:
    """
    Dynamically retrieves and sanitizes the Gemini API key from either:
    1. OS Environment / .env file
    2. Streamlit Cloud Secrets (handles uppercase, lowercase, and TOML sections)
    """
    # 1. Check OS Environment / Local .env
    env_key = sanitize_api_key(os.getenv("GEMINI_API_KEY", ""))
    if env_key and env_key != "your_actual_gemini_api_key_here":
        return env_key

    # 2. Check Streamlit Cloud Secrets at runtime
    try:
        import streamlit as st
        # Direct uppercase key
        if "GEMINI_API_KEY" in st.secrets:
            val = sanitize_api_key(st.secrets["GEMINI_API_KEY"])
            if val:
                return val

        # Lowercase variation
        if "gemini_api_key" in st.secrets:
            val = sanitize_api_key(st.secrets["gemini_api_key"])
            if val:
                return val

        # Check nested TOML tables (e.g., [gemini] or [general])
        for section in st.secrets.values():
            if isinstance(section, dict):
                if "GEMINI_API_KEY" in section:
                    val = sanitize_api_key(section["GEMINI_API_KEY"])
                    if val:
                        return val
                if "gemini_api_key" in section:
                    val = sanitize_api_key(section["gemini_api_key"])
                    if val:
                        return val
    except Exception:
        pass

    return ""


# Module-level variable for backward compatibility
GEMINI_API_KEY = get_gemini_api_key()


def get_api_key_diagnostics() -> Dict[str, Any]:
    """
    Returns non-sensitive diagnostic info about the API key resolution.
    NEVER returns or displays the actual API key.
    """
    env_found = bool(os.getenv("GEMINI_API_KEY", "").strip())
    secrets_available = False
    secrets_keys = []
    secrets_error = None

    try:
        import streamlit as st
        secrets_keys = list(st.secrets.keys())
        secrets_available = len(secrets_keys) > 0
    except Exception as e:
        secrets_error = f"{type(e).__name__}: {str(e)}"

    resolved_key = get_gemini_api_key()
    is_configured = bool(resolved_key)

    source = "Not Found"
    if env_found and resolved_key == sanitize_api_key(os.getenv("GEMINI_API_KEY", "")):
        source = "Local .env / OS Environment"
    elif is_configured:
        source = "Streamlit Cloud Secrets"

    return {
        "is_configured": is_configured,
        "source": source,
        "key_length": len(resolved_key) if is_configured else 0,
        "secrets_accessible": secrets_available,
        "secrets_keys_found": secrets_keys,
        "secrets_error": secrets_error,
    }


def validate_config() -> bool:
    """
    Validates that a valid Gemini API key is configured.
    """
    key = get_gemini_api_key()
    if not key or key == "your_actual_gemini_api_key_here":
        diag = get_api_key_diagnostics()
        error_details = []
        if diag.get("secrets_error"):
            error_details.append(f"Secrets Error: {diag['secrets_error']}")
        elif not diag.get("secrets_accessible"):
            error_details.append("No secrets detected in st.secrets.")
        else:
            error_details.append(f"Secrets found: {diag['secrets_keys_found']}, but GEMINI_API_KEY is missing.")

        hint = (
            "GEMINI_API_KEY is not set.\n"
            "If running locally: Add GEMINI_API_KEY to your .env file.\n"
            "If on Streamlit Cloud: In App Settings > Secrets, add:\n"
            'GEMINI_API_KEY = "your_actual_key_here"\n'
            "(Make sure the key is enclosed in double quotes for valid TOML syntax!)"
        )
        if error_details:
            hint += f"\nDiagnostic info: {'; '.join(error_details)}"

        raise ValueError(hint)

    return True
