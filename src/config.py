"""
Configuration module for TRUTHGAP.
Loads environment variables, validates API credentials, and sets system constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from the project root
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Gemini Model Settings
# Using gemini-2.5-flash as default: fast, highly capable for structured extraction and comparison
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2  # Low temperature ensures deterministic, structured evaluations

# Expectation Gap Classification Thresholds
GAP_THRESHOLDS = {
    "LOW": (0, 30),        # Product largely delivers on advertised promises
    "MODERATE": (31, 60),  # Noticeable compromise or partial delivery
    "HIGH": (61, 100),     # Severe disparity between claims and user reality
}

def validate_config() -> bool:
    """Validates that all essential configurations are present."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_actual_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set or still contains placeholder. "
            "Please add your valid Gemini API key in the .env file."
        )
    return True
