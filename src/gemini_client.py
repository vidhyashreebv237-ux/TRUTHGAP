"""
src/gemini_client.py - Gemini API Interface for TRUTHGAP

Handles communication with Google's Gemini API using the official google-genai SDK.
Includes automatic JSON sanitization, markdown fence removal, and graceful error handling.
"""

import json
import re
from typing import Any, Dict, List, Union
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.config import GEMINI_API_KEY, DEFAULT_MODEL, DEFAULT_TEMPERATURE, validate_config


def extract_json_from_text(raw_text: str) -> Union[Dict, List]:
    """
    Cleans and parses JSON from raw LLM output.
    Handles markdown code fences (```json ... ```) and leading/trailing chatter.
    """
    cleaned = raw_text.strip()

    # Strip markdown code blocks if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: Locate the outer brackets using regex
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Failed to parse valid JSON from AI response: {raw_text[:200]}...")


class GeminiClient:
    """Wrapper for interacting with the Google Gemini API."""

    def __init__(
        self,
        api_key: str = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        validate_config()
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.temperature = temperature
        
        # Initialize client with v1beta for developer system instruction support
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1beta"}
        )

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Sends a prompt to Gemini and returns plain text."""
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config
            )
            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")
            return response.text.strip()
        except APIError as e:
            raise RuntimeError(f"Gemini API error: {e.message} (Code: {e.code})") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error communicating with Gemini: {str(e)}") from e

    def generate_json(self, system_prompt: str, user_prompt: str) -> Union[Dict, List]:
        """Sends a prompt to Gemini and returns structured JSON (dict or list)."""
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                response_mime_type="application/json"
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config
            )
            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")
            return extract_json_from_text(response.text)
        except APIError as e:
            raise RuntimeError(f"Gemini API error: {e.message} (Code: {e.code})") from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate structured JSON: {str(e)}") from e
