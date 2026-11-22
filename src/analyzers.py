"""
src/analyzers.py - Core Analytical Pipeline for TRUTHGAP

Orchestrates the entire Expectation Gap analysis workflow:
1. Advertised Claim Extraction (Role-Based + Few-Shot + Structured Output)
2. Review Sentiment & Ground-Truth Synthesis (Few-Shot + Structured Output)
3. Zero-Shot Comparative Expectation Gap Scoring (0-100)
4. Low, Moderate, High Gap Categorization
5. Final AI Executive Audit Report Generation
"""

import json
from typing import Any, Dict, List, Union
import pandas as pd

from src.gemini_client import GeminiClient
from src.config import GAP_THRESHOLDS
from src.prompts import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    CLAIM_EXTRACTION_USER_TEMPLATE,
    REVIEW_SYNTHESIS_SYSTEM_PROMPT,
    REVIEW_SYNTHESIS_USER_TEMPLATE,
    GAP_COMPARISON_SYSTEM_PROMPT,
    GAP_COMPARISON_USER_TEMPLATE,
    EXECUTIVE_REPORT_SYSTEM_PROMPT,
    EXECUTIVE_REPORT_USER_TEMPLATE,
)


def categorize_gap_score(score: float) -> str:
    """Classifies a numerical gap score into Low, Moderate, or High category."""
    if score <= GAP_THRESHOLDS["LOW"][1]:
        return "Low"
    elif score <= GAP_THRESHOLDS["MODERATE"][1]:
        return "Moderate"
    else:
        return "High"


def parse_reviews_input(reviews_input: Union[str, pd.DataFrame, List[str]]) -> str:
    """
    Standardizes review input from different formats (raw text string, DataFrame, list)
    into a numbered text format for LLM processing.
    """
    if isinstance(reviews_input, pd.DataFrame):
        # Look for standard review column names
        possible_cols = ["Review_Text", "review_text", "Review", "review", "text", "Comment", "comment"]
        target_col = None
        for col in possible_cols:
            if col in reviews_input.columns:
                target_col = col
                break
        
        if target_col is None:
            # Fall back to the first text column or join all columns
            text_cols = reviews_input.select_dtypes(include=["object", "string"]).columns
            target_col = text_cols[0] if len(text_cols) > 0 else reviews_input.columns[0]
            
        review_list = reviews_input[target_col].dropna().astype(str).tolist()
    elif isinstance(reviews_input, list):
        review_list = [str(r).strip() for r in reviews_input if str(r).strip()]
    elif isinstance(reviews_input, str):
        # Split by newlines or treat as numbered entries
        lines = [line.strip() for line in reviews_input.strip().split("\n") if line.strip()]
        review_list = lines
    else:
        raise ValueError(f"Unsupported review input type: {type(reviews_input)}")

    if not review_list:
        raise ValueError("No valid customer reviews found in the provided input.")

    # Format as numbered entries for the prompt
    formatted_reviews = []
    for idx, rev in enumerate(review_list, 1):
        # Clean up existing numbering if present
        clean_rev = rev.lstrip("0123456789.-) ")
        formatted_reviews.append(f"{idx}. \"{clean_rev}\"")
        
    return "\n".join(formatted_reviews)


class TruthGapAnalyzer:
    """Main pipeline orchestrator for detecting product expectation gaps."""

    def __init__(self, client: GeminiClient = None):
        self.client = client or GeminiClient()

    def extract_claims(self, ad_text: str) -> List[Dict[str, Any]]:
        """
        Extracts testable product claims from marketing copy.
        Technique: Role-Based + Few-Shot + Structured Output Prompting.
        """
        if not ad_text or not ad_text.strip():
            raise ValueError("Advertisement text cannot be empty.")

        prompt = CLAIM_EXTRACTION_USER_TEMPLATE.format(ad_text=ad_text.strip())
        claims = self.client.generate_json(
            system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=prompt
        )

        if not isinstance(claims, list) or len(claims) == 0:
            raise ValueError("Could not extract any concrete product claims from the advertisement.")

        return claims

    def synthesize_customer_reviews(
        self,
        reviews_input: Union[str, pd.DataFrame, List[str]],
        claimed_features: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes customer reviews to extract ground-truth experiences for claimed features.
        Technique: Few-Shot + Structured Output Prompting.
        """
        formatted_reviews = parse_reviews_input(reviews_input)
        features_str = json.dumps(claimed_features)

        prompt = REVIEW_SYNTHESIS_USER_TEMPLATE.format(
            claimed_features=features_str,
            reviews_text=formatted_reviews
        )

        synthesized = self.client.generate_json(
            system_prompt=REVIEW_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=prompt
        )

        if not isinstance(synthesized, list):
            raise ValueError("Unexpected output format from customer review synthesis.")

        return synthesized

    def evaluate_expectation_gaps(
        self,
        claims: List[Dict[str, Any]],
        experiences: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compares claims against customer reality to calculate 0-100 expectation gap scores.
        Technique: Zero-Shot Comparative Reasoning + Structured Output Prompting.
        """
        claims_json = json.dumps(claims, indent=2)
        experiences_json = json.dumps(experiences, indent=2)

        prompt = GAP_COMPARISON_USER_TEMPLATE.format(
            claims_json=claims_json,
            experiences_json=experiences_json
        )

        audit_result = self.client.generate_json(
            system_prompt=GAP_COMPARISON_SYSTEM_PROMPT,
            user_prompt=prompt
        )

        if not isinstance(audit_result, dict):
            raise ValueError("Unexpected format from expectation gap evaluation.")

        # Post-process and ensure consistent score boundaries
        evaluations = audit_result.get("feature_evaluations", [])
        for item in evaluations:
            raw_score = float(item.get("gap_score", 0))
            clamped_score = max(0.0, min(100.0, round(raw_score, 1)))
            item["gap_score"] = clamped_score
            item["gap_category"] = categorize_gap_score(clamped_score)

        # Re-verify or compute overall gap score
        if evaluations:
            avg_score = round(sum(f["gap_score"] for f in evaluations) / len(evaluations), 1)
            # Use AI's overall score if present, else fallback to empirical average
            reported_overall = audit_result.get("overall_gap_score", avg_score)
            final_overall = max(0.0, min(100.0, float(reported_overall)))
            audit_result["overall_gap_score"] = round(final_overall, 1)
            audit_result["overall_category"] = categorize_gap_score(final_overall)
        else:
            audit_result["overall_gap_score"] = 0.0
            audit_result["overall_category"] = "Low"

        return audit_result

    def generate_final_report(self, audit_result: Dict[str, Any]) -> str:
        """
        Generates an executive-level diagnostic audit report in Markdown.
        Technique: Role-Based Diagnostic Synthesis.
        """
        evaluations = audit_result.get("feature_evaluations", [])
        summary_lines = []
        for f in evaluations:
            summary_lines.append(
                f"- Feature: {f.get('feature')}\n"
                f"  Claim: {f.get('advertised_claim')}\n"
                f"  Reality: {f.get('experienced_reality')}\n"
                f"  Gap Score: {f.get('gap_score')}/100 ({f.get('gap_category')} Gap)\n"
                f"  Verdict: {f.get('verdict')}\n"
                f"  Reasoning: {f.get('gap_reasoning')}"
            )
        evaluations_summary = "\n\n".join(summary_lines)

        prompt = EXECUTIVE_REPORT_USER_TEMPLATE.format(
            overall_gap_score=audit_result.get("overall_gap_score", 0),
            overall_category=audit_result.get("overall_category", "Unknown"),
            audit_headline=audit_result.get("audit_headline", "Product Claim Audit"),
            evaluations_summary=evaluations_summary
        )

        return self.client.generate_text(
            system_prompt=EXECUTIVE_REPORT_SYSTEM_PROMPT,
            user_prompt=prompt
        )

    def run_full_audit(
        self,
        ad_text: str,
        reviews_input: Union[str, pd.DataFrame, List[str]]
    ) -> Dict[str, Any]:
        """
        Executes the entire end-to-end TRUTHGAP pipeline.
        Returns all intermediate and final results in a structured bundle.
        """
        # Step 1: Extract product claims
        claims = self.extract_claims(ad_text)
        claimed_features = [c.get("feature", "") for c in claims if c.get("feature")]

        # Step 2: Extract real customer experiences
        experiences = self.synthesize_customer_reviews(reviews_input, claimed_features)

        # Step 3: Comparative gap evaluation
        gap_results = self.evaluate_expectation_gaps(claims, experiences)

        # Step 4: Generate executive report
        report_md = self.generate_final_report(gap_results)

        return {
            "claims": claims,
            "experiences": experiences,
            "gap_results": gap_results,
            "report_markdown": report_md
        }
