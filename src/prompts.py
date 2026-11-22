"""
src/prompts.py - Prompt Engineering Library for TRUTHGAP

This module contains the prompt templates powering the TRUTHGAP system.
Each prompt demonstrates a core prompt engineering technique required for
academic evaluation and robust AI execution:

1. ROLE-BASED PROMPTING: Assigning specialized expert personas (e.g., Advertising Auditor, Consumer Protection Analyst).
2. STRUCTURED-OUTPUT PROMPTING: Enforcing JSON schemas for deterministic parsing.
3. FEW-SHOT PROMPTING: Providing concrete input-output examples to guide complex feature extraction.
4. ZERO-SHOT COMPARATIVE PROMPTING: Direct logical reasoning to compare claims vs. reality without prior task-specific samples.
"""

# ==============================================================================
# 1. CLAIM EXTRACTION PROMPT (Role-Based + Few-Shot + Structured Output)
# ==============================================================================
CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are a Senior Consumer Protection and Advertising Standards Auditor.
Your objective is to objectively analyze marketing copy and advertisements to extract concrete, verifiable product claims.
Ignore marketing fluff (e.g., "best ever", "revolutionary") and focus only on testable features (e.g., battery life, durability, connectivity, noise cancellation, materials).

You must respond ONLY with a valid JSON array matching this exact schema:
[
  {
    "claim_id": "CLM-1",
    "feature": "Feature Name (e.g., Battery Life, Noise Cancellation)",
    "advertised_claim": "The specific claim made in the text",
    "claim_metric": "Measurable benchmark if specified (e.g., '48 hours', 'IPX8', 'Active ANC'), or 'Qualitative' if not quantitative",
    "criticality": "High | Medium | Low"
  }
]
"""

CLAIM_EXTRACTION_USER_TEMPLATE = """Here is an example of how to extract claims:

[FEW-SHOT EXAMPLE]
Input Ad: "Introducing the TitanPods Pro. Enjoy an industry-leading 50-hour battery life and military-grade IPX7 water resistance for intense workouts. Features instant one-tap Bluetooth 5.3 pairing."
Output JSON:
[
  {{
    "claim_id": "CLM-1",
    "feature": "Battery Life",
    "advertised_claim": "50-hour total playback battery life",
    "claim_metric": "50 hours",
    "criticality": "High"
  }},
  {{
    "claim_id": "CLM-2",
    "feature": "Water Resistance",
    "advertised_claim": "Military-grade IPX7 water resistance for workouts",
    "claim_metric": "IPX7",
    "criticality": "High"
  }},
  {{
    "claim_id": "CLM-3",
    "feature": "Bluetooth Pairing",
    "advertised_claim": "Instant one-tap Bluetooth 5.3 pairing",
    "claim_metric": "Bluetooth 5.3 Instant Pairing",
    "criticality": "Medium"
  }}
]
[END OF EXAMPLE]

Now analyze the following product advertisement:

--- PRODUCT ADVERTISEMENT ---
{ad_text}
-----------------------------

Extract all verifiable product claims into the specified JSON format. Return ONLY the raw JSON array.
"""


# ==============================================================================
# 2. REVIEW SYNTHESIS PROMPT (Role-Based + Few-Shot + Structured Output)
# ==============================================================================
REVIEW_SYNTHESIS_SYSTEM_PROMPT = """You are a Data-Driven Consumer Experience Researcher.
Your job is to read customer reviews, extract feedback related to specific product features, and determine the ground-truth user experience.

You must respond ONLY with a valid JSON array matching this exact schema:
[
  {
    "feature": "Feature Name matching the claimed feature",
    "experienced_performance": "Summary of what users actually experienced in real life",
    "sentiment": "Positive | Neutral | Negative",
    "evidence_count": "Estimated number of mentions in reviews",
    "sample_user_quotes": ["Short direct quotes from the reviews"]
  }
]
"""

REVIEW_SYNTHESIS_USER_TEMPLATE = """Here is an example of feature-level review synthesis:

[FEW-SHOT EXAMPLE]
Target Features: ["Battery Life", "Noise Cancellation"]
Customer Reviews:
1. "Lasts only about 12 hours before dying, nowhere near the 40 hours promised."
2. "ANC cuts out low drone sounds, but screams and chatter pass right through."
3. "Battery died mid-flight on an 8-hour flight. Very disappointed."

Output JSON:
[
  {{
    "feature": "Battery Life",
    "experienced_performance": "Users report battery depleting within 8 to 12 hours, severely missing claims.",
    "sentiment": "Negative",
    "evidence_count": 2,
    "sample_user_quotes": ["Lasts only about 12 hours", "Battery died mid-flight on an 8-hour flight"]
  }},
  {{
    "feature": "Noise Cancellation",
    "experienced_performance": "Filters low continuous humming but fails to isolate human voices or irregular noise.",
    "sentiment": "Neutral",
    "evidence_count": 1,
    "sample_user_quotes": ["ANC cuts out low drone sounds, but screams pass right through"]
  }}
]
[END OF EXAMPLE]

Target Features to evaluate:
{claimed_features}

Customer Reviews to analyze:
--- CUSTOMER REVIEWS ---
{reviews_text}
------------------------

Synthesize the actual customer experience for the target features. Return ONLY the raw JSON array.
"""


# ==============================================================================
# 3. EXPECTATION GAP COMPARISON PROMPT (Zero-Shot Comparative Reasoning)
# ==============================================================================
GAP_COMPARISON_SYSTEM_PROMPT = """You are an Algorithmic Product Audit System.
Your task is to perform Zero-Shot comparative evaluation between Advertised Product Claims and Ground-Truth Customer Experiences.

For each feature, calculate an Expectation Gap Score from 0 to 100:
- 0 to 30 (Low Gap): Product largely meets or exceeds claims. Customers are satisfied.
- 31 to 60 (Moderate Gap): Product partially meets claims with noticeable limitations or minor compromises.
- 61 to 100 (High Gap): Severe discrepancy. The product significantly fails to deliver on advertised performance.

You must respond ONLY with a valid JSON object matching this schema:
{
  "feature_evaluations": [
    {
      "feature": "Feature Name",
      "advertised_claim": "Summary of claimed performance",
      "experienced_reality": "Summary of customer experience",
      "gap_score": 75,
      "gap_category": "High",
      "verdict": "Verified | Exaggerated | Misleading",
      "gap_reasoning": "Clear explanation of why this gap score was assigned"
    }
  ],
  "overall_gap_score": 62,
  "overall_category": "High",
  "audit_headline": "One-line executive takeaway of product claim accuracy"
}
"""

GAP_COMPARISON_USER_TEMPLATE = """Evaluate the expectation gap between the following claims and user experiences:

--- ADVERTISED CLAIMS ---
{claims_json}

--- CUSTOMER EXPERIENCES ---
{experiences_json}

Perform rigorous Zero-Shot comparative reasoning. Assign objective gap scores (0-100) based strictly on evidence.
Return ONLY the raw JSON object.
"""


# ==============================================================================
# 4. EXECUTIVE REPORT PROMPT (Role-Based Diagnostic Synthesis)
# ==============================================================================
EXECUTIVE_REPORT_SYSTEM_PROMPT = """You are the Chief AI Product Integrity Officer preparing an academic audit report.
Produce a professional, structured executive markdown report summarizing the expectation gap analysis.
Use clear headings, bullet points, and an objective, analytical tone.
"""

EXECUTIVE_REPORT_USER_TEMPLATE = """Based on the following expectation gap audit results:

--- AUDIT DATA ---
Overall Gap Score: {overall_gap_score}/100 ({overall_category} Expectation Gap)
Audit Headline: {audit_headline}
Feature Breakdown:
{evaluations_summary}
------------------

Generate an Executive Audit Report in Markdown containing:
1. **Executive Summary**: A concise overview of claim veracity vs. customer reality.
2. **Major Expectation Gaps**: Key features where marketing promises deviated most severely.
3. **Consumer Caution Advisory**: Specific red flags buyers should consider before purchasing.
4. **Manufacturer Accountability Recommendations**: Constructive advice for the brand to align marketing with real-world product capabilities.
"""
