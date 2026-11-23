"""
app.py - TRUTHGAP Streamlit Web Dashboard

A prompt-engineered AI system for detecting expectation gaps between
product advertising claims and real-world consumer experiences.
"""

import io
import json
import streamlit as st
import pandas as pd

from src.config import GAP_THRESHOLDS, DEFAULT_MODEL, get_api_key_diagnostics
from src.analyzers import TruthGapAnalyzer, categorize_gap_score
from src.visualizer import (
    create_gauge_chart,
    create_feature_bar_chart,
    create_radar_chart,
    create_verdict_pie_chart,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & High-Contrast Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TRUTHGAP | AI Expectation Gap Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ultra-High Contrast Styling for guaranteed readability across all monitors & themes
st.markdown(
    """
    <style>
    /* Font family and global reset */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Executive Hero Header - High Contrast Dark Obsidian Card */
    .hero-container {
        background: #0B1120;
        padding: 32px 36px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        border: 2px solid #1E293B;
    }
    .hero-title {
        font-size: 2.9rem;
        font-weight: 900;
        letter-spacing: -1px;
        line-height: 1.1;
        margin: 0;
        color: #FFFFFF !important;
    }
    .hero-title .brand-highlight {
        color: #F59E0B !important; /* Bright Gold Accent for extreme visibility */
        text-shadow: 0 0 15px rgba(245, 158, 11, 0.4);
    }
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF !important; /* PURE WHITE: 100% visible, zero wash-out */
        margin-top: 12px;
        line-height: 1.4;
        letter-spacing: -0.2px;
    }
    .hero-meta {
        font-size: 0.95rem;
        color: #E2E8F0 !important; /* Crisp Light Silver */
        margin-top: 10px;
        font-weight: 500;
    }

    /* Section Headings - Guaranteed High-Contrast Pill Headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 900;
        color: #0F172A !important;
        background: #F1F5F9;
        border-left: 6px solid #2563EB;
        padding: 10px 16px;
        border-radius: 8px;
        margin: 18px 0 12px 0;
        letter-spacing: -0.3px;
        display: block;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Metric Scorecard Cards */
    .metric-card {
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 800;
        color: #334155 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #0F172A !important;
        margin-top: 6px;
        letter-spacing: -0.5px;
    }

    /* Modern Gap Category Badges */
    .badge-low {
        background-color: #DCFCE7 !important;
        color: #14532D !important;
        border: 2px solid #86EFAC !important;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #FEF3C7 !important;
        color: #78350F !important;
        border: 2px solid #FCD34D !important;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #FEE2E2 !important;
        color: #7F1D1D !important;
        border: 2px solid #FCA5A5 !important;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Feature Breakdown Card */
    .card-box {
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .pill-ad {
        background: #0284C7;
        color: #FFFFFF !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 800;
        display: inline-block;
        margin-bottom: 6px;
    }
    .pill-rev {
        background: #E11D48;
        color: #FFFFFF !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 800;
        display: inline-block;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Preset Datasets (Multiple Real-World Scenarios)
# -----------------------------------------------------------------------------
DEMO_PRESETS = {
    "🎧 Wireless Earbuds (Severe Discrepancy - High Gap)": {
        "ad": (
            "Experience pure acoustic revolution with the SonicPulse Pro Wireless Earbuds!\n"
            "- Unmatched 48-Hour Ultra Endurance Battery Life on a single charge.\n"
            "- Flagship Active Noise Cancellation (ANC) eliminating 99% of background ambient noise.\n"
            "- Military-Grade IPX8 Waterproof Rating engineered for extreme sports and deep submersion.\n"
            "- Lightning HyperCharge: Enjoy 5 hours of playback with just 10 minutes of charging.\n"
            "- Instant One-Tap Bluetooth 5.3 Auto-Connect pairing up to 50 meters."
        ),
        "reviews": (
            "1. \"The sound is decent but battery life is awful. It claims 48 hours but died in barely 13 hours on a full charge.\"\n"
            "2. \"ANC barely filters anything on flights or the subway. High pitched sounds and voices pass right through.\"\n"
            "3. \"Took them for a swim because it advertised IPX8 waterproof. The left earbud broke permanently within 10 minutes.\"\n"
            "4. \"Fast charging claim is completely false. Took nearly 3 hours to reach 100% using the provided cable.\"\n"
            "5. \"Bluetooth drops constantly when my phone is in my back pocket. Nowhere near the 50m range advertised.\"\n"
            "6. \"Great fit for gym workouts and clear mic for phone calls, but battery is not even one third of what was promised.\"\n"
            "7. \"ANC is practically non-existent. Only cuts out low air conditioner hums, nothing else.\"\n"
            "8. \"Sweat from a cardio session caused the touch controls to malfunction and turn off unexpectedly.\""
        ),
    },
    "⌚ Smart Fitness Watch (Moderate Compromise - Moderate Gap)": {
        "ad": (
            "Introducing the ApexTrack V3 Smartwatch – Built for elite athletes.\n"
            "- Clinical-grade ECG and Continuous Heart Rate Monitoring with 99.8% medical accuracy.\n"
            "- 14-Day Marathon Battery Life on standby and normal use.\n"
            "- 1000 Nits Ultra-Bright AMOLED Display readable in blazing direct sunlight.\n"
            "- Scratch-Proof Diamond Sapphire Crystal Glass face."
        ),
        "reviews": (
            "1. \"The screen is bright and very easy to read even in direct noon sun. Very impressed with the display.\"\n"
            "2. \"Battery lasts about 6 to 7 days with normal notification use. Decent, but half of the 14 days advertised.\"\n"
            "3. \"Heart rate monitor fluctuates wildly during high intensity interval training. Off by 25 bpm compared to chest strap.\"\n"
            "4. \"Glass holds up well with no scratches after a month of mountain biking.\"\n"
            "5. \"Battery easily lasts a full week. Not 14 days, but still very practical.\"\n"
            "6. \"ECG feature occasionally fails to take readings if skin is cold or sweaty.\""
        ),
    },
    "👟 Performance Running Shoes (Verified Promises - Low Gap)": {
        "ad": (
            "AeroStride Pro Ultra Running Shoes:\n"
            "- Featherweight Carbon Fiber Plate providing 15% enhanced energetic spring propulsion.\n"
            "- Breathable Matrix Engineered Mesh keeping feet cool during marathons.\n"
            "- Continental Wet-Traction Grip Outsole preventing slippage on wet asphalt.\n"
            "- True-to-size ergonomic arch support reducing heel fatigue."
        ),
        "reviews": (
            "1. \"The energy return from the carbon plate is phenomenal. Shaved 40 seconds off my 5K personal best.\"\n"
            "2. \"Incredible grip on wet rainy roads. Never felt slippery or unstable on sharp turns.\"\n"
            "3. \"Super breathable mesh. Ran a 20km training session in 32C heat and feet stayed dry.\"\n"
            "4. \"Fit is exactly true to size and arch support eliminated my plantar fasciitis pain.\"\n"
            "5. \"Extremely light and responsive shoe. Delivers exactly what was advertised.\""
        ),
    },
}

# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
DEFAULT_PRESET_KEY = list(DEMO_PRESETS.keys())[0]

if "ad_input_area" not in st.session_state:
    st.session_state["ad_input_area"] = DEMO_PRESETS[DEFAULT_PRESET_KEY]["ad"]

if "reviews_text_area" not in st.session_state:
    st.session_state["reviews_text_area"] = DEMO_PRESETS[DEFAULT_PRESET_KEY]["reviews"]

if "audit_bundle" not in st.session_state:
    st.session_state["audit_bundle"] = None


def apply_preset(preset_name: str):
    """Callback to forcefully update session state widget values and reload UI."""
    data = DEMO_PRESETS[preset_name]
    st.session_state["ad_input_area"] = data["ad"]
    st.session_state["reviews_text_area"] = data["reviews"]


# -----------------------------------------------------------------------------
# 4. Sidebar: Configuration, Demo Selector & Academic Info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 12px 0;">
            <span style="font-size: 2.8rem;">⚖️</span>
            <h2 style="margin: 0; color: #0F172A; font-weight: 900; letter-spacing: -0.5px;">TRUTHGAP</h2>
            <div style="font-size: 0.85rem; color: #475569; font-weight: 700; margin-top: 4px;">Expectation Gap Auditor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### ⚡ Live Demo Scenarios")
    st.caption("Select a pre-loaded real-world case study for rapid demonstration:")
    
    selected_preset = st.selectbox(
        "Choose Evaluation Scenario:",
        list(DEMO_PRESETS.keys()),
        index=0,
    )

    if st.button("📥 Load Selected Scenario", use_container_width=True, type="secondary"):
        apply_preset(selected_preset)
        st.success("✅ Scenario data loaded into input fields!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔑 API Key Status")
    diag = get_api_key_diagnostics()
    if diag["is_configured"]:
        st.success(f"✅ Active ({diag['source']})")
    else:
        st.error("❌ Key Not Detected")
        with st.expander("🔍 Diagnostic Details"):
            st.caption(
                "Ensure your key is configured in **Streamlit Cloud > App Settings > Secrets** using valid TOML format:\n\n"
                '```toml\nGEMINI_API_KEY = "your_actual_key_here"\n```\n'
                "⚠️ In TOML, values must be enclosed in double quotes."
            )
            if diag.get("secrets_error"):
                st.warning(f"Secrets Error: {diag['secrets_error']}")
            elif diag.get("secrets_keys_found"):
                st.info(f"Available keys in secrets: {diag['secrets_keys_found']}")
            else:
                st.info("No secrets detected in st.secrets.")

    st.markdown("---")
    st.markdown("### 🧠 System Architecture")
    st.markdown(
        f"""
        - **Inference Model:** `{DEFAULT_MODEL}`
        - **Temperature:** `0.2` (Deterministic)
        - **Prompt Methodologies:**
          - 🎭 *Role-Based Persona Prompting*
          - 🎯 *Few-Shot Demonstration*
          - 📐 *Structured JSON Schema Output*
          - ⚖️ *Zero-Shot Comparative Reasoning*
        """
    )

    st.markdown("---")
    st.markdown("### 📊 Scoring Taxonomy")
    st.markdown(
        """
        - <span class="badge-low">0 – 30 : Low Gap (Verified)</span><br>
        - <span class="badge-moderate" style="margin-top:6px;">31 – 60 : Moderate Gap (Exaggerated)</span><br>
        - <span class="badge-high" style="margin-top:6px;">61 – 100 : High Gap (Misleading)</span>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 5. High-Contrast Hero Header (Pure White + Gold Accent)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">
            TRUTH<span class="brand-highlight">GAP</span> ⚖️
        </div>
        <div class="hero-subtitle">
            A Prompt-Engineered AI System for Detecting Expectation Gaps Between Product Claims and Consumer Experience
        </div>
        <div class="hero-meta">
            <strong>Third-Year Engineering Capstone Project</strong> &bull; Powered by Google Gemini AI &bull; Interactive Plotly Analytics
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 6. Two-Column Input Section: Ad Copy & Customer Reviews
# -----------------------------------------------------------------------------
col_ad, col_rev = st.columns([1, 1], gap="large")

with col_ad:
    st.markdown('<div class="section-title">1. Product Advertisement</div>', unsafe_allow_html=True)
    st.caption("Paste marketing copy, promotional claims, or product descriptions:")
    
    ad_text = st.text_area(
        label="Product Advertisement Text",
        value=st.session_state["ad_input_area"],
        height=240,
        placeholder="e.g. Introducing the all-new UltraPhone. Featuring 72-hour battery life and unbreakable sapphire glass...",
        key="ad_input_area",
        label_visibility="collapsed",
    )

with col_rev:
    st.markdown('<div class="section-title">2. Customer Experiences</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Choose review source:",
        ["✍️ Direct Text Input", "📁 Upload Reviews CSV"],
        horizontal=True,
        key="input_mode_radio",
    )

    reviews_payload = None

    if input_mode == "✍️ Direct Text Input":
        reviews_payload = st.text_area(
            label="Customer Reviews Text",
            value=st.session_state["reviews_text_area"],
            height=200,
            placeholder="1. 'The battery died after just 12 hours.'\n2. 'Screen cracked on first minor drop.'",
            key="reviews_text_area",
            label_visibility="collapsed",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV containing reviews",
            type=["csv"],
            help="Upload a CSV with a column named 'Review_Text', 'review', or 'comment'",
        )
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.success(f"Uploaded CSV with {len(df_uploaded)} reviews.")
                
                # Column selection
                text_cols = list(df_uploaded.select_dtypes(include=["object", "string"]).columns)
                default_col = 0
                for idx, col in enumerate(text_cols):
                    if any(term in col.lower() for term in ["review", "text", "comment", "feedback"]):
                        default_col = idx
                        break
                selected_col = st.selectbox("Select review column:", text_cols, index=default_col)
                st.dataframe(df_uploaded[[selected_col]].head(3), use_container_width=True)
                reviews_payload = df_uploaded[selected_col]
            except Exception as e:
                st.error(f"Error reading CSV: {e}")


# -----------------------------------------------------------------------------
# 7. Audit Execution Trigger
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
col_btn, col_info = st.columns([1.2, 2.8])

with col_btn:
    run_audit = st.button("🚀 Run TRUTHGAP Audit", type="primary", use_container_width=True)

with col_info:
    st.markdown(
        """
        <div style="padding: 8px 0; color: #334155; font-size: 0.95rem; font-weight: 600;">
            <strong>Multi-Stage Pipeline:</strong>
            Claim Extraction (Few-Shot) &rarr; Review Synthesis (Few-Shot) &rarr; Zero-Shot Comparative Gap Engine &rarr; Diagnostic Report.
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_audit:
    # Input Validations
    if not ad_text or not ad_text.strip():
        st.warning("⚠️ Please provide product advertisement text in Section 1.")
    elif reviews_payload is None or (isinstance(reviews_payload, str) and not reviews_payload.strip()):
        st.warning("⚠️ Please provide customer reviews in Section 2.")
    else:
        try:
            with st.spinner("⚖️ TRUTHGAP is extracting claims and evaluating expectation gaps with Gemini..."):
                analyzer = TruthGapAnalyzer()
                audit_bundle = analyzer.run_full_audit(ad_text=ad_text, reviews_input=reviews_payload)
                st.session_state["audit_bundle"] = audit_bundle
                st.success("✅ Expectation Gap Audit Completed Successfully!")
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")


# -----------------------------------------------------------------------------
# 8. Results Dashboard Section
# -----------------------------------------------------------------------------
if st.session_state["audit_bundle"] is not None:
    bundle = st.session_state["audit_bundle"]
    gap_results = bundle["gap_results"]
    evaluations = gap_results.get("feature_evaluations", [])
    overall_score = gap_results.get("overall_gap_score", 0.0)
    overall_category = gap_results.get("overall_category", "Low")
    headline = gap_results.get("audit_headline", "")
    report_md = bundle.get("report_markdown", "")

    st.markdown("---")
    st.markdown('<div class="section-title">3. Audit Findings & Metric Analysis</div>', unsafe_allow_html=True)
    
    # Audit Headline Alert Banner
    if overall_category == "High":
        st.error(f"🚨 **Critical Discrepancy Detected:** {headline}")
    elif overall_category == "Moderate":
        st.warning(f"⚠️ **Moderate Discrepancy Detected:** {headline}")
    else:
        st.success(f"✅ **Claims Well Aligned:** {headline}")

    # Top KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall Gap Score</div>
                <div class="metric-value">{overall_score:.1f}<span style="font-size:1.1rem;color:#64748B;">/100</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        badge_class = f"badge-{overall_category.lower()}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Gap Classification</div>
                <div class="metric-value"><span class="{badge_class}">{overall_category} Gap</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Audited Features</div>
                <div class="metric-value">{len(evaluations)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        misleading_count = sum(1 for e in evaluations if e.get("verdict") == "Misleading")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Misleading Claims</div>
                <div class="metric-value" style="color:#B91C1C;">{misleading_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Dashboard Views
    tab_charts, tab_breakdown, tab_report, tab_raw = st.tabs([
        "📈 Interactive Analytics",
        "🔍 Feature Discrepancy Breakdown",
        "📄 AI Executive Audit Report",
        "🧩 Prompt Engineering & Traceability",
    ])

    # TAB 1: INTERACTIVE CHARTS
    with tab_charts:
        row1_col1, row1_col2 = st.columns([1, 1])
        with row1_col1:
            st.plotly_chart(create_gauge_chart(overall_score, overall_category), use_container_width=True)
        with row1_col2:
            st.plotly_chart(create_verdict_pie_chart(evaluations), use_container_width=True)

        row2_col1, row2_col2 = st.columns([1.2, 0.8])
        with row2_col1:
            st.plotly_chart(create_feature_bar_chart(evaluations), use_container_width=True)
        with row2_col2:
            radar_fig = create_radar_chart(evaluations)
            if radar_fig.data:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("Radar discrepancy profile requires 3 or more audited features.")

    # TAB 2: DETAILED FEATURE BREAKDOWN
    with tab_breakdown:
        st.markdown("### Feature-by-Feature Expectation Audits")
        for f in evaluations:
            feat_name = f.get("feature", "Feature")
            score = f.get("gap_score", 0)
            category = f.get("gap_category", "Low")
            verdict = f.get("verdict", "Unverified")
            claim = f.get("advertised_claim", "N/A")
            reality = f.get("experienced_reality", "N/A")
            reason = f.get("gap_reasoning", "N/A")
            
            badge_class = f"badge-{category.lower()}"

            st.markdown(
                f"""
                <div class="card-box">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <h3 style="margin:0; font-size:1.35rem; font-weight:900; color:#0F172A;">{feat_name}</h3>
                        <div>
                            <span class="{badge_class}">{category} Gap ({score:.0f}/100)</span>
                            <span style="font-weight:800; margin-left:10px; font-size:0.95rem; color:#1E293B;">Verdict: {verdict}</span>
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                        <div style="background:#F0F9FF; border:1.5px solid #0284C7; padding:14px; border-radius:10px;">
                            <span class="pill-ad">📢 Advertised Claim</span>
                            <p style="margin:6px 0 0 0; color:#0F172A; font-size:1rem; font-weight:600;">{claim}</p>
                        </div>
                        <div style="background:#FFF1F2; border:1.5px solid #E11D48; padding:14px; border-radius:10px;">
                            <span class="pill-rev">👤 Customer Reality (Experience)</span>
                            <p style="margin:6px 0 0 0; color:#0F172A; font-size:1rem; font-weight:600;">{reality}</p>
                        </div>
                    </div>
                    <div style="margin-top:14px; font-size:0.95rem; color:#1E293B; background:#F8FAFC; padding:12px 16px; border-radius:8px; border:1.5px solid #E2E8F0;">
                        <strong style="color:#0F172A; font-weight:800;">🔎 AI Diagnostic Reasoning:</strong> {reason}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # TAB 3: EXECUTIVE AUDIT REPORT
    with tab_report:
        st.markdown("### AI-Generated Product Integrity Audit Report")
        st.markdown(report_md)
        
        # Download report button
        st.download_button(
            label="📥 Download Audit Report (.md)",
            data=report_md,
            file_name="TruthGap_Audit_Report.md",
            mime="text/markdown",
            use_container_width=False,
        )

    # TAB 4: PROMPT ENGINEERING TRACEABILITY
    with tab_raw:
        st.markdown("### Prompt Engineering & Intermediate Traceability")
        st.caption("Inspect the structured JSON payloads generated by each prompt stage for technical evaluation:")
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("**1. Extracted Product Claims (Role-Based + Few-Shot Prompt):**")
            st.json(bundle.get("claims", []))
            
        with exp_col2:
            st.markdown("**2. Ground-Truth Experience Synthesis (Few-Shot Prompt):**")
            st.json(bundle.get("experiences", []))

        st.markdown("**3. Zero-Shot Comparative Gap Evaluation Schema:**")
        st.json(bundle.get("gap_results", {}))
