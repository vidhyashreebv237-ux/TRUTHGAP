"""
src/visualizer.py - Interactive Plotly Visualizations for TRUTHGAP

Creates modern, publication-ready interactive charts for the TRUTHGAP dashboard:
1. Expectation Gap Gauge / Speedometer (0-100 score with colored danger zones)
2. Feature-Level Gap Bar Chart (color-coded by Low, Moderate, High gap)
3. Multi-Feature Radar / Spider Chart (discrepancy profile across all features)
4. Claim Veracity Donut Chart (proportion of Verified, Exaggerated, Misleading claims)
"""

from typing import Any, Dict, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# Color Palette for Gap Categories
CATEGORY_COLORS = {
    "Low": "#2ECC71",       # Emerald Green: High trust / low gap
    "Moderate": "#F39C12",  # Amber/Orange: Noticeable discrepancy
    "High": "#E74C3C",      # Coral Red: Severe expectation gap
}

VERDICT_COLORS = {
    "Verified": "#2ECC71",
    "Exaggerated": "#F39C12",
    "Misleading": "#E74C3C",
}


def create_gauge_chart(score: float, category: str) -> go.Figure:
    """
    Renders an interactive speedometer gauge chart representing the Overall Gap Score (0-100).
    """
    color = CATEGORY_COLORS.get(category, "#E74C3C")
    
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"<b>Overall Expectation Gap</b><br><span style='font-size:16px;color:{color}'>{category} Gap Level</span>",
                "font": {"size": 20, "color": "#1E293B"},
            },
            number={"suffix": "/100", "font": {"size": 36, "color": "#0F172A", "family": "Arial"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#F8FAFC",
                "borderwidth": 2,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 30], "color": "rgba(46, 204, 113, 0.2)"},    # Low gap zone
                    {"range": [30, 60], "color": "rgba(243, 156, 18, 0.2)"},  # Moderate gap zone
                    {"range": [60, 100], "color": "rgba(231, 76, 60, 0.2)"},   # High gap zone
                ],
                "threshold": {
                    "line": {"color": "#1E293B", "width": 4},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        height=320,
        margin={"l": 25, "r": 25, "t": 60, "b": 25},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif"},
    )
    return fig


def create_feature_bar_chart(evaluations: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders a horizontal bar chart of expectation gap scores per product feature.
    """
    if not evaluations:
        return go.Figure()

    df = pd.DataFrame(evaluations)
    # Ensure columns exist
    df["gap_score"] = df["gap_score"].astype(float)
    df["color"] = df["gap_category"].map(CATEGORY_COLORS).fillna("#E74C3C")
    
    # Sort for visual hierarchy
    df = df.sort_values(by="gap_score", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["gap_score"],
            y=df["feature"],
            orientation="h",
            marker=dict(
                color=df["color"],
                line=dict(color="#FFFFFF", width=1.5),
            ),
            text=[f" {score:.0f} ({cat})" for score, cat in zip(df["gap_score"], df["gap_category"])],
            textposition="auto",
            customdata=list(zip(df["advertised_claim"], df["experienced_reality"], df["verdict"])),
            hovertemplate=(
                "<b>Feature:</b> %{y}<br>"
                "<b>Gap Score:</b> %{x}/100<br>"
                "<b>Verdict:</b> %{customdata[2]}<br>"
                "<b>Claim:</b> %{customdata[0]}<br>"
                "<b>Reality:</b> %{customdata[1]}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="<b>Feature-by-Feature Expectation Gap Scores</b>",
        title_font={"size": 18, "color": "#1E293B"},
        xaxis=dict(
            title="Expectation Gap Score (0 = Fully Aligned, 100 = Severe Discrepancy)",
            range=[0, 105],
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(title="", tickfont={"size": 13, "color": "#334155"}),
        height=max(280, len(df) * 55),
        margin={"l": 20, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
    )
    return fig


def create_radar_chart(evaluations: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders a radar / spider chart showing the multi-dimensional expectation gap profile.
    """
    if not evaluations or len(evaluations) < 3:
        # Radar charts require at least 3 points for a polygon
        return go.Figure()

    features = [e["feature"] for e in evaluations]
    scores = [e["gap_score"] for e in evaluations]
    
    # Close the radar polygon
    features_closed = features + [features[0]]
    scores_closed = scores + [scores[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=scores_closed,
            theta=features_closed,
            fill="toself",
            fillcolor="rgba(231, 76, 60, 0.25)",
            line=dict(color="#E74C3C", width=2.5),
            name="Expectation Gap",
            marker=dict(size=7, color="#E74C3C"),
        )
    )

    fig.update_layout(
        title="<b>Multi-Dimensional Discrepancy Profile</b>",
        title_font={"size": 18, "color": "#1E293B"},
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color="#64748B"),
                gridcolor="#E2E8F0",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#1E293B", family="Arial"),
                gridcolor="#E2E8F0",
            ),
            bgcolor="rgba(248, 250, 252, 0.5)",
        ),
        height=340,
        margin={"l": 30, "r": 30, "t": 50, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def create_verdict_pie_chart(evaluations: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders a donut chart illustrating the proportion of claims by Verdict
    (Verified, Exaggerated, Misleading).
    """
    if not evaluations:
        return go.Figure()

    df = pd.DataFrame(evaluations)
    verdict_counts = df["verdict"].value_counts().reset_index()
    verdict_counts.columns = ["Verdict", "Count"]
    
    colors = [VERDICT_COLORS.get(v, "#94A3B8") for v in verdict_counts["Verdict"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=verdict_counts["Verdict"],
                values=verdict_counts["Count"],
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        title="<b>Claim Veracity Distribution</b>",
        title_font={"size": 18, "color": "#1E293B"},
        height=320,
        margin={"l": 20, "r": 20, "t": 50, "b": 25},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig
