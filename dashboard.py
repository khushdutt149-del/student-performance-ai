"""
dashboard.py
----------------------------------------------------
All Plotly figure builders used by the Dashboard page.
Kept separate from app.py so chart logic can be tested,
reused, or restyled independently of the page layout.
----------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

# Shared dark-theme palette
BG_COLOR = "rgba(0,0,0,0)"
FONT_COLOR = "#E2E8F0"
GRID_COLOR = "rgba(148,163,184,0.15)"
ACCENT_BLUE = "#3B82F6"
ACCENT_PURPLE = "#8B5CF6"
ACCENT_GREEN = "#22C55E"
ACCENT_ORANGE = "#F97316"
ACCENT_RED = "#EF4444"


def _apply_dark_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=FONT_COLOR, family="Poppins, sans-serif"),
        height=height,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def grade_gauge(prediction: float, color: str) -> go.Figure:
    """Gauge chart showing predicted grade on a 0-20 scale."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(float(prediction), 2),
        number={"suffix": " / 20", "font": {"size": 34, "color": FONT_COLOR}},
        gauge={
            "axis": {"range": [0, 20], "tickcolor": FONT_COLOR},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 1,
            "bordercolor": GRID_COLOR,
            "steps": [
                {"range": [0, 8], "color": "rgba(239,68,68,0.25)"},
                {"range": [8, 10], "color": "rgba(249,115,22,0.25)"},
                {"range": [10, 14], "color": "rgba(234,179,8,0.25)"},
                {"range": [14, 16], "color": "rgba(59,130,246,0.25)"},
                {"range": [16, 20], "color": "rgba(34,197,94,0.25)"},
            ],
        },
    ))
    return _apply_dark_layout(fig, "Predicted Grade", height=320)


def risk_meter(risk_label: str) -> go.Figure:
    """Gauge chart representing categorical risk on a 0-100 scale."""
    risk_scale = {
        "Very Low": 10, "Low": 30, "Medium": 55, "High": 78, "Very High": 95,
    }
    risk_colors = {
        "Very Low": ACCENT_GREEN, "Low": ACCENT_BLUE, "Medium": "#EAB308",
        "High": ACCENT_ORANGE, "Very High": ACCENT_RED,
    }
    value = risk_scale.get(risk_label, 50)
    color = risk_colors.get(risk_label, ACCENT_BLUE)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 30, "color": FONT_COLOR}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": FONT_COLOR},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 1,
            "bordercolor": GRID_COLOR,
        },
    ))
    return _apply_dark_layout(fig, f"Risk Level: {risk_label}", height=320)


def performance_pie(category: str) -> go.Figure:
    """Donut chart highlighting where the current prediction sits among categories."""
    categories = ["Excellent", "Very Good", "Average", "Needs Improvement", "Critical"]
    colors = [ACCENT_GREEN, ACCENT_BLUE, "#EAB308", ACCENT_ORANGE, ACCENT_RED]
    values = [1 if c == category else 0.15 for c in categories]

    fig = go.Figure(go.Pie(
        labels=categories,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0F172A", width=2)),
        textinfo="label",
        sort=False,
    ))
    return _apply_dark_layout(fig, "Performance Category", height=380)


def lifestyle_radar(inputs: dict[str, Any]) -> go.Figure:
    """Radar chart comparing lifestyle-related attributes (all on a 1-5 scale)."""
    labels = ["Family Relationship", "Free Time", "Going Out", "Weekday Alcohol", "Weekend Alcohol", "Health"]
    values = [
        inputs.get("famrel", 0), inputs.get("freetime", 0), inputs.get("goout", 0),
        inputs.get("Dalc", 0), inputs.get("Walc", 0), inputs.get("health", 0),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + values[:1],
        theta=labels + labels[:1],
        fill="toself",
        line_color=ACCENT_PURPLE,
        fillcolor="rgba(139,92,246,0.3)",
        name="Lifestyle",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, range=[0, 5], gridcolor=GRID_COLOR, color=FONT_COLOR),
            angularaxis=dict(gridcolor=GRID_COLOR, color=FONT_COLOR),
        ),
        showlegend=False,
    )
    return _apply_dark_layout(fig, "Lifestyle Analysis", height=420)


def study_analysis_bar(inputs: dict[str, Any]) -> go.Figure:
    """Bar chart summarizing study-related factors."""
    labels = ["Study Time", "Travel Time", "Failures", "G1", "G2"]
    values = [
        inputs.get("studytime", 0), inputs.get("traveltime", 0),
        inputs.get("failures", 0), inputs.get("G1", 0), inputs.get("G2", 0),
    ]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=[ACCENT_BLUE, ACCENT_PURPLE, ACCENT_RED, ACCENT_GREEN, ACCENT_GREEN]),
        text=values, textposition="outside",
    ))
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return _apply_dark_layout(fig, "Study Analysis", height=380)


def family_analysis_bar(inputs: dict[str, Any]) -> go.Figure:
    """Bar chart summarizing family-related factors."""
    labels = ["Mother Edu", "Father Edu", "Family Relationship", "Family Size (GT3=1)"]
    values = [
        inputs.get("Medu", 0), inputs.get("Fedu", 0), inputs.get("famrel", 0),
        1 if inputs.get("famsize") == "GT3" else 0,
    ]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=ACCENT_PURPLE),
        text=values, textposition="outside",
    ))
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return _apply_dark_layout(fig, "Family Analysis", height=380)


def attendance_chart(inputs: dict[str, Any]) -> go.Figure:
    """Simple bar showing absences relative to a healthy attendance band."""
    absences = inputs.get("absences", 0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Absences"], y=[absences],
        marker_color=ACCENT_RED if absences > 10 else ACCENT_GREEN,
        text=[absences], textposition="outside", width=[0.4],
    ))
    fig.add_hline(y=10, line_dash="dash", line_color=ACCENT_ORANGE,
                  annotation_text="High-risk threshold (10)", annotation_font_color=FONT_COLOR)
    fig.update_yaxes(range=[0, max(20, absences + 5)], gridcolor=GRID_COLOR)
    fig.update_xaxes(gridcolor=GRID_COLOR)
    return _apply_dark_layout(fig, "Attendance Overview", height=340)


def performance_timeline(history_df: pd.DataFrame) -> go.Figure:
    """Line chart of predicted grade across all predictions made this session."""
    fig = go.Figure()
    if history_df.empty:
        fig.add_annotation(
            text="No prediction history yet — run a prediction to populate this chart.",
            showarrow=False, font=dict(color=FONT_COLOR, size=14),
        )
    else:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history_df) + 1)),
            y=history_df["Predicted Grade"],
            mode="lines+markers",
            line=dict(color=ACCENT_BLUE, width=3),
            marker=dict(size=9, color=ACCENT_PURPLE),
            name="Predicted Grade",
        ))
        fig.update_xaxes(title="Prediction #", dtick=1, gridcolor=GRID_COLOR)
        fig.update_yaxes(title="Grade (0-20)", range=[0, 20], gridcolor=GRID_COLOR)
    return _apply_dark_layout(fig, "Prediction Timeline (This Session)", height=380)
