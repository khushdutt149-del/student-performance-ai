"""
utils.py
----------------------------------------------------
Shared helper functions for the AI Student Performance
Analytics application.

Contains:
    - Feature option constants (used by app.py to build the form)
    - Performance category / risk classification
    - Dynamic AI recommendation engine
    - Session-state based prediction history management
    - Small formatting helpers

Keeping these here means app.py, dashboard.py, and report.py
can all import the same single source of truth instead of
duplicating logic.
----------------------------------------------------
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

# ==========================================================
# Feature Option Constants
# (Matches the UCI "Student Performance" dataset schema)
# ==========================================================

SCHOOL_OPTIONS = ["GP", "MS"]
SEX_OPTIONS = ["F", "M"]
ADDRESS_OPTIONS = ["U", "R"]
FAMSIZE_OPTIONS = ["LE3", "GT3"]
PSTATUS_OPTIONS = ["T", "A"]
JOB_OPTIONS = ["teacher", "health", "services", "at_home", "other"]
REASON_OPTIONS = ["home", "reputation", "course", "other"]
GUARDIAN_OPTIONS = ["mother", "father", "other"]
YES_NO_OPTIONS = ["yes", "no"]

MODEL_FILENAME = "student_performance_pipeline (1).pkl"

ALL_FEATURE_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2",
]

assert len(ALL_FEATURE_COLUMNS) == 32, "Feature schema must contain exactly 32 features."


# ==========================================================
# Performance Category + Risk Classification
# ==========================================================

def get_performance_category(grade: float) -> dict[str, str]:
    """
    Map a predicted final grade (0-20 scale) to a performance
    category, a risk level, and a theme color / emoji used
    consistently across the Prediction, Dashboard, and Report pages.
    """
    if grade >= 16:
        return {
            "category": "Excellent",
            "emoji": "🌟",
            "risk": "Very Low",
            "color": "#22C55E",
            "message": "Outstanding academic performance. Keep up the excellent work!",
        }
    elif grade >= 14:
        return {
            "category": "Very Good",
            "emoji": "🏆",
            "risk": "Low",
            "color": "#3B82F6",
            "message": "Strong academic performance with solid fundamentals.",
        }
    elif grade >= 10:
        return {
            "category": "Average",
            "emoji": "👍",
            "risk": "Medium",
            "color": "#EAB308",
            "message": "Passing performance, but there is meaningful room to improve.",
        }
    elif grade >= 8:
        return {
            "category": "Needs Improvement",
            "emoji": "⚠️",
            "risk": "High",
            "color": "#F97316",
            "message": "Performance is below the passing threshold. Intervention is recommended.",
        }
    else:
        return {
            "category": "Critical",
            "emoji": "❌",
            "risk": "Very High",
            "color": "#EF4444",
            "message": "Performance indicates serious academic risk. Immediate support is needed.",
        }


# ==========================================================
# AI Recommendation Engine
# ==========================================================

def generate_recommendations(inputs: dict[str, Any], prediction: float) -> list[str]:
    """
    Generate 8-10 personalized, rule-based recommendations from
    the student's input profile and predicted grade. Rules fire
    conditionally based on the actual input values; if fewer than
    8 conditional rules apply, general best-practice guidance is
    appended so the list always contains 8-10 items.
    """
    recs: list[str] = []

    if inputs.get("studytime", 0) <= 1:
        recs.append("📚 Increase weekly study time — aim for at least 5-10 hours per week to build stronger fundamentals.")

    if inputs.get("failures", 0) > 0:
        recs.append("🎯 Focus extra effort on subjects with past failures; consider a structured revision plan for those topics.")

    if inputs.get("absences", 0) > 10:
        recs.append("🏫 Improve class attendance — missed classes are strongly correlated with lower grades.")

    if inputs.get("internet") == "no":
        recs.append("🌐 Gain access to digital learning resources (library, online tutorials, e-books) to supplement classroom learning.")

    if inputs.get("health", 5) <= 2:
        recs.append("❤️ Prioritize physical health — better sleep, nutrition, and exercise routines support cognitive performance.")

    if inputs.get("famrel", 5) <= 2:
        recs.append("👨‍👩‍👧 Consider family counselling or open communication sessions to strengthen home support systems.")

    if inputs.get("higher") == "no":
        recs.append("🎓 Explore the long-term benefits of higher education — even undecided students benefit from keeping options open.")

    if inputs.get("Dalc", 1) >= 3 or inputs.get("Walc", 1) >= 3:
        recs.append("🚫 Reduce alcohol consumption, which can negatively affect concentration, memory, and attendance.")

    if inputs.get("goout", 3) >= 4 and inputs.get("studytime", 3) <= 2:
        recs.append("⚖️ Rebalance time between socializing and studying to protect academic performance.")

    if inputs.get("freetime", 3) >= 4 and inputs.get("studytime", 3) <= 2:
        recs.append("⏰ Redirect some free time into structured revision sessions or practice tests.")

    if inputs.get("schoolsup") == "no" and inputs.get("failures", 0) > 0:
        recs.append("🏛️ Consider enrolling in school-provided educational support programs.")

    if inputs.get("paid") == "no" and inputs.get("failures", 0) > 0:
        recs.append("📖 Extra paid tutoring classes may help close specific knowledge gaps quickly.")

    if inputs.get("traveltime", 1) >= 3:
        recs.append("🚌 Account for long commute time by building a fixed study block into the daily schedule.")

    if inputs.get("romantic") == "yes" and inputs.get("studytime", 3) <= 2:
        recs.append("💞 Balance time spent in relationships with dedicated, distraction-free study periods.")

    g1 = inputs.get("G1")
    g2 = inputs.get("G2")
    if g1 is not None and g2 is not None and g2 < g1:
        recs.append("📉 Grades show a declining trend between grading periods — a mid-term check-in with a teacher or mentor is recommended.")

    if prediction >= 16:
        recs.append("✅ Maintain current study habits and consider mentoring peers to reinforce your own mastery.")

    # Guarantee 8-10 recommendations regardless of how many rules fired
    general_pool = [
        "🧭 Set specific, measurable weekly academic goals and track progress.",
        "🗓️ Use a study planner or digital calendar to stay consistent across subjects.",
        "🤝 Form or join a peer study group for collaborative learning.",
        "🧑‍🏫 Schedule periodic check-ins with teachers to identify blind spots early.",
        "💤 Maintain a consistent sleep schedule to support memory consolidation.",
        "📱 Limit distracting screen time during dedicated study hours.",
        "🧘 Practice stress-management techniques ahead of exams.",
        "📝 Review class notes within 24 hours to improve long-term retention.",
    ]

    i = 0
    while len(recs) < 8 and i < len(general_pool):
        if general_pool[i] not in recs:
            recs.append(general_pool[i])
        i += 1

    return recs[:10]


# ==========================================================
# Prediction History (session-state backed)
# ==========================================================

HISTORY_KEY = "prediction_history"


def init_session_state() -> None:
    """Ensure all session-state keys used by the app exist."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    if "latest_prediction" not in st.session_state:
        st.session_state["latest_prediction"] = None


def add_to_history(student_inputs: dict[str, Any], prediction: float, percentage: float, meta: dict[str, str]) -> None:
    """Append a completed prediction to the session-state history table."""
    init_session_state()
    record = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted Grade": round(float(prediction), 2),
        "Percentage": round(float(percentage), 1),
        "Performance": meta["category"],
        "Risk": meta["risk"],
    }
    st.session_state[HISTORY_KEY].append(record)
    st.session_state["latest_prediction"] = {
        "inputs": student_inputs,
        "prediction": float(prediction),
        "percentage": float(percentage),
        "meta": meta,
    }


def get_history_df() -> pd.DataFrame:
    init_session_state()
    if not st.session_state[HISTORY_KEY]:
        return pd.DataFrame(columns=["Date", "Predicted Grade", "Percentage", "Performance", "Risk"])
    return pd.DataFrame(st.session_state[HISTORY_KEY])


def clear_history() -> None:
    st.session_state[HISTORY_KEY] = []
    st.session_state["latest_prediction"] = None


def history_to_csv_bytes() -> bytes:
    df = get_history_df()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ==========================================================
# Formatting Helpers
# ==========================================================

def format_grade(grade: float) -> str:
    return f"{grade:.2f} / 20"


def format_percentage(pct: float) -> str:
    return f"{pct:.1f}%"


def safe_load_pipeline(joblib_module, path: str = MODEL_FILENAME):
    """
    Load the trained pipeline with defensive error handling.
    Returns (pipeline, error_message). error_message is None on success.
    """
    try:
        pipeline = joblib_module.load(path)
        return pipeline, None
    except FileNotFoundError:
        return None, (
            f"Model file '{path}' was not found in the working directory. "
            "Make sure it is uploaded alongside app.py before predicting."
        )
    except Exception as exc:  # noqa: BLE001 - surface any load error to the UI
        return None, f"Failed to load the model pipeline: {exc}"
