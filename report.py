"""
report.py
----------------------------------------------------
Generates a professional PDF report summarizing a single
student's prediction, using ReportLab's Platypus layout engine.
----------------------------------------------------
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, ListFlowable, ListItem
)

PRIMARY_COLOR = colors.HexColor("#3B82F6")
SECONDARY_COLOR = colors.HexColor("#8B5CF6")
DARK_TEXT = colors.HexColor("#1E293B")
MUTED_TEXT = colors.HexColor("#475569")


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=22, leading=26, textColor=PRIMARY_COLOR,
        spaceAfter=6, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontSize=11, leading=14, textColor=MUTED_TEXT,
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", fontSize=14, leading=18, textColor=SECONDARY_COLOR,
        spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="BodyMuted", fontSize=10, leading=14, textColor=DARK_TEXT,
    ))
    return styles


def generate_pdf_report(
    student_inputs: dict[str, Any],
    prediction: float,
    percentage: float,
    meta: dict[str, str],
    recommendations: list[str],
) -> bytes:
    """
    Build a complete PDF report and return it as raw bytes,
    suitable for st.download_button(data=...).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = _build_styles()
    story = []

    # ---- Header ----
    story.append(Paragraph("AI Student Performance Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')} &nbsp;|&nbsp; "
        f"Model: Random Forest Pipeline",
        styles["ReportSubtitle"],
    ))
    story.append(HRFlowable(width="100%", color=PRIMARY_COLOR, thickness=1.2))
    story.append(Spacer(1, 12))

    # ---- Prediction Summary ----
    story.append(Paragraph("Prediction Summary", styles["SectionHeading"]))
    summary_data = [
        ["Predicted Grade", f"{prediction:.2f} / 20"],
        ["Percentage", f"{percentage:.1f}%"],
        ["Performance Category", f"{meta['emoji']} {meta['category']}"],
        ["Risk Level", meta["risk"]],
    ]
    summary_table = Table(summary_data, colWidths=[7 * cm, 8 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_TEXT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(meta["message"], styles["BodyMuted"]))

    # ---- Student Profile ----
    story.append(Paragraph("Student Profile", styles["SectionHeading"]))
    profile_rows = [[k, str(v)] for k, v in student_inputs.items()]
    # Render the profile in a two-column grid (label | value, label | value)
    grid_rows = []
    for i in range(0, len(profile_rows), 2):
        left = profile_rows[i]
        right = profile_rows[i + 1] if i + 1 < len(profile_rows) else ["", ""]
        grid_rows.append([left[0], left[1], right[0], right[1]])

    profile_table = Table(grid_rows, colWidths=[3.5 * cm, 4 * cm, 3.5 * cm, 4 * cm])
    profile_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_TEXT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
    ]))
    story.append(profile_table)

    # ---- Recommendations ----
    story.append(Paragraph("AI Recommendations", styles["SectionHeading"]))
    rec_items = [ListItem(Paragraph(rec, styles["BodyMuted"]), spaceAfter=4) for rec in recommendations]
    story.append(ListFlowable(rec_items, bulletType="bullet", start="•"))

    # ---- Footer ----
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#CBD5E1"), thickness=0.75))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was generated automatically by the AI Student Performance Analytics "
        "application using a Random Forest regression pipeline trained on the UCI Student "
        "Performance dataset. Predictions are statistical estimates and should be used to "
        "support, not replace, professional academic guidance.",
        styles["BodyMuted"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
