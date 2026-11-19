
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "research_note.md"
OUTPUT = ROOT / "report.pdf"

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontSize=19,
    leading=24,
    alignment=TA_CENTER,
    spaceAfter=16
))

styles.add(ParagraphStyle(
    name="SectionHeading",
    parent=styles["Heading2"],
    fontSize=12,
    leading=16,
    spaceBefore=12,
    spaceAfter=6
))

styles.add(ParagraphStyle(
    name="ReportBody",
    parent=styles["BodyText"],
    fontSize=9,
    leading=13,
    spaceAfter=6
))

story = []

lines = SOURCE.read_text(encoding="utf-8").splitlines()

for line in lines:
    line = line.strip()

    if not line:
        story.append(Spacer(1, 3))
        continue

    if line.startswith("# "):
        story.append(Paragraph(
            escape(line[2:]), styles["ReportTitle"]
        ))

    elif line.startswith("## "):
        story.append(HRFlowable(
            width="100%", color=colors.lightgrey
        ))
        story.append(Paragraph(
            escape(line[3:]), styles["SectionHeading"]
        ))

    elif line.startswith("- "):
        story.append(Paragraph(
            "&#8226; " + escape(line[2:]),
            styles["ReportBody"]
        ))

    elif line.startswith("|"):
        # Preserve table content in readable text form.
        if "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            story.append(Paragraph(
                escape("     |     ".join(cells)),
                styles["ReportBody"]
            ))

    else:
        story.append(Paragraph(
            escape(line), styles["ReportBody"]
        ))

doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=20*mm,
    leftMargin=20*mm,
    topMargin=18*mm,
    bottomMargin=18*mm,
    title="Weight-Space Programming for Swedish Language Modelling"
)

doc.build(story)

print(f"PDF successfully created: {OUTPUT}")
