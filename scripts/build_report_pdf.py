"""Render docs/REPORT.md to docs/REPORT.pdf (reportlab; Markdown subset).

    python scripts/build_report_pdf.py

Supports: # / ## / ### headings, paragraphs, bullet and numbered lists, pipe
tables, images ![alt](path) (relative to docs/), --- rules, **bold**, *italic*,
`code`, [text](url). The PDF is git-ignored (regenerate with this script).
"""
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SRC = DOCS / "REPORT.md"
OUT = DOCS / "REPORT.pdf"

fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(fonts / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(fonts / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(fonts / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(fonts / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(fonts / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ACC = colors.HexColor("#1F4E79")
body = ParagraphStyle("body", fontName="DV", fontSize=9, leading=12.6, alignment=TA_LEFT, spaceAfter=5)
h1 = ParagraphStyle("h1", parent=body, fontName="DV-B", fontSize=17, leading=21, textColor=ACC, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=body, fontName="DV-B", fontSize=13, leading=16, textColor=ACC, spaceBefore=10, spaceAfter=5)
h3 = ParagraphStyle("h3", parent=body, fontName="DV-B", fontSize=10.5, leading=13, textColor=ACC, spaceBefore=7, spaceAfter=3)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
cell = ParagraphStyle("cell", parent=body, fontSize=7.4, leading=9.2, spaceAfter=0)
cellh = ParagraphStyle("cellh", parent=cell, fontName="DV-B", textColor=colors.white)
cap = ParagraphStyle("cap", parent=body, fontName="DV-I", fontSize=7.6, leading=9.5, textColor=colors.HexColor("#555555"))


def inline(t: str) -> str:
    t = escape(t)
    t = re.sub(r"`([^`]+)`", r'<font name="DVM" size="8">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)  # links -> text
    return t


def table(rows):
    data = []
    ncol = len(rows[0])
    for i, r in enumerate(rows):
        r = (r + [""] * ncol)[:ncol]
        data.append([Paragraph(inline(c.strip()), cellh if i == 0 else cell) for c in r])
    avail = A4[0] - 36 * mm
    lens = [max(len(rows[j][c]) if c < len(rows[j]) else 0 for j in range(len(rows))) for c in range(ncol)]
    weights = [min(max(l, 6), 38) for l in lens]
    widths = [avail * w / sum(weights) for w in weights]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACC),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D1DA")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def image(path: str, alt: str):
    p = (DOCS / path).resolve()
    if not p.exists():
        return [Paragraph(f"[missing figure: {escape(path)}]", cap)]
    from PIL import Image as PI

    w, h = PI.open(p).size
    maxw, maxh = A4[0] - 40 * mm, 95 * mm
    sc = min(maxw / w, maxh / h)
    return [KeepTogether([Image(str(p), w * sc, h * sc), Paragraph(inline(alt), cap), Spacer(1, 4)])]


def build():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    flow, para, i = [], [], 0

    def flush():
        nonlocal para
        if para:
            flow.append(Paragraph(inline(" ".join(para)), body))
            para = []

    first_h1 = True
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            flush(); i += 1; continue
        if ln.startswith("# "):
            flush()
            flow.append(Spacer(1, 40))
            flow.append(Paragraph(inline(ln[2:]), ParagraphStyle("title", parent=h1, fontSize=20, leading=25)))
            flow.append(HRFlowable(width="100%", thickness=1.2, color=ACC, spaceAfter=8))
            first_h1 = False
        elif ln.startswith("## "):
            flush(); flow.append(Paragraph(inline(ln[3:]), h2))
        elif ln.startswith("### "):
            flush(); flow.append(Paragraph(inline(ln[4:]), h3))
        elif ln.strip() == "---":
            flush(); flow.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#BBBBBB"), spaceBefore=4, spaceAfter=6))
        elif ln.startswith("!["):
            flush()
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
            if m:
                flow += image(m.group(2), m.group(1))
        elif ln.lstrip().startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = lines[i].strip().strip("|").split("|")
                if not all(re.fullmatch(r"\s*:?-{2,}:?\s*", c) for c in cells):
                    rows.append(cells)
                i += 1
            flow.append(table(rows)); flow.append(Spacer(1, 6))
            continue
        elif re.match(r"^\s*([*-])\s+", ln) or re.match(r"^\s*\d+\.\s+", ln):
            flush()
            num = re.match(r"^\s*(\d+)\.\s+(.*)", ln)
            text = num.group(2) if num else re.sub(r"^\s*[*-]\s+", "", ln)
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and not re.match(r"^\s*([*-]|\d+\.)\s+", lines[i]):
                text += " " + lines[i].strip(); i += 1
            flow.append(Paragraph(inline(text), bullet, bulletText=(num.group(1) + "." if num else "•")))
            continue
        else:
            para.append(ln.strip())
        i += 1
    flush()

    def footer(c, d):
        c.saveState(); c.setFont("DV", 7); c.setFillColor(colors.HexColor("#777777"))
        c.drawString(18 * mm, 10 * mm, "ECG+PPG alarm verification: leakage-audited two-dataset study (research prototype, not a medical device)")
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=17 * mm, title="Multimodal Cardiac Signal Verification",
                            author="ecg-ppg-pipeline")
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    build()
