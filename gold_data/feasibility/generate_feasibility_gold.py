#!/usr/bin/env python3
"""
Generate PDFs for the RareDis-Feasibility gold standard dataset.

Converts plain text documents in docs/ to simple PDFs in pdfs/ for pipeline processing.
Each document simulates a clinical trial feasibility report or treatment guideline
for a rare disease in a specific country.

20 documents total:
- 10 paper-style (clinical trial reports)
- 10 guideline-style (clinical practice guidelines)

Usage:
    python generate_feasibility_gold.py
    python generate_feasibility_gold.py --max-docs=5
    python generate_feasibility_gold.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR / "docs"
PDF_DIR = SCRIPT_DIR / "pdfs"
GOLD_FILE = SCRIPT_DIR / "feasibility_gold.json"


def generate_pdf(text_path: Path, output_path: Path) -> None:
    """Generate a PDF from a plain text document.

    Uses fpdf2 for clinical text containing special characters.
    Lines beginning with '-' or bullets are treated as list items.
    """
    from fpdf import FPDF

    text = text_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    first_line = True

    for line in lines:
        safe_line = line.encode("latin-1", "replace").decode("latin-1")

        if not safe_line.strip():
            pdf.set_x(pdf.l_margin)
            pdf.ln(3)
            continue

        stripped = safe_line.strip()
        # Reset x to left margin before every render
        pdf.set_x(pdf.l_margin)

        # Main title: first non-empty line
        if first_line:
            first_line = False
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 6, stripped)
            pdf.ln(4)
            continue

        # Detect headings
        is_heading = False

        # Section headings: "1. Introduction", "2.1 Diagnostic Pathway", etc.
        if (len(stripped) > 2
                and stripped[0].isdigit()
                and ("." in stripped[:4])
                and len(stripped) < 120
                and not stripped[-1].isdigit()):
            is_heading = True

        # Keyword headings
        heading_keywords = [
            "Abstract", "Introduction", "Methods", "Results", "Discussion",
            "Eligibility Criteria", "Inclusion criteria", "Exclusion criteria",
            "Endpoints", "Operational Procedures", "Patient Disposition",
            "Study Design",
        ]
        if any(stripped.startswith(kw) for kw in heading_keywords):
            is_heading = True

        if is_heading:
            pdf.ln(2)
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 5, stripped)
            pdf.ln(1)
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 4, stripped)

    pdf.output(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PDFs for RareDis-Feasibility gold standard"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Max documents to convert (default: all)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available documents without generating PDFs"
    )
    args = parser.parse_args()

    # Find all text files
    txt_files = sorted(DOCS_DIR.glob("*.txt"))

    if not txt_files:
        print(f"No text files found in {DOCS_DIR}")
        return

    if args.list:
        print(f"Found {len(txt_files)} documents in {DOCS_DIR}:")
        for f in txt_files:
            print(f"  {f.name}")
        return

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    docs_to_process = txt_files[:args.max_docs] if args.max_docs else txt_files

    print(f"Generating {len(docs_to_process)} PDFs...")
    for i, txt_path in enumerate(docs_to_process, 1):
        pdf_name = txt_path.stem + ".pdf"
        pdf_path = PDF_DIR / pdf_name
        print(f"  [{i}/{len(docs_to_process)}] {txt_path.name} -> {pdf_name}")
        generate_pdf(txt_path, pdf_path)

    print(f"\nDone. {len(docs_to_process)} PDFs written to {PDF_DIR}")

    # Verify gold file exists
    if GOLD_FILE.exists():
        import json
        with open(GOLD_FILE) as f:
            gold = json.load(f)
        n_docs = len(gold.get("documents", []))
        print(f"Gold annotations: {GOLD_FILE} ({n_docs} documents)")
    else:
        print(f"WARNING: Gold file not found at {GOLD_FILE}")


if __name__ == "__main__":
    main()
