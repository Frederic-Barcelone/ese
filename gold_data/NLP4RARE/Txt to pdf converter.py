#!/usr/bin/env python3
"""
Convert text files to PDF for NLP4RARE dataset.
Creates standard PDFs from .txt files in dev/, test/, train/ folders.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch


def txt_to_pdf(txt_path: Path, pdf_path: Path) -> None:
    """Convert a single text file to PDF."""
    # Read text content
    text = txt_path.read_text(encoding="utf-8")
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )
    
    # Setup styles
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,  # Line spacing
        spaceAfter=6,
    )
    
    # Build content - split by paragraphs (double newlines) or lines
    story = []
    paragraphs = text.split("\n\n")
    
    for para in paragraphs:
        # Clean up the paragraph and preserve single newlines as line breaks
        cleaned = para.strip()
        if cleaned:
            # Replace single newlines with <br/> for line breaks within paragraphs
            cleaned = cleaned.replace("\n", "<br/>")
            story.append(Paragraph(cleaned, body_style))
            story.append(Spacer(1, 6))
    
    # Build the PDF
    if story:
        doc.build(story)
    else:
        # Handle empty files - create PDF with placeholder
        story.append(Paragraph("(Empty document)", body_style))
        doc.build(story)


def convert_folder(folder_path: Path) -> int:
    """Convert all txt files in a folder to PDFs. Returns count of converted files."""
    count = 0
    txt_files = list(folder_path.glob("*.txt"))
    
    for txt_file in txt_files:
        pdf_file = txt_file.with_suffix(".pdf")
        try:
            txt_to_pdf(txt_file, pdf_file)
            print(f"  ✓ {txt_file.name} → {pdf_file.name}")
            count += 1
        except Exception as e:
            print(f"  ✗ {txt_file.name}: {e}")
    
    return count


def main():
    base_path = Path("/Users/frederictetard/Projects/ese/gold_data/NLP4RARE")
    folders = ["dev", "test", "train"]
    
    total = 0
    for folder_name in folders:
        folder_path = base_path / folder_name
        if folder_path.exists():
            print(f"\nProcessing {folder_name}/")
            count = convert_folder(folder_path)
            total += count
            print(f"  Converted {count} files")
        else:
            print(f"\n⚠ Folder not found: {folder_path}")
    
    print(f"\n✓ Total: {total} PDFs created")


if __name__ == "__main__":
    main()