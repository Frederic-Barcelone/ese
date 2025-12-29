# corpus_metadata/corpus_abbreviations/Z_tests/small_script.py

import sys
from pathlib import Path

# ✅ Make "corpus_metadata/corpus_abbreviations" importable
ROOT = Path(__file__).resolve().parents[1]  # .../corpus_abbreviations
sys.path.insert(0, str(ROOT))

from pathlib import Path

from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser, document_to_markdown


def main():
    pdf_path = Path(
        "/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/documents/4.pdf"
    )

    parser = PDFToDocGraphParser(
        config={
            "min_repeat_count": 3,
            "y_tolerance": 3.0,
        }
    )

    doc = parser.parse(str(pdf_path))

    total_blocks = sum(len(p.blocks) for p in doc.pages.values())
    roles = {}
    for p in doc.pages.values():
        for b in p.blocks:
            roles[b.role] = roles.get(b.role, 0) + 1

    print("✅ PIPELINE OK (B01 + B02 only)")
    print("PDF:", pdf_path)
    print("Pages:", len(doc.pages))
    print("Blocks:", total_blocks)
    print("\nBy role:")
    for k, v in sorted(roles.items(), key=lambda x: str(x[0])):
        print(f"  - {k}: {v}")

    # ✅ updated call (no table placeholders exist anymore)
    md = document_to_markdown(
        doc,
        skip_header_footer=True,
    )

    out_path = pdf_path.with_suffix(".b01_b02.preview.md")
    out_path.write_text(md, encoding="utf-8")
    print("\n📝 Preview Markdown written to:\n ", out_path)


if __name__ == "__main__":
    main()
