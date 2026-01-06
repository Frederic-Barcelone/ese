# corpus_metadata/corpus_abbreviations/Z_tests/test_parsing_with_unstructured_local.py

from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_import_path() -> Path:
    """
    Necesitas que Python vea `A_core/` y `B_parsing/` como top-level modules,
    porque tu código importa `from A_core...` y `from B_parsing...`.

    Este test vive en:
      corpus_metadata/corpus_abbreviations/Z_tests/...

    Así que añadimos a sys.path:
      corpus_metadata/corpus_abbreviations/
    """
    this_file = Path(__file__).resolve()
    pkg_root = this_file.parents[1]  # .../corpus_metadata/corpus_abbreviations
    sys.path.insert(0, str(pkg_root))
    return pkg_root


def main() -> None:
    pkg_root = _bootstrap_import_path()

    # ✅ Pon aquí tu PDF (sin CLI)
    PDF_FILE = Path(
        "/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/documents/4.pdf"
    )

    if not PDF_FILE.exists():
        raise FileNotFoundError(f"PDF no encontrado: {PDF_FILE}")

    from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser, document_to_markdown
    from B_parsing.B02_doc_graph import ContentRole

    parser = PDFToDocGraphParser(
        config={
            "strategy": "hi_res",      # prueba: "fast" si quieres algo más ligero
            "languages": ["eng"],      # OCR packs si hace falta
            "min_repeat_count": 3,
            "header_top_pct": 0.07,
            "footer_bottom_pct": 0.93,
            "skip_tables": True,
        }
    )

    doc = parser.parse(str(PDF_FILE))

    # Stats rápidos
    total_pages = len(doc.pages)
    total_blocks = sum(len(p.blocks) for p in doc.pages.values())
    by_role = {}
    for p in doc.pages.values():
        for b in p.blocks:
            by_role[b.role] = by_role.get(b.role, 0) + 1

    print("\n✅ PARSING OK")
    print(f"Pages:  {total_pages}")
    print(f"Blocks: {total_blocks}")
    print("By role:")
    for r, c in sorted(by_role.items(), key=lambda x: x[0].value):
        print(f"  - {r.value}: {c}")

    # Export markdown para inspección humana/LLM/debug
    md = document_to_markdown(doc, include_tables=True, skip_header_footer=True)
    out_path = Path(__file__).with_suffix(".out.md")
    out_path.write_text(md, encoding="utf-8")
    print(f"\n📝 Markdown: {out_path}")


if __name__ == "__main__":
    main()
