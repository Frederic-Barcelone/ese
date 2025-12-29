# corpus_metadata/corpus_abbreviations/Z_tests/small_script.py

import json
import sys
from pathlib import Path

# Make "corpus_metadata/corpus_abbreviations" importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser, document_to_markdown
from B_parsing.B03_table_extractor import TableExtractor


def main():
    pdf_path = Path(
        "/Users/frederictetard/Projects/ese/Pdfs/JIR-559438-eosinophilic-granulomatosis-with-polyangiitis-presenting-as-.pdf"
    )

    # B01: Parse PDF to DocumentGraph (text blocks)
    parser = PDFToDocGraphParser()
    doc = parser.parse(str(pdf_path))

    # B03: Extract tables and add to DocumentGraph
    table_extractor = TableExtractor()
    doc = table_extractor.populate_document_graph(doc, str(pdf_path))

    # Stats
    total_blocks = sum(len(p.blocks) for p in doc.pages.values())
    total_tables = sum(len(p.tables) for p in doc.pages.values())
    roles = {}
    for p in doc.pages.values():
        for b in p.blocks:
            roles[b.role] = roles.get(b.role, 0) + 1

    print("PDF:", pdf_path.name)
    print(f"Pages: {len(doc.pages)}")
    print(f"Blocks: {total_blocks}")
    print(f"Tables: {total_tables}")
    print("\nBlocks by role:")
    for k, v in sorted(roles.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")

    # Show tables
    if total_tables:
        print("\nTables:")
        for t in doc.iter_tables():
            print(f"  Page {t.page_num}: {t.table_type.value}, {len(t.logical_rows)} rows")
            if t.metadata.get("headers"):
                headers = list(t.metadata["headers"].values())[:3]
                print(f"    Headers: {headers}")

    # Write markdown preview
    md = document_to_markdown(doc, skip_header_footer=True)
    out_path = pdf_path.with_suffix(".preview.md")
    out_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown: {out_path}")

    # Write tables JSON
    tables_json = []
    for t in doc.iter_tables():
        tables_json.append({
            "page": t.page_num,
            "type": t.table_type.value,
            "headers": list(t.metadata.get("headers", {}).values()),
            "rows": t.logical_rows,
        })
    if tables_json:
        json_path = pdf_path.with_suffix(".tables.json")
        json_path.write_text(json.dumps(tables_json, indent=2), encoding="utf-8")
        print(f"Tables JSON: {json_path}")


if __name__ == "__main__":
    main()
