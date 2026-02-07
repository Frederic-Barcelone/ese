#!/usr/bin/env python3
"""
Generate NLM-Gene gold standard for gene evaluation.

Parses NLM-Gene BioC XML files, filters to annotations matching the pipeline's
Entrez→HGNC mapping (from Orphadata), generates PDFs from title+abstract text,
and outputs a gold JSON file.

Filtering rules:
- Only type=Gene annotations (skip GENERIF, STARGENE, Other, Domain)
- Exclude code=222 (family names: "cytokine", "transcription factor")
- Exclude code=333 (non-specific abbreviations: "IFN")
- For comma-separated NCBI Gene IDs, take first that maps to HGNC
- Deduplicate: unique (doc_id, hgnc_symbol) pairs per document

Usage:
    python generate_nlm_gene_gold.py
    python generate_nlm_gene_gold.py --split=test
    python generate_nlm_gene_gold.py --skip=10 --no-pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = SCRIPT_DIR / "corpus" / "Corpus"
XML_DIR = CORPUS_DIR / "FINAL"
TEST_PMIDS = CORPUS_DIR / "Pmidlist.Test.txt"
TRAIN_PMIDS = CORPUS_DIR / "Pmidlist.Train.txt"
PDF_DIR = SCRIPT_DIR / "pdfs"
GOLD_OUTPUT = SCRIPT_DIR.parent / "nlm_gene_gold.json"
ORPHADATA_GENES = SCRIPT_DIR.parent.parent / "ouput_datasources" / "2025_08_orphadata_genes.json"

# Codes to exclude
EXCLUDE_CODES = {"222", "333"}


def load_entrez_to_hgnc() -> dict[str, str]:
    """Build Entrez Gene ID → HGNC symbol mapping from Orphadata."""
    with open(ORPHADATA_GENES, "r", encoding="utf-8") as f:
        genes = json.load(f)

    mapping: dict[str, str] = {}
    for g in genes:
        entrez_id = g.get("entrez_id")
        symbol = g.get("hgnc_symbol")
        if entrez_id and symbol:
            mapping[str(entrez_id)] = symbol
    return mapping


def load_split_pmids() -> tuple[set[str], set[str]]:
    """Load test/train PMID split lists."""
    test_pmids: set[str] = set()
    train_pmids: set[str] = set()

    if TEST_PMIDS.exists():
        test_pmids = {line.strip() for line in TEST_PMIDS.read_text().splitlines() if line.strip()}
    if TRAIN_PMIDS.exists():
        train_pmids = {line.strip() for line in TRAIN_PMIDS.read_text().splitlines() if line.strip()}

    return test_pmids, train_pmids


def parse_xml(xml_path: Path, entrez_to_hgnc: dict[str, str]) -> tuple[str, str, list[dict]]:
    """Parse a BioC XML file and extract title, abstract, and gene annotations.

    Returns (title, abstract, annotations_list).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    title = ""
    abstract = ""
    annotations: list[dict] = []
    seen_symbols: set[str] = set()

    for doc in root.iter("document"):
        # Extract passages
        for passage in doc.iter("passage"):
            ptype_elem = passage.find('infon[@key="type"]')
            text_elem = passage.find("text")
            if ptype_elem is not None and text_elem is not None and text_elem.text:
                ptype = ptype_elem.text
                if ptype == "title":
                    title = text_elem.text
                elif ptype == "abstract":
                    abstract = text_elem.text

        # Extract gene annotations
        for ann in doc.iter("annotation"):
            infons = {i.get("key"): i.text for i in ann.findall("infon")}
            ann_type = infons.get("type", "")
            code = infons.get("code", "")

            # Only Gene type, exclude family (222) and non-specific (333)
            if ann_type != "Gene":
                continue
            if code in EXCLUDE_CODES:
                continue

            ncbi_ids_raw = infons.get("NCBI Gene identifier", "")
            if not ncbi_ids_raw:
                continue

            # Handle comma-separated IDs — take first that maps
            ncbi_ids = [x.strip() for x in ncbi_ids_raw.split(",")]
            hgnc_symbol = None
            matched_ncbi = None
            for ncbi_id in ncbi_ids:
                if ncbi_id in entrez_to_hgnc:
                    hgnc_symbol = entrez_to_hgnc[ncbi_id]
                    matched_ncbi = ncbi_id
                    break

            if not hgnc_symbol:
                continue

            # Deduplicate per document
            if hgnc_symbol in seen_symbols:
                continue
            seen_symbols.add(hgnc_symbol)

            ann_text_elem = ann.find("text")
            ann_text = ann_text_elem.text if ann_text_elem is not None else ""

            annotations.append({
                "symbol": hgnc_symbol,
                "name": ann_text,
                "ncbi_gene_id": matched_ncbi,
            })
        break  # Only first document in XML

    return title, abstract, annotations


def generate_pdf(pmid: str, title: str, abstract: str, output_path: Path) -> None:
    """Generate a simple PDF from title and abstract text."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 14)
    safe_title = title.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 7, safe_title)
    pdf.ln(5)

    # Abstract
    pdf.set_font("Helvetica", "", 10)
    safe_abstract = abstract.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 5, safe_abstract)

    pdf.output(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NLM-Gene gold standard")
    parser.add_argument("--split", choices=["test", "train", "all"], default="all",
                        help="Which split to process (default: all)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N documents")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process")
    args = parser.parse_args()

    print("Loading Entrez→HGNC mapping from Orphadata...")
    entrez_to_hgnc = load_entrez_to_hgnc()
    print(f"  {len(entrez_to_hgnc)} Entrez→HGNC mappings loaded")

    print("Loading split PMID lists...")
    test_pmids, train_pmids = load_split_pmids()
    print(f"  Test: {len(test_pmids)} PMIDs, Train: {len(train_pmids)} PMIDs")

    # Find XML files to process
    xml_files = sorted(XML_DIR.glob("*.BioC.XML"))
    print(f"  Found {len(xml_files)} XML files")

    # Filter by split
    if args.split == "test":
        xml_files = [x for x in xml_files if x.stem.replace(".BioC", "") in test_pmids]
    elif args.split == "train":
        xml_files = [x for x in xml_files if x.stem.replace(".BioC", "") in train_pmids]

    # Apply skip and max
    if args.skip:
        xml_files = xml_files[args.skip:]
    if args.max_docs:
        xml_files = xml_files[:args.max_docs]

    print(f"  Processing {len(xml_files)} XML files ({args.split} split)")

    all_annotations: list[dict] = []
    docs_with_genes = 0
    docs_without_genes = 0
    pdfs_generated = 0

    for i, xml_path in enumerate(xml_files):
        pmid = xml_path.stem.replace(".BioC", "")
        title, abstract, annotations = parse_xml(xml_path, entrez_to_hgnc)

        if not title and not abstract:
            continue

        # Determine split
        if pmid in test_pmids:
            split = "test"
        elif pmid in train_pmids:
            split = "train"
        else:
            split = "unknown"

        doc_id = f"{pmid}.pdf"

        if annotations:
            docs_with_genes += 1
            for ann in annotations:
                all_annotations.append({
                    "doc_id": doc_id,
                    "symbol": ann["symbol"],
                    "name": ann["name"],
                    "ncbi_gene_id": ann["ncbi_gene_id"],
                    "split": split,
                })
        else:
            docs_without_genes += 1

        # Generate PDF
        if not args.no_pdf:
            pdf_out = PDF_DIR / split / doc_id
            pdf_out.parent.mkdir(parents=True, exist_ok=True)
            if not pdf_out.exists():
                generate_pdf(pmid, title, abstract, pdf_out)
                pdfs_generated += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(xml_files)} files...")

    # Write gold JSON
    gold = {
        "corpus": "NLM-Gene",
        "description": "NLM-Gene corpus filtered to Orphadata Entrez→HGNC mappings",
        "genes": {
            "total": len(all_annotations),
            "annotations": all_annotations,
        },
    }

    with open(GOLD_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    # Summary
    unique_symbols = {a["symbol"] for a in all_annotations}
    test_anns = [a for a in all_annotations if a["split"] == "test"]
    train_anns = [a for a in all_annotations if a["split"] == "train"]

    print(f"\n{'='*60}")
    print(f"NLM-Gene Gold Standard Generated")
    print(f"{'='*60}")
    print(f"  Total annotations:    {len(all_annotations)}")
    print(f"  Unique HGNC symbols:  {len(unique_symbols)}")
    print(f"  Docs with genes:      {docs_with_genes}")
    print(f"  Docs without genes:   {docs_without_genes}")
    print(f"  Test annotations:     {len(test_anns)} ({len({a['doc_id'] for a in test_anns})} docs)")
    print(f"  Train annotations:    {len(train_anns)} ({len({a['doc_id'] for a in train_anns})} docs)")
    print(f"  PDFs generated:       {pdfs_generated}")
    print(f"  Gold output:          {GOLD_OUTPUT}")


if __name__ == "__main__":
    main()
