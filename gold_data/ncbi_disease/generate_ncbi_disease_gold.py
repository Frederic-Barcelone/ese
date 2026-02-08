#!/usr/bin/env python3
"""
Generate NCBI Disease gold standard for disease evaluation.

Parses NCBI Disease Corpus PubTator files, generates PDFs from title+abstract text,
and outputs a gold JSON file for disease evaluation on PubMed scientific literature.

Source: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
Format: PubTator (PMID|t|title, PMID|a|abstract, PMID\tstart\tend\tmention\ttype\tconcept)
Splits: train (592 abstracts, ~5433 mentions), dev (100, ~924), test (100, ~941)

Usage:
    python generate_ncbi_disease_gold.py
    python generate_ncbi_disease_gold.py --split=test
    python generate_ncbi_disease_gold.py --no-pdf --max-docs=50
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = SCRIPT_DIR / "corpus"
PDF_DIR = SCRIPT_DIR / "pdfs"
GOLD_OUTPUT = SCRIPT_DIR.parent / "ncbi_disease_gold.json"

# Download URLs for PubTator format files
PUBTATOR_URLS = {
    "train": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip",
    "dev": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip",
    "test": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip",
}

# Expected PubTator filenames inside the zips
PUBTATOR_FILES = {
    "train": "NCBItrainset_corpus.txt",
    "dev": "NCBIdevelopset_corpus.txt",
    "test": "NCBItestset_corpus.txt",
}


def download_corpus(splits: list[str]) -> None:
    """Download PubTator files for requested splits."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        txt_path = CORPUS_DIR / PUBTATOR_FILES[split]
        if txt_path.exists():
            print(f"  {split}: already exists at {txt_path}")
            continue

        url = PUBTATOR_URLS[split]
        zip_path = CORPUS_DIR / f"{split}.zip"

        print(f"  {split}: downloading from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "ESE-Pipeline/0.8"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        zip_path.write_bytes(data)
        print(f"    Downloaded {len(data)} bytes")

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CORPUS_DIR)
        zip_path.unlink()
        print(f"    Extracted to {CORPUS_DIR}")


def parse_pubtator(file_path: Path, split: str) -> tuple[list[dict], list[dict]]:
    """Parse a PubTator format file.

    Returns (documents, annotations) where:
    - documents: [{"pmid": str, "title": str, "abstract": str, "split": str}]
    - annotations: [{"doc_id": str, "text": str, "type": str, "concept_id": str, "split": str}]
    """
    documents: list[dict] = []
    annotations: list[dict] = []

    text = file_path.read_text(encoding="utf-8")
    blocks = text.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        title = ""
        abstract = ""
        pmid = ""
        block_annotations: list[dict] = []
        seen_mentions: set[str] = set()

        for line in lines:
            if "|t|" in line:
                parts = line.split("|t|", 1)
                pmid = parts[0].strip()
                title = parts[1].strip() if len(parts) > 1 else ""
            elif "|a|" in line:
                parts = line.split("|a|", 1)
                if not pmid:
                    pmid = parts[0].strip()
                abstract = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Annotation line: PMID\tstart\tend\tmention\ttype\tconcept
                fields = line.split("\t")
                if len(fields) >= 5:
                    mention = fields[3].strip()
                    entity_type = fields[4].strip()
                    concept_id = fields[5].strip() if len(fields) > 5 else ""

                    # Only include Disease and Modifier types
                    # (the corpus uses "Disease" and sometimes "Modifier" for disease-related mentions)
                    if entity_type not in ("Disease", "Modifier"):
                        continue

                    # Deduplicate by mention text (case-insensitive) per document
                    mention_key = mention.lower().strip()
                    if mention_key in seen_mentions:
                        continue
                    seen_mentions.add(mention_key)

                    doc_id = f"{pmid}.pdf"
                    block_annotations.append({
                        "doc_id": doc_id,
                        "text": mention,
                        "type": "DISEASE",
                        "concept_id": concept_id,
                        "split": split,
                    })

        if pmid and (title or abstract):
            documents.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "split": split,
            })
            annotations.extend(block_annotations)

    return documents, annotations


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
    parser = argparse.ArgumentParser(description="Generate NCBI Disease gold standard")
    parser.add_argument("--split", choices=["test", "dev", "train", "all"], default="all",
                        help="Which split to process (default: all)")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    parser.add_argument("--no-download", action="store_true", help="Skip corpus download")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process")
    args = parser.parse_args()

    splits = [args.split] if args.split != "all" else ["train", "dev", "test"]

    # Download corpus
    if not args.no_download:
        print("Downloading NCBI Disease Corpus...")
        download_corpus(splits)
    else:
        print("Skipping download (using existing files)")

    # Parse PubTator files
    all_documents: list[dict] = []
    all_annotations: list[dict] = []

    for split in splits:
        txt_path = CORPUS_DIR / PUBTATOR_FILES[split]
        if not txt_path.exists():
            print(f"  WARNING: {txt_path} not found, skipping {split} split")
            continue

        print(f"Parsing {split} split: {txt_path}...")
        docs, anns = parse_pubtator(txt_path, split)
        all_documents.extend(docs)
        all_annotations.extend(anns)
        print(f"  {len(docs)} documents, {len(anns)} disease annotations")

    # Apply max-docs limit
    if args.max_docs:
        all_documents = all_documents[:args.max_docs]
        valid_doc_ids = {f"{d['pmid']}.pdf" for d in all_documents}
        all_annotations = [a for a in all_annotations if a["doc_id"] in valid_doc_ids]
        print(f"Limited to {len(all_documents)} documents, {len(all_annotations)} annotations")

    # Generate PDFs
    pdfs_generated = 0
    if not args.no_pdf:
        print("Generating PDFs...")
        for doc in all_documents:
            pdf_out = PDF_DIR / doc["split"] / f"{doc['pmid']}.pdf"
            pdf_out.parent.mkdir(parents=True, exist_ok=True)
            if not pdf_out.exists():
                generate_pdf(doc["pmid"], doc["title"], doc["abstract"], pdf_out)
                pdfs_generated += 1

            if pdfs_generated % 100 == 0 and pdfs_generated > 0:
                print(f"  Generated {pdfs_generated} PDFs...")

    # Write gold JSON
    gold = {
        "corpus": "NCBI-Disease",
        "description": "NCBI Disease Corpus - PubMed abstracts with disease annotations (MeSH/OMIM)",
        "source": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/",
        "license": "Public (NIH)",
        "diseases": {
            "total": len(all_annotations),
            "annotations": all_annotations,
        },
    }

    with open(GOLD_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    # Summary
    unique_mentions = {a["text"].lower() for a in all_annotations}
    by_split: dict[str, list] = {}
    for a in all_annotations:
        by_split.setdefault(a["split"], []).append(a)

    print(f"\n{'='*60}")
    print("NCBI Disease Gold Standard Generated")
    print(f"{'='*60}")
    print(f"  Total annotations:     {len(all_annotations)}")
    print(f"  Unique disease texts:  {len(unique_mentions)}")
    print(f"  Total documents:       {len(all_documents)}")
    for split in sorted(by_split):
        split_anns = by_split[split]
        split_docs = {a["doc_id"] for a in split_anns}
        print(f"  {split:5s} annotations:   {len(split_anns)} ({len(split_docs)} docs)")
    print(f"  PDFs generated:        {pdfs_generated}")
    print(f"  Gold output:           {GOLD_OUTPUT}")


if __name__ == "__main__":
    main()
