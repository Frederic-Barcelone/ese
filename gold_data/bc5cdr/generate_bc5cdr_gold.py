#!/usr/bin/env python3
"""
Generate BC5CDR gold standard for disease AND drug evaluation.

Parses BioCreative V CDR corpus PubTator files containing both Chemical (drug)
and Disease annotations from 1,500 PubMed articles. Tests generalization of both
drug and disease extraction on scientific literature.

Source: https://github.com/JHnlp/BioCreative-V-CDR-Corpus
Format: PubTator (same as NCBI Disease)
Size: 1,500 articles, ~4,409 chemicals, ~5,818 diseases
License: Free for research

Usage:
    python generate_bc5cdr_gold.py
    python generate_bc5cdr_gold.py --split=test
    python generate_bc5cdr_gold.py --no-pdf --max-docs=50
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
GOLD_OUTPUT = SCRIPT_DIR.parent / "bc5cdr_gold.json"

# GitHub raw download URL for CDR_Data.zip
# The corpus is distributed as a zip containing PubTator files
CDR_ZIP_URL = "https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_Data.zip"

# Expected PubTator filenames inside the zip (after extraction)
PUBTATOR_FILES = {
    "train": "CDR_TrainingSet.PubTator.txt",
    "dev": "CDR_DevelopmentSet.PubTator.txt",
    "test": "CDR_TestSet.PubTator.txt",
}


def download_corpus() -> None:
    """Download and extract BC5CDR corpus from GitHub."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    any_exists = any((CORPUS_DIR / f).exists() for f in PUBTATOR_FILES.values())
    if any_exists:
        print("  Corpus files already exist")
        return

    zip_path = CORPUS_DIR / "CDR_Data.zip"
    print(f"  Downloading from {CDR_ZIP_URL}...")
    req = urllib.request.Request(CDR_ZIP_URL, headers={"User-Agent": "ESE-Pipeline/0.8"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    zip_path.write_bytes(data)
    print(f"    Downloaded {len(data)} bytes")

    # Extract - the zip may have a subdirectory
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CORPUS_DIR)

    # Move files from subdirectory if needed
    for split, filename in PUBTATOR_FILES.items():
        target = CORPUS_DIR / filename
        if not target.exists():
            # Look in subdirectories
            for candidate in CORPUS_DIR.rglob(filename):
                candidate.rename(target)
                print(f"    Moved {candidate} -> {target}")
                break

    zip_path.unlink()
    print("    Extracted successfully")


def parse_pubtator(file_path: Path, split: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse a BC5CDR PubTator format file.

    Returns (documents, disease_annotations, drug_annotations) where:
    - documents: [{"pmid": str, "title": str, "abstract": str, "split": str}]
    - disease_annotations: [{"doc_id": str, "text": str, "type": str, "mesh_id": str, "split": str}]
    - drug_annotations: [{"doc_id": str, "name": str, "mesh_id": str, "split": str}]
    """
    documents: list[dict] = []
    disease_annotations: list[dict] = []
    drug_annotations: list[dict] = []

    text = file_path.read_text(encoding="utf-8")
    blocks = text.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        title = ""
        abstract = ""
        pmid = ""
        seen_diseases: set[str] = set()
        seen_drugs: set[str] = set()

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
                # Annotation line: PMID\tstart\tend\tmention\ttype\tconcept_id
                fields = line.split("\t")
                if len(fields) >= 5:
                    mention = fields[3].strip()
                    entity_type = fields[4].strip()
                    concept_id = fields[5].strip() if len(fields) > 5 else ""

                    doc_id = f"{pmid}.pdf"

                    if entity_type == "Disease":
                        mention_key = mention.lower().strip()
                        if mention_key not in seen_diseases:
                            seen_diseases.add(mention_key)
                            disease_annotations.append({
                                "doc_id": doc_id,
                                "text": mention,
                                "type": "DISEASE",
                                "mesh_id": concept_id,
                                "split": split,
                            })
                    elif entity_type == "Chemical":
                        mention_key = mention.lower().strip()
                        if mention_key not in seen_drugs:
                            seen_drugs.add(mention_key)
                            drug_annotations.append({
                                "doc_id": doc_id,
                                "name": mention,
                                "mesh_id": concept_id,
                                "split": split,
                            })

        if pmid and (title or abstract):
            documents.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "split": split,
            })

    return documents, disease_annotations, drug_annotations


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
    parser = argparse.ArgumentParser(description="Generate BC5CDR gold standard")
    parser.add_argument("--split", choices=["test", "dev", "train", "all"], default="all",
                        help="Which split to process (default: all)")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    parser.add_argument("--no-download", action="store_true", help="Skip corpus download")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process")
    args = parser.parse_args()

    splits = [args.split] if args.split != "all" else ["train", "dev", "test"]

    # Download corpus
    if not args.no_download:
        print("Downloading BC5CDR Corpus...")
        download_corpus()
    else:
        print("Skipping download (using existing files)")

    # Parse PubTator files
    all_documents: list[dict] = []
    all_disease_annotations: list[dict] = []
    all_drug_annotations: list[dict] = []

    for split in splits:
        txt_path = CORPUS_DIR / PUBTATOR_FILES[split]
        if not txt_path.exists():
            print(f"  WARNING: {txt_path} not found, skipping {split} split")
            continue

        print(f"Parsing {split} split: {txt_path}...")
        docs, diseases, drugs = parse_pubtator(txt_path, split)
        all_documents.extend(docs)
        all_disease_annotations.extend(diseases)
        all_drug_annotations.extend(drugs)
        print(f"  {len(docs)} documents, {len(diseases)} disease annotations, {len(drugs)} drug annotations")

    # Apply max-docs limit
    if args.max_docs:
        all_documents = all_documents[:args.max_docs]
        valid_doc_ids = {f"{d['pmid']}.pdf" for d in all_documents}
        all_disease_annotations = [a for a in all_disease_annotations if a["doc_id"] in valid_doc_ids]
        all_drug_annotations = [a for a in all_drug_annotations if a["doc_id"] in valid_doc_ids]
        print(f"Limited to {len(all_documents)} docs, {len(all_disease_annotations)} diseases, {len(all_drug_annotations)} drugs")

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
        "corpus": "BC5CDR",
        "description": "BioCreative V CDR - PubMed articles with chemical (drug) and disease annotations",
        "source": "https://github.com/JHnlp/BioCreative-V-CDR-Corpus",
        "license": "Free for research",
        "diseases": {
            "total": len(all_disease_annotations),
            "annotations": all_disease_annotations,
        },
        "drugs": {
            "total": len(all_drug_annotations),
            "annotations": all_drug_annotations,
        },
    }

    with open(GOLD_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    # Summary
    unique_diseases = {a["text"].lower() for a in all_disease_annotations}
    unique_drugs = {a["name"].lower() for a in all_drug_annotations}

    disease_by_split: dict[str, list] = {}
    drug_by_split: dict[str, list] = {}
    for a in all_disease_annotations:
        disease_by_split.setdefault(a["split"], []).append(a)
    for a in all_drug_annotations:
        drug_by_split.setdefault(a["split"], []).append(a)

    print(f"\n{'='*60}")
    print("BC5CDR Gold Standard Generated")
    print(f"{'='*60}")
    print(f"  Total documents:       {len(all_documents)}")
    print(f"  Disease annotations:   {len(all_disease_annotations)} ({len(unique_diseases)} unique)")
    print(f"  Drug annotations:      {len(all_drug_annotations)} ({len(unique_drugs)} unique)")
    for split in sorted(set(list(disease_by_split.keys()) + list(drug_by_split.keys()))):
        d_anns = disease_by_split.get(split, [])
        dr_anns = drug_by_split.get(split, [])
        d_docs = {a["doc_id"] for a in d_anns}
        dr_docs = {a["doc_id"] for a in dr_anns}
        all_docs = d_docs | dr_docs
        print(f"  {split:5s}: {len(d_anns)} diseases, {len(dr_anns)} drugs ({len(all_docs)} docs)")
    print(f"  PDFs generated:        {pdfs_generated}")
    print(f"  Gold output:           {GOLD_OUTPUT}")


if __name__ == "__main__":
    main()
