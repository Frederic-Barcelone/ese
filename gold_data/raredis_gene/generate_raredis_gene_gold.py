#!/usr/bin/env python3
"""
Generate RareDisGene gold standard for gene evaluation.

Downloads the "Gene-RD-Provenance V2" dataset from Figshare (CC0 license),
fetches PubMed abstracts for gene-disease association PMIDs, generates PDFs,
and outputs a gold JSON file.

Source: https://figshare.com/articles/dataset/Gene-RD-Provenance_V2/7718537
Contains 3,163 genes with HGNC symbols linked to 4,166 rare diseases via PubMed IDs.

The gold standard says: for PMID X, the pipeline should find gene Y (HGNC symbol).
Only HGNC symbols present in the pipeline lexicon (~4,100 Orphadata genes) are included.

Usage:
    python generate_raredis_gene_gold.py
    python generate_raredis_gene_gold.py --max-docs=50
    python generate_raredis_gene_gold.py --skip-download --no-fetch
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
TSV_PATH = SCRIPT_DIR / "gene-RD-provenance_v2.txt"
ABSTRACT_CACHE = SCRIPT_DIR / "abstracts_cache.json"
PDF_DIR = SCRIPT_DIR / "pdfs"
GOLD_OUTPUT = SCRIPT_DIR.parent / "raredis_gene_gold.json"
ORPHADATA_GENES = SCRIPT_DIR.parent.parent / "ouput_datasources" / "2025_08_orphadata_genes.json"

# Figshare download URL for the TSV
FIGSHARE_URL = "https://ndownloader.figshare.com/files/14367506"

# NCBI E-utilities
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200
RATE_LIMIT_DELAY = 0.4  # seconds between batches


def load_pipeline_symbols() -> set[str]:
    """Load HGNC symbols from the pipeline's Orphadata gene file."""
    with open(ORPHADATA_GENES, "r", encoding="utf-8") as f:
        genes = json.load(f)

    symbols: set[str] = set()
    for g in genes:
        symbol = g.get("hgnc_symbol")
        if symbol:
            symbols.add(symbol)
    return symbols


def download_tsv() -> None:
    """Download the Gene-RD-Provenance V2 file from Figshare."""
    if TSV_PATH.exists():
        print(f"  File already exists: {TSV_PATH}")
        return

    print(f"  Downloading from Figshare...")
    req = urllib.request.Request(FIGSHARE_URL, headers={"User-Agent": "ESE-Pipeline/0.8"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    TSV_PATH.write_bytes(data)
    print(f"  Downloaded to {TSV_PATH} ({len(data)} bytes)")


def parse_tsv(pipeline_symbols: set[str]) -> dict[str, set[str]]:
    """Parse TSV and build PMID → set of HGNC symbols mapping.

    Returns only symbols present in the pipeline lexicon.
    """
    pmid_to_genes: dict[str, set[str]] = {}
    total_rows = 0
    matched_rows = 0
    skipped_no_pmid = 0
    skipped_no_symbol = 0

    with open(TSV_PATH, "r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

        # Find column indices
        col_map = {h.strip().lower(): i for i, h in enumerate(header)}
        # Try common column names
        gene_col = None
        pmid_col = None

        for name in ["hgnc", "gene symbol", "gene_symbol", "genesymbol"]:
            if name in col_map:
                gene_col = col_map[name]
                break

        for name in ["pmid gene-disease", "pmid gene-disease association", "pmid_gene_disease"]:
            if name in col_map:
                pmid_col = col_map[name]
                break

        if gene_col is None or pmid_col is None:
            # Fallback: try to find by inspection
            print(f"  Header columns: {header}")
            for i, h in enumerate(header):
                h_lower = h.strip().lower()
                if gene_col is None and ("hgnc" in h_lower or ("gene" in h_lower and "symbol" in h_lower)):
                    gene_col = i
                if pmid_col is None and "pmid" in h_lower and ("gene" in h_lower or "disease" in h_lower):
                    pmid_col = i

        if gene_col is None or pmid_col is None:
            print(f"  ERROR: Could not find gene symbol or PMID columns in header: {header}")
            sys.exit(1)

        print(f"  Gene symbol column: {header[gene_col]} (index {gene_col})")
        print(f"  PMID column: {header[pmid_col]} (index {pmid_col})")

        for row in reader:
            total_rows += 1
            if len(row) <= max(gene_col, pmid_col):
                continue

            symbol = row[gene_col].strip()
            pmid_raw = row[pmid_col].strip()

            if not pmid_raw or pmid_raw == "-" or pmid_raw.lower() == "na":
                skipped_no_pmid += 1
                continue

            if symbol not in pipeline_symbols:
                skipped_no_symbol += 1
                continue

            # Handle multiple PMIDs separated by semicolons or commas
            for pmid in pmid_raw.replace(";", ",").split(","):
                pmid = pmid.strip()
                if pmid and pmid.isdigit():
                    pmid_to_genes.setdefault(pmid, set()).add(symbol)
                    matched_rows += 1

    print(f"  Total rows: {total_rows}")
    print(f"  Matched rows (in pipeline lexicon): {matched_rows}")
    print(f"  Skipped (no PMID): {skipped_no_pmid}")
    print(f"  Skipped (symbol not in lexicon): {skipped_no_symbol}")
    print(f"  Unique PMIDs: {len(pmid_to_genes)}")

    return pmid_to_genes


def fetch_abstracts(pmids: list[str], cache: dict[str, dict]) -> dict[str, dict]:
    """Fetch PubMed abstracts via NCBI E-utilities in batches.

    Returns dict: pmid → {"title": str, "abstract": str}
    """
    to_fetch = [p for p in pmids if p not in cache]
    if not to_fetch:
        print(f"  All {len(pmids)} abstracts cached")
        return cache

    print(f"  Fetching {len(to_fetch)} abstracts from PubMed ({len(pmids) - len(to_fetch)} cached)...")

    for batch_start in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[batch_start:batch_start + BATCH_SIZE]
        ids_param = ",".join(batch)
        url = f"{EFETCH_URL}?db=pubmed&id={ids_param}&rettype=abstract&retmode=xml"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ESE-Pipeline/0.8"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            for article in root.iter("PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                if pmid_elem is None or pmid_elem.text is None:
                    continue
                pmid = pmid_elem.text.strip()

                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None and title_elem.text else ""

                # Collect abstract text from all AbstractText elements
                abstract_parts = []
                for at in article.findall(".//AbstractText"):
                    label = at.get("Label", "")
                    text = "".join(at.itertext())
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                cache[pmid] = {"title": title, "abstract": abstract}

        except Exception as e:
            print(f"    WARNING: Batch fetch failed: {e}")

        if batch_start + BATCH_SIZE < len(to_fetch):
            time.sleep(RATE_LIMIT_DELAY)

        fetched = min(batch_start + BATCH_SIZE, len(to_fetch))
        if fetched % 1000 == 0 or fetched == len(to_fetch):
            print(f"    Fetched {fetched}/{len(to_fetch)}...")

    return cache


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
    parser = argparse.ArgumentParser(description="Generate RareDisGene gold standard")
    parser.add_argument("--max-docs", type=int, default=None, help="Max PMIDs to process")
    parser.add_argument("--skip-download", action="store_true", help="Skip TSV download")
    parser.add_argument("--no-fetch", action="store_true", help="Use cached abstracts only")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    # Load pipeline lexicon
    print("Loading pipeline HGNC symbols from Orphadata...")
    pipeline_symbols = load_pipeline_symbols()
    print(f"  {len(pipeline_symbols)} HGNC symbols in pipeline lexicon")

    # Download TSV
    if not args.skip_download:
        print("Downloading Gene-RD-Provenance V2 TSV...")
        download_tsv()
    else:
        print("Skipping TSV download (using existing file)")

    if not TSV_PATH.exists():
        print(f"ERROR: TSV not found at {TSV_PATH}")
        sys.exit(1)

    # Parse TSV
    print("Parsing TSV...")
    pmid_to_genes = parse_tsv(pipeline_symbols)

    # Split PMIDs 80/20 (train/test)
    all_pmids = sorted(pmid_to_genes.keys())
    random.seed(args.seed)
    random.shuffle(all_pmids)

    split_idx = int(len(all_pmids) * 0.8)
    train_pmids = set(all_pmids[:split_idx])
    test_pmids = set(all_pmids[split_idx:])

    print(f"  Split: {len(train_pmids)} train, {len(test_pmids)} test")

    # Apply max-docs limit
    pmids_to_process = all_pmids
    if args.max_docs:
        pmids_to_process = all_pmids[:args.max_docs]
        print(f"  Limited to {args.max_docs} PMIDs")

    # Load/fetch abstracts
    cache: dict[str, dict] = {}
    if ABSTRACT_CACHE.exists():
        with open(ABSTRACT_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached abstracts")

    if not args.no_fetch:
        cache = fetch_abstracts(pmids_to_process, cache)
        # Save cache
        with open(ABSTRACT_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"  Cache saved ({len(cache)} abstracts)")

    # Generate gold + PDFs
    all_annotations: list[dict] = []
    pdfs_generated = 0
    skipped_no_abstract = 0

    for pmid in pmids_to_process:
        if pmid not in cache or not cache[pmid].get("title"):
            skipped_no_abstract += 1
            continue

        title = cache[pmid]["title"]
        abstract = cache[pmid].get("abstract", "")
        split = "train" if pmid in train_pmids else "test"
        doc_id = f"{pmid}.pdf"

        for symbol in sorted(pmid_to_genes[pmid]):
            all_annotations.append({
                "doc_id": doc_id,
                "symbol": symbol,
                "split": split,
            })

        # Generate PDF
        if not args.no_pdf:
            pdf_out = PDF_DIR / split / doc_id
            pdf_out.parent.mkdir(parents=True, exist_ok=True)
            if not pdf_out.exists():
                generate_pdf(pmid, title, abstract, pdf_out)
                pdfs_generated += 1

    # Write gold JSON
    gold = {
        "corpus": "RareDisGene",
        "description": "Gene-RD-Provenance V2 (Figshare) - HGNC symbols from rare disease gene-disease associations",
        "source": "https://figshare.com/articles/dataset/Gene-RD-Provenance_V2/7718537",
        "license": "CC0",
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
    print(f"RareDisGene Gold Standard Generated")
    print(f"{'='*60}")
    print(f"  Total annotations:    {len(all_annotations)}")
    print(f"  Unique HGNC symbols:  {len(unique_symbols)}")
    print(f"  Unique PMIDs:         {len({a['doc_id'] for a in all_annotations})}")
    print(f"  Test annotations:     {len(test_anns)} ({len({a['doc_id'] for a in test_anns})} docs)")
    print(f"  Train annotations:    {len(train_anns)} ({len({a['doc_id'] for a in train_anns})} docs)")
    print(f"  Skipped (no abstract):{skipped_no_abstract}")
    print(f"  PDFs generated:       {pdfs_generated}")
    print(f"  Gold output:          {GOLD_OUTPUT}")


if __name__ == "__main__":
    main()
