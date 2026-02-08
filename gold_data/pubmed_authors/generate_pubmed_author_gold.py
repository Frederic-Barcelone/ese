#!/usr/bin/env python3
"""
Generate PubMed Author/Citation gold standard for evaluation.

Uses PMIDs from existing gene corpora (NLM-Gene + RareDisGene) which already have
generated PDFs. Fetches PubMed XML metadata to get structured author lists and
citation identifiers (PMID, DOI). Compares against pipeline's author/citation extraction.

No PDF generation needed -- reuses existing PDFs from gene corpora.

Source: NCBI E-utilities (PubMed XML)
License: Public (NIH)

Usage:
    python generate_pubmed_author_gold.py
    python generate_pubmed_author_gold.py --max-docs=50
    python generate_pubmed_author_gold.py --no-fetch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
GOLD_OUTPUT = SCRIPT_DIR.parent / "pubmed_author_gold.json"
METADATA_CACHE = SCRIPT_DIR / "metadata_cache.json"

# Gene corpus PDF directories (reuse existing PDFs)
NLM_GENE_PDF_DIR = SCRIPT_DIR.parent / "nlm_gene" / "pdfs"
RAREDIS_GENE_PDF_DIR = SCRIPT_DIR.parent / "raredis_gene" / "pdfs"

# NCBI E-utilities
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200
RATE_LIMIT_DELAY = 0.4  # seconds between batches


def collect_pmids(splits: list[str]) -> list[str]:
    """Collect PMIDs from existing gene corpus PDF directories."""
    pmids: set[str] = set()

    for pdf_dir in [NLM_GENE_PDF_DIR, RAREDIS_GENE_PDF_DIR]:
        for split in splits:
            split_dir = pdf_dir / split
            if not split_dir.exists():
                continue
            for pdf in split_dir.glob("*.pdf"):
                pmid = pdf.stem  # e.g., "12345678"
                if pmid.isdigit():
                    pmids.add(pmid)

    return sorted(pmids)


def fetch_pubmed_metadata(pmids: list[str], cache: dict[str, dict]) -> dict[str, dict]:
    """Fetch PubMed XML metadata via NCBI E-utilities in batches.

    Returns dict: pmid -> {
        "authors": [{"last_name": str, "first_name": str, "initials": str, "affiliation": str}],
        "doi": str | None,
        "pmcid": str | None,
    }
    """
    to_fetch = [p for p in pmids if p not in cache]
    if not to_fetch:
        print(f"  All {len(pmids)} PMIDs cached")
        return cache

    print(f"  Fetching metadata for {len(to_fetch)} PMIDs ({len(pmids) - len(to_fetch)} cached)...")

    for batch_start in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[batch_start:batch_start + BATCH_SIZE]
        ids_param = ",".join(batch)
        url = f"{EFETCH_URL}?db=pubmed&id={ids_param}&rettype=xml&retmode=xml"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ESE-Pipeline/0.8"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)

            for article in root.iter("PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                if pmid_elem is None or pmid_elem.text is None:
                    continue
                pmid = pmid_elem.text.strip()

                # Extract authors
                authors: list[dict] = []
                for author_elem in article.findall(".//Author"):
                    last_name_elem = author_elem.find("LastName")
                    first_name_elem = author_elem.find("ForeName")
                    initials_elem = author_elem.find("Initials")
                    affiliation_elem = author_elem.find(".//Affiliation")

                    # Skip collective names (consortiums etc.)
                    if last_name_elem is None or last_name_elem.text is None:
                        continue

                    author = {
                        "last_name": last_name_elem.text.strip(),
                        "first_name": first_name_elem.text.strip() if first_name_elem is not None and first_name_elem.text else "",
                        "initials": initials_elem.text.strip() if initials_elem is not None and initials_elem.text else "",
                        "affiliation": affiliation_elem.text.strip() if affiliation_elem is not None and affiliation_elem.text else "",
                    }
                    authors.append(author)

                # Extract identifiers (DOI, PMCID)
                doi = None
                pmcid = None
                for article_id in article.findall(".//ArticleId"):
                    id_type = article_id.get("IdType", "")
                    id_text = article_id.text.strip() if article_id.text else ""
                    if id_type == "doi" and id_text:
                        doi = id_text
                    elif id_type == "pmc" and id_text:
                        pmcid = id_text

                cache[pmid] = {
                    "authors": authors,
                    "doi": doi,
                    "pmcid": pmcid,
                }

        except Exception as e:
            print(f"    WARNING: Batch fetch failed: {e}")

        if batch_start + BATCH_SIZE < len(to_fetch):
            time.sleep(RATE_LIMIT_DELAY)

        fetched = min(batch_start + BATCH_SIZE, len(to_fetch))
        if fetched % 500 == 0 or fetched == len(to_fetch):
            print(f"    Fetched {fetched}/{len(to_fetch)}...")

    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PubMed Author/Citation gold standard")
    parser.add_argument("--max-docs", type=int, default=None, help="Max PMIDs to process")
    parser.add_argument("--no-fetch", action="store_true", help="Use cached metadata only")
    parser.add_argument("--splits", nargs="+", default=["test", "train"],
                        help="Which splits to collect PMIDs from (default: test train)")
    args = parser.parse_args()

    # Collect PMIDs from existing gene corpus PDFs
    print("Collecting PMIDs from gene corpus PDFs...")
    pmids = collect_pmids(args.splits)
    print(f"  Found {len(pmids)} unique PMIDs")

    if not pmids:
        print("ERROR: No PMIDs found. Ensure gene corpus PDFs exist at:")
        print(f"  {NLM_GENE_PDF_DIR}")
        print(f"  {RAREDIS_GENE_PDF_DIR}")
        sys.exit(1)

    # Apply max-docs limit
    if args.max_docs:
        pmids = pmids[:args.max_docs]
        print(f"  Limited to {args.max_docs} PMIDs")

    # Load/fetch metadata
    cache: dict[str, dict] = {}
    if METADATA_CACHE.exists():
        with open(METADATA_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached metadata entries")

    if not args.no_fetch:
        cache = fetch_pubmed_metadata(pmids, cache)
        # Save cache
        with open(METADATA_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"  Cache saved ({len(cache)} entries)")

    # Build gold annotations
    author_annotations: list[dict] = []
    citation_annotations: list[dict] = []

    # Determine which split each PMID belongs to
    pmid_splits: dict[str, str] = {}
    for pdf_dir in [NLM_GENE_PDF_DIR, RAREDIS_GENE_PDF_DIR]:
        for split in args.splits:
            split_dir = pdf_dir / split
            if not split_dir.exists():
                continue
            for pdf in split_dir.glob("*.pdf"):
                pmid = pdf.stem
                if pmid.isdigit():
                    pmid_splits[pmid] = split

    for pmid in pmids:
        if pmid not in cache:
            continue

        meta = cache[pmid]
        doc_id = f"{pmid}.pdf"
        split = pmid_splits.get(pmid, "unknown")

        # Author annotations
        for author in meta.get("authors", []):
            if not author.get("last_name"):
                continue
            author_annotations.append({
                "doc_id": doc_id,
                "last_name": author["last_name"],
                "first_name": author.get("first_name", ""),
                "initials": author.get("initials", ""),
                "affiliation": author.get("affiliation", ""),
                "split": split,
            })

        # Citation annotations (the paper's own PMID and DOI)
        citation_entry: dict[str, str | None] = {
            "doc_id": doc_id,
            "pmid": pmid,
            "doi": meta.get("doi"),
            "pmcid": meta.get("pmcid"),
            "split": split,
        }
        citation_annotations.append(citation_entry)

    # Write gold JSON
    gold = {
        "corpus": "PubMed-Authors",
        "description": "PubMed structured metadata for author/citation evaluation (from gene corpus PMIDs)",
        "source": "NCBI E-utilities (PubMed XML)",
        "license": "Public (NIH)",
        "authors": {
            "total": len(author_annotations),
            "annotations": author_annotations,
        },
        "citations": {
            "total": len(citation_annotations),
            "annotations": citation_annotations,
        },
    }

    with open(GOLD_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    # Summary
    unique_authors = {(a["last_name"].lower(), a.get("initials", "").lower()) for a in author_annotations}
    docs_with_authors = {a["doc_id"] for a in author_annotations}
    docs_with_doi = {c["doc_id"] for c in citation_annotations if c.get("doi")}

    author_by_split: dict[str, list] = {}
    for a in author_annotations:
        author_by_split.setdefault(a["split"], []).append(a)

    print(f"\n{'='*60}")
    print("PubMed Author/Citation Gold Standard Generated")
    print(f"{'='*60}")
    print(f"  Total author annotations:  {len(author_annotations)}")
    print(f"  Unique authors:            {len(unique_authors)}")
    print(f"  Docs with authors:         {len(docs_with_authors)}")
    print(f"  Citation annotations:      {len(citation_annotations)}")
    print(f"  Docs with DOI:             {len(docs_with_doi)}")
    for split in sorted(author_by_split):
        split_auths = author_by_split[split]
        split_docs = {a["doc_id"] for a in split_auths}
        print(f"  {split:5s} authors:          {len(split_auths)} ({len(split_docs)} docs)")
    print(f"  Gold output:               {GOLD_OUTPUT}")


if __name__ == "__main__":
    main()
