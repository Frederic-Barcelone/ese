#!/usr/bin/env python3
"""
Generate PubMed Author/Citation gold standard for evaluation.

Downloads real open-access PDFs from PubMed Central (PMC) and fetches structured
author/citation metadata from PubMed XML. Only PMIDs with a PMCID (open-access
full-text) are included, since the pipeline needs real PDFs with author information.

Source PMIDs come from existing gene corpora (NLM-Gene + RareDisGene).

Source: NCBI E-utilities (PubMed XML) + PMC Open Access PDFs
License: Public (NIH)

Usage:
    python generate_pubmed_author_gold.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import xml.etree.ElementTree as ET

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
GOLD_OUTPUT = SCRIPT_DIR.parent / "pubmed_author_gold.json"
METADATA_CACHE = SCRIPT_DIR / "metadata_cache.json"
PDF_DIR = SCRIPT_DIR / "PDFs"

# Gene corpus PDF directories (source of PMIDs)
NLM_GENE_PDF_DIR = SCRIPT_DIR.parent / "nlm_gene" / "pdfs"
RAREDIS_GENE_PDF_DIR = SCRIPT_DIR.parent / "raredis_gene" / "pdfs"

# ── NCBI E-utilities ──
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
METADATA_BATCH_SIZE = 200
RATE_LIMIT_DELAY = 0.4  # seconds between API batches

# PMC PDF download (via Europe PMC — returns actual PDF bytes)
PMC_PDF_URL = "https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
PDF_DOWNLOAD_TIMEOUT = 30  # seconds per PDF
PDF_DOWNLOAD_DELAY = 0.35  # seconds between downloads

# ── Parameters (edit here) ──
MAX_DOCS: int | None = None        # None = all PMIDs; set to e.g. 50 to limit
NO_FETCH: bool = False             # True = skip PubMed metadata fetch, use cache only
NO_DOWNLOAD: bool = False          # True = skip PDF downloads, use existing PDFs only
SPLITS: list[str] = ["test", "train"]


def collect_pmids() -> list[str]:
    """Collect PMIDs from existing gene corpus PDF directories."""
    pmids: set[str] = set()

    for pdf_dir in [NLM_GENE_PDF_DIR, RAREDIS_GENE_PDF_DIR]:
        for split in SPLITS:
            split_dir = pdf_dir / split
            if not split_dir.exists():
                continue
            for pdf in split_dir.glob("*.pdf"):
                pmid = pdf.stem
                if pmid.isdigit():
                    pmids.add(pmid)

    return sorted(pmids)


def determine_splits() -> dict[str, str]:
    """Determine which split each PMID belongs to."""
    pmid_splits: dict[str, str] = {}
    for pdf_dir in [NLM_GENE_PDF_DIR, RAREDIS_GENE_PDF_DIR]:
        for split in SPLITS:
            split_dir = pdf_dir / split
            if not split_dir.exists():
                continue
            for pdf in split_dir.glob("*.pdf"):
                pmid = pdf.stem
                if pmid.isdigit():
                    pmid_splits[pmid] = split
    return pmid_splits


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

    for batch_start in range(0, len(to_fetch), METADATA_BATCH_SIZE):
        batch = to_fetch[batch_start:batch_start + METADATA_BATCH_SIZE]
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

        if batch_start + METADATA_BATCH_SIZE < len(to_fetch):
            time.sleep(RATE_LIMIT_DELAY)

        fetched = min(batch_start + METADATA_BATCH_SIZE, len(to_fetch))
        if fetched % 500 == 0 or fetched == len(to_fetch):
            print(f"    Fetched {fetched}/{len(to_fetch)}...")

    return cache


def download_pmc_pdfs(
    pmids: list[str],
    cache: dict[str, dict],
    pmid_splits: dict[str, str],
) -> dict[str, Path]:
    """Download open-access PDFs from PMC for PMIDs that have a PMCID.

    Returns dict: pmid -> local PDF path (only for successfully downloaded PDFs).
    """
    downloaded: dict[str, Path] = {}
    skipped_no_pmcid = 0
    skipped_existing = 0
    failed = 0

    # Collect PMIDs that have a PMCID
    candidates = []
    for pmid in pmids:
        meta = cache.get(pmid)
        if not meta or not meta.get("pmcid"):
            skipped_no_pmcid += 1
            continue
        candidates.append((pmid, meta["pmcid"]))

    print(f"  {len(candidates)} PMIDs have PMCID ({skipped_no_pmcid} without)")

    for i, (pmid, pmcid) in enumerate(candidates):
        split = pmid_splits.get(pmid, "test")
        pdf_path = PDF_DIR / split / f"{pmid}.pdf"

        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            downloaded[pmid] = pdf_path
            skipped_existing += 1
            continue

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        url = PMC_PDF_URL.format(pmcid=pmcid)

        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "ESE-Pipeline/0.8 (academic research)",
                "Accept": "application/pdf",
            })
            with urllib.request.urlopen(req, timeout=PDF_DOWNLOAD_TIMEOUT) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()

                # Verify we got a PDF (not an HTML error page)
                if data[:5] == b"%PDF-" or "pdf" in content_type.lower():
                    pdf_path.write_bytes(data)
                    downloaded[pmid] = pdf_path
                else:
                    failed += 1

        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            failed += 1
            if failed <= 5:
                print(f"    WARNING: {pmcid} ({pmid}): {e}")
            elif failed == 6:
                print("    (suppressing further download warnings)")

        time.sleep(PDF_DOWNLOAD_DELAY)

        done = i + 1
        if done % 100 == 0 or done == len(candidates):
            new_downloads = done - skipped_existing - failed
            print(f"    Progress: {done}/{len(candidates)} "
                  f"(existing: {skipped_existing}, new: {new_downloads}, failed: {failed})")

    print(f"  Download complete: {len(downloaded)} PDFs available "
          f"({skipped_existing} existing, {len(downloaded) - skipped_existing} new, {failed} failed)")

    return downloaded


def main() -> None:
    # 1. Collect PMIDs from gene corpus
    print("Collecting PMIDs from gene corpus PDFs...")
    pmids = collect_pmids()
    print(f"  Found {len(pmids)} unique PMIDs")

    if not pmids:
        print("ERROR: No PMIDs found. Ensure gene corpus PDFs exist at:")
        print(f"  {NLM_GENE_PDF_DIR}")
        print(f"  {RAREDIS_GENE_PDF_DIR}")
        sys.exit(1)

    if MAX_DOCS:
        pmids = pmids[:MAX_DOCS]
        print(f"  Limited to {MAX_DOCS} PMIDs")

    pmid_splits = determine_splits()

    # 2. Fetch PubMed metadata
    cache: dict[str, dict] = {}
    if METADATA_CACHE.exists():
        with open(METADATA_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached metadata entries")

    if not NO_FETCH:
        cache = fetch_pubmed_metadata(pmids, cache)
        with open(METADATA_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"  Cache saved ({len(cache)} entries)")

    # 3. Download real PDFs from PMC
    if not NO_DOWNLOAD:
        print("Downloading PDFs from PubMed Central...")
        downloaded = download_pmc_pdfs(pmids, cache, pmid_splits)
    else:
        print("Skipping PDF downloads (NO_DOWNLOAD=True), using existing PDFs...")
        downloaded = {}
        for pmid in pmids:
            split = pmid_splits.get(pmid, "test")
            pdf_path = PDF_DIR / split / f"{pmid}.pdf"
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                downloaded[pmid] = pdf_path

    # 4. Build gold (only for PMIDs with downloaded PDFs)
    author_annotations: list[dict] = []
    citation_annotations: list[dict] = []

    for pmid in pmids:
        if pmid not in downloaded or pmid not in cache:
            continue

        meta = cache[pmid]
        doc_id = f"{pmid}.pdf"
        split = pmid_splits.get(pmid, "unknown")

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

        citation_annotations.append({
            "doc_id": doc_id,
            "pmid": pmid,
            "doi": meta.get("doi"),
            "pmcid": meta.get("pmcid"),
            "split": split,
        })

    # 5. Write gold JSON
    gold = {
        "corpus": "PubMed-Authors",
        "description": "PubMed structured metadata for author/citation evaluation (real PMC PDFs)",
        "source": "NCBI E-utilities (PubMed XML) + PMC Open Access PDFs",
        "license": "Public (NIH)",
        "pdf_dir": str(PDF_DIR),
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

    # 6. Summary
    unique_authors = {(a["last_name"].lower(), a.get("initials", "").lower()) for a in author_annotations}
    docs_with_authors = {a["doc_id"] for a in author_annotations}
    docs_with_doi = {c["doc_id"] for c in citation_annotations if c.get("doi")}

    author_by_split: dict[str, list] = {}
    for a in author_annotations:
        author_by_split.setdefault(a["split"], []).append(a)

    print(f"\n{'='*60}")
    print("PubMed Author/Citation Gold Standard Generated")
    print(f"{'='*60}")
    print(f"  PDFs downloaded:           {len(downloaded)}")
    print(f"  Total author annotations:  {len(author_annotations)}")
    print(f"  Unique authors:            {len(unique_authors)}")
    print(f"  Docs with authors:         {len(docs_with_authors)}")
    print(f"  Citation annotations:      {len(citation_annotations)}")
    print(f"  Docs with DOI:             {len(docs_with_doi)}")
    for split in sorted(author_by_split):
        split_auths = author_by_split[split]
        split_docs = {a["doc_id"] for a in split_auths}
        print(f"  {split:5s} authors:          {len(split_auths)} ({len(split_docs)} docs)")
    print(f"  PDF dir:                   {PDF_DIR}")
    print(f"  Gold output:               {GOLD_OUTPUT}")


if __name__ == "__main__":
    main()
