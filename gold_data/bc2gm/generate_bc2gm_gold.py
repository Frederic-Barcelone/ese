#!/usr/bin/env python3
"""
Generate golden_bc2gm.json and PDFs from the BioCreative II Gene Mention corpus.

BioCreative II GM corpus:
- 5,000 test sentences from PubMed abstracts
- Annotated gene/protein mentions with character offsets
- Simple pipe-delimited format (no external dependencies)
- Reference: https://github.com/spyysalo/bc2gm-corpus

Input files (in corpus/):
    test.in    -- Sentences: "SENTENCE_ID TEXT"
    GENE.eval  -- Annotations: "SENTENCE_ID|START END|GENE_TEXT"

Output:
    golden_bc2gm.json  -- Gold standard gene annotations for F03 evaluation
    pdfs/              -- Single-page PDFs (one per sentence)

Usage:
    python generate_bc2gm_gold.py [--max N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fpdf import FPDF


BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus"
PDFS_DIR = BASE_DIR / "pdfs"
SENTENCES_FILE = CORPUS_DIR / "test.in"
ANNOTATIONS_FILE = CORPUS_DIR / "GENE.eval"
OUTPUT_FILE = BASE_DIR.parent / "golden_bc2gm.json"


def load_sentences(path: Path) -> dict[str, str]:
    """Load sentences from test.in.

    Format: "SENTENCE_ID TEXT" (space-separated, first token is ID).
    """
    sentences: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # First token is the sentence ID, rest is text
        parts = line.split(" ", 1)
        if len(parts) == 2:
            sentences[parts[0]] = parts[1]
    return sentences


def load_annotations(path: Path) -> dict[str, list[str]]:
    """Load gene annotations from GENE.eval.

    Format: "SENTENCE_ID|START END|GENE_TEXT"
    Returns dict mapping sentence_id -> list of unique gene texts.
    """
    annotations: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}  # per-sentence dedup

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 3:
            continue

        sent_id = parts[0].strip()
        gene_text = parts[2].strip()

        if not gene_text:
            continue

        # Deduplicate by lowercased text within same sentence
        dedup_key = gene_text.lower()
        if sent_id not in seen:
            seen[sent_id] = set()
        if dedup_key in seen[sent_id]:
            continue
        seen[sent_id].add(dedup_key)

        annotations.setdefault(sent_id, []).append(gene_text)

    return annotations


def _sanitize_for_latin1(text: str) -> str:
    """Replace non-latin1 characters with ASCII equivalents for PDF generation."""
    greek_map = {
        "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03b4": "delta",
        "\u03b5": "epsilon", "\u03b6": "zeta", "\u03b7": "eta", "\u03b8": "theta",
        "\u03b9": "iota", "\u03ba": "kappa", "\u03bb": "lambda", "\u03bc": "mu",
        "\u03bd": "nu", "\u03be": "xi", "\u03c0": "pi", "\u03c1": "rho",
        "\u03c3": "sigma", "\u03c4": "tau", "\u03c5": "upsilon", "\u03c6": "phi",
        "\u03c7": "chi", "\u03c8": "psi", "\u03c9": "omega",
        "\u0394": "Delta", "\u03a3": "Sigma", "\u03a9": "Omega",
    }
    for greek, ascii_eq in greek_map.items():
        text = text.replace(greek, ascii_eq)

    result = []
    for ch in text:
        try:
            ch.encode("latin-1")
            result.append(ch)
        except UnicodeEncodeError:
            result.append("?")
    return "".join(result)


def generate_pdf(sent_id: str, text: str, output_dir: Path) -> Path:
    """Generate a single-page PDF from a sentence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{sent_id}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _sanitize_for_latin1(text))

    pdf.output(str(pdf_path))
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Generate BioCreative II GM gold standard")
    parser.add_argument("--max", type=int, default=None,
                        help="Max sentences to process")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N annotated sentences")
    args = parser.parse_args()

    print("=" * 60)
    print("BioCreative II GM Gold Standard Generator")
    print("=" * 60)

    # Load data
    if not SENTENCES_FILE.exists():
        print(f"ERROR: Sentences file not found: {SENTENCES_FILE}")
        print("Download bc2gm corpus first: https://github.com/spyysalo/bc2gm-corpus")
        return

    sentences = load_sentences(SENTENCES_FILE)
    annotations = load_annotations(ANNOTATIONS_FILE)

    print(f"Sentences loaded:    {len(sentences)}")
    print(f"Annotated sentences: {len(annotations)}")
    total_genes = sum(len(genes) for genes in annotations.values())
    print(f"Total gene mentions: {total_genes} (deduplicated per sentence)")

    # Only process sentences that have gene annotations
    annotated_ids = sorted(annotations.keys())
    if args.skip:
        annotated_ids = annotated_ids[args.skip:]
    if args.max:
        annotated_ids = annotated_ids[:args.max]

    print(f"Sentences to process: {len(annotated_ids)}")
    print("-" * 60)

    # Process each annotated sentence
    all_annotations: list[dict] = []
    pdfs_created = 0

    for i, sent_id in enumerate(annotated_ids, 1):
        text = sentences.get(sent_id, "")
        if not text:
            continue

        doc_id = f"{sent_id}.pdf"
        genes = annotations[sent_id]

        # Add gene annotations
        for gene_text in genes:
            all_annotations.append({
                "doc_id": doc_id,
                "symbol": gene_text,
            })

        # Generate PDF
        generate_pdf(sent_id, text, PDFS_DIR)
        pdfs_created += 1

        if i % 500 == 0 or i == len(annotated_ids):
            print(f"  Processed {i}/{len(annotated_ids)} sentences")

    # Build output JSON
    output = {
        "corpus": "BioCreative-II-GM",
        "genes": {
            "total": len(all_annotations),
            "annotations": all_annotations,
        },
    }

    # Write golden_bc2gm.json
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Gene annotations: {output['genes']['total']}")
    print(f"PDFs created: {pdfs_created}")
    print(f"PDFs directory: {PDFS_DIR}")
    print("=" * 60)

    # Show samples
    if all_annotations:
        print("\nSample gene annotations:")
        for ann in all_annotations[:10]:
            print(f"  [{ann['doc_id']}] {ann['symbol']}")


if __name__ == "__main__":
    main()
