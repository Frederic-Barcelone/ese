# corpus_metadata/corpus_abbreviations/F_evaluation/F00_gold_audit.py
"""
Phase 0: Gold Audit

Determines the extraction ceiling by classifying gold annotations into buckets:
- EXTRACTABLE_PAIR: SF+LF with evidence (pattern, glossary, or proximity)
- SF_ONLY: SF found, LF not found or not extractable
- UNSUPPORTED_PAIR: SF+LF both exist but no evidence linking them
- PARSE_ISSUE: SF not found in parsed text

Usage:
    python F00_gold_audit.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


class AuditBucket(str, Enum):
    EXTRACTABLE_PAIR = "EXTRACTABLE_PAIR"  # SF+LF with evidence
    SF_ONLY = "SF_ONLY"                    # SF found, LF not extractable
    UNSUPPORTED_PAIR = "UNSUPPORTED_PAIR"  # SF+LF both exist but no link
    PARSE_ISSUE = "PARSE_ISSUE"            # SF not found in text


@dataclass
class AuditDetail:
    doc_id: str
    short_form: str
    long_form: Optional[str]
    bucket: AuditBucket
    sf_occurrences: int
    lf_occurrences: int
    has_definition_pattern: bool
    in_glossary: bool
    min_distance: Optional[int]
    reason: str = ""


@dataclass
class AuditReport:
    total: int = 0
    buckets: Dict[AuditBucket, int] = field(default_factory=lambda: {
        AuditBucket.EXTRACTABLE_PAIR: 0,
        AuditBucket.SF_ONLY: 0,
        AuditBucket.UNSUPPORTED_PAIR: 0,
        AuditBucket.PARSE_ISSUE: 0,
    })
    details: List[AuditDetail] = field(default_factory=list)

    @property
    def pair_ceiling(self) -> float:
        if self.total == 0:
            return 0.0
        return self.buckets[AuditBucket.EXTRACTABLE_PAIR] / self.total

    @property
    def sf_ceiling(self) -> float:
        if self.total == 0:
            return 0.0
        non_parse = self.total - self.buckets[AuditBucket.PARSE_ISSUE]
        return non_parse / self.total


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def normalize_text(text: str) -> str:
    """
    Unify hyphens, spaces, and whitespace for consistent matching.
    """
    if not text:
        return ""

    # Unify various dash/hyphen characters to ASCII hyphen
    dash_chars = [
        "\u2010",  # HYPHEN
        "\u2011",  # NON-BREAKING HYPHEN
        "\u2012",  # FIGURE DASH
        "\u2013",  # EN DASH
        "\u2014",  # EM DASH
        "\u2212",  # MINUS SIGN
        "−",       # Another minus
    ]
    for dash in dash_chars:
        text = text.replace(dash, "-")

    # Unify spaces
    text = text.replace("\u00A0", " ")  # NBSP
    text = text.replace("\t", " ")

    # Collapse multiple spaces
    text = re.sub(r" +", " ", text)

    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SF/LF SEARCH (ROBUST)
# ═══════════════════════════════════════════════════════════════════════════════


def find_sf_occurrences(sf: str, text: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of SF in text using lookarounds (not word boundaries).
    Handles hyphenated, parenthesized, and space-containing SFs.

    Returns list of (start, end) spans.
    """
    if not sf or not text:
        return []

    text_n = normalize_text(text)
    sf_n = normalize_text(sf)
    sf_esc = re.escape(sf_n)

    # For SF with spaces (e.g., "CC BY", "IL 2")
    if " " in sf_n:
        # Allow flexible whitespace between parts
        sf_flex = re.escape(sf_n).replace(r"\ ", r"\s+")
        pattern = rf"(?<![A-Za-z0-9]){sf_flex}(?=[\s\)\]\}},;:\-.]|$)"
    else:
        # Standard: lookarounds instead of \b
        pattern = rf"(?<![A-Za-z0-9]){sf_esc}(?![A-Za-z0-9])"

    matches = []
    for m in re.finditer(pattern, text_n, flags=re.IGNORECASE):
        matches.append((m.start(), m.end()))

    return matches


def find_lf_occurrences(lf: str, text: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of LF in text (simple substring, case-insensitive).

    Returns list of (start, end) spans.
    """
    if not lf or not text:
        return []

    text_n = normalize_text(text).lower()
    lf_n = normalize_text(lf).lower()

    spans = []
    start = 0
    while True:
        idx = text_n.find(lf_n, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(lf_n)))
        start = idx + 1

    return spans


# ═══════════════════════════════════════════════════════════════════════════════
# DEFINITION PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def has_definition_pattern(text: str, sf: str, lf: Optional[str]) -> bool:
    """
    Check if text contains an explicit definition pattern linking SF and LF.

    Patterns detected:
    - LF (SF)
    - SF (LF)
    - SF, defined as LF
    - SF stands for LF
    - LF, or SF
    """
    if not lf:
        return False

    text_n = normalize_text(text)
    sf_r = re.escape(normalize_text(sf))
    lf_r = re.escape(normalize_text(lf))

    patterns = [
        # LF (SF) - most common
        rf"{lf_r}\s*\(\s*{sf_r}\s*\)",
        # SF (LF)
        rf"{sf_r}\s*\(\s*{lf_r}\s*\)",
        # SF, defined as LF / SF, abbreviated as LF / SF, hereafter LF
        rf"{sf_r}\s*,?\s*(?:defined as|abbreviated as|hereafter|i\.?e\.?)\s+{lf_r}",
        # SF stands for LF / SF refers to LF
        rf"{sf_r}\s+(?:stands for|refers to|short for|means)\s+{lf_r}",
        # LF, or SF / LF (or SF)
        rf"{lf_r}\s*,?\s*(?:\()?or\s+{sf_r}(?:\))?",
        # LF [SF]
        rf"{lf_r}\s*\[\s*{sf_r}\s*\]",
    ]

    for pattern in patterns:
        if re.search(pattern, text_n, flags=re.IGNORECASE):
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# PROXIMITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════


def min_span_distance(spans_a: List[Tuple[int, int]], spans_b: List[Tuple[int, int]]) -> Optional[int]:
    """
    Calculate minimum distance between any pair of spans.
    Uses midpoint-to-midpoint distance.
    """
    if not spans_a or not spans_b:
        return None

    min_dist = float("inf")
    for (a_start, a_end) in spans_a:
        a_mid = (a_start + a_end) / 2
        for (b_start, b_end) in spans_b:
            b_mid = (b_start + b_end) / 2
            dist = abs(a_mid - b_mid)
            min_dist = min(min_dist, dist)

    return int(min_dist) if min_dist != float("inf") else None


def cooccur_within_k(text: str, sf: str, lf: str, k: int = 500) -> bool:
    """
    Check if SF and LF appear within k characters of each other.
    """
    sf_spans = find_sf_occurrences(sf, text)
    lf_spans = find_lf_occurrences(lf, text)
    dist = min_span_distance(sf_spans, lf_spans)
    return dist is not None and dist <= k


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def classify_bucket(
    sf: str,
    lf: Optional[str],
    text: str,
    glossary_text: str = "",
    proximity_k: int = 500,
) -> Tuple[AuditBucket, str, Dict[str, Any]]:
    """
    Classify a gold annotation into one of the audit buckets.

    Returns (bucket, reason, metadata).
    """
    metadata = {
        "sf_occurrences": 0,
        "lf_occurrences": 0,
        "has_definition_pattern": False,
        "in_glossary": False,
        "min_distance": None,
    }

    # Find SF occurrences
    sf_spans = find_sf_occurrences(sf, text)
    metadata["sf_occurrences"] = len(sf_spans)

    # BUCKET 1: SF not found → PARSE_ISSUE
    if not sf_spans:
        return AuditBucket.PARSE_ISSUE, "SF not found in parsed text", metadata

    # Check definition pattern
    if has_definition_pattern(text, sf, lf):
        metadata["has_definition_pattern"] = True
        return AuditBucket.EXTRACTABLE_PAIR, "Definition pattern found", metadata

    # Check glossary
    if glossary_text:
        glossary_sf_spans = find_sf_occurrences(sf, glossary_text)
        if glossary_sf_spans:
            metadata["in_glossary"] = True
            # Also check if LF is in glossary
            if lf:
                glossary_lf_spans = find_lf_occurrences(lf, glossary_text)
                if glossary_lf_spans:
                    return AuditBucket.EXTRACTABLE_PAIR, "SF+LF in glossary", metadata
            return AuditBucket.EXTRACTABLE_PAIR, "SF in glossary", metadata

    # Check LF existence and proximity
    if lf:
        lf_spans = find_lf_occurrences(lf, text)
        metadata["lf_occurrences"] = len(lf_spans)

        if lf_spans:
            dist = min_span_distance(sf_spans, lf_spans)
            metadata["min_distance"] = dist

            if dist is not None and dist <= proximity_k:
                return AuditBucket.EXTRACTABLE_PAIR, f"SF+LF within {dist} chars", metadata

            # SF and LF both exist but not close → UNSUPPORTED_PAIR
            return AuditBucket.UNSUPPORTED_PAIR, f"SF+LF exist but distance={dist}", metadata

    # SF found but LF not found or no LF provided → SF_ONLY
    return AuditBucket.SF_ONLY, "SF found, LF not extractable", metadata


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AUDIT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def audit_gold(
    gold_path: str,
    pdf_folder: str,
    proximity_k: int = 500,
    verbose: bool = True,
) -> AuditReport:
    """
    Run gold audit on all documents.

    Args:
        gold_path: Path to gold JSON file
        pdf_folder: Folder containing PDFs
        proximity_k: Max distance for SF-LF proximity
        verbose: Print progress

    Returns:
        AuditReport with bucket counts and details
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
    from B_parsing.B03_table_extractor import TableExtractor

    # Load gold
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    annotations = gold_data.get("annotations", [])

    # Group by doc_id
    by_doc: Dict[str, List[Dict]] = {}
    for anno in annotations:
        doc_id = anno.get("doc_id", "")
        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(anno)

    if verbose:
        print(f"Gold annotations: {len(annotations)}")
        print(f"Documents: {len(by_doc)}")
        print()

    # Initialize parser
    parser = PDFToDocGraphParser()
    table_extractor = TableExtractor()

    report = AuditReport()
    pdf_folder_path = Path(pdf_folder)

    for doc_id, doc_annos in by_doc.items():
        # Find PDF
        pdf_path = pdf_folder_path / doc_id
        if not pdf_path.exists():
            # Try without extension matching
            candidates = list(pdf_folder_path.glob(f"*{doc_id.replace('.pdf', '')}*"))
            if candidates:
                pdf_path = candidates[0]
            else:
                if verbose:
                    print(f"⚠ PDF not found: {doc_id}")
                # Mark all as PARSE_ISSUE
                for anno in doc_annos:
                    report.total += 1
                    report.buckets[AuditBucket.PARSE_ISSUE] += 1
                    report.details.append(AuditDetail(
                        doc_id=doc_id,
                        short_form=anno.get("short_form", ""),
                        long_form=anno.get("long_form"),
                        bucket=AuditBucket.PARSE_ISSUE,
                        sf_occurrences=0,
                        lf_occurrences=0,
                        has_definition_pattern=False,
                        in_glossary=False,
                        min_distance=None,
                        reason="PDF not found",
                    ))
                continue

        if verbose:
            print(f"Processing: {doc_id}")

        # Parse PDF
        try:
            doc = parser.parse(str(pdf_path))
            doc = table_extractor.populate_document_graph(doc, str(pdf_path))
        except Exception as e:
            if verbose:
                print(f"  ⚠ Parse error: {e}")
            for anno in doc_annos:
                report.total += 1
                report.buckets[AuditBucket.PARSE_ISSUE] += 1
                report.details.append(AuditDetail(
                    doc_id=doc_id,
                    short_form=anno.get("short_form", ""),
                    long_form=anno.get("long_form"),
                    bucket=AuditBucket.PARSE_ISSUE,
                    sf_occurrences=0,
                    lf_occurrences=0,
                    has_definition_pattern=False,
                    in_glossary=False,
                    min_distance=None,
                    reason=f"Parse error: {e}",
                ))
            continue

        # Extract full text
        full_text = " ".join(
            block.text for block in doc.iter_linear_blocks() if block.text
        )
        full_text = normalize_text(full_text)

        # Extract glossary text (from tables marked as glossary or "abbreviation" tables)
        glossary_parts = []
        for page in doc.pages.values():
            for table in page.tables:
                # Check if table looks like glossary
                table_text = ""
                if hasattr(table, "to_markdown"):
                    table_text = table.to_markdown()
                elif hasattr(table, "rows"):
                    for row in table.rows:
                        table_text += " ".join(str(c) for c in row) + " "

                # Heuristic: glossary tables often have "abbreviation" in headers
                # or have short first column values
                table_lower = table_text.lower()
                if "abbreviation" in table_lower or "acronym" in table_lower:
                    glossary_parts.append(table_text)

        glossary_text = normalize_text(" ".join(glossary_parts))

        # Classify each annotation
        for anno in doc_annos:
            report.total += 1

            sf = anno.get("short_form", "").strip()
            lf = anno.get("long_form", "").strip() if anno.get("long_form") else None

            bucket, reason, meta = classify_bucket(
                sf=sf,
                lf=lf,
                text=full_text,
                glossary_text=glossary_text,
                proximity_k=proximity_k,
            )

            report.buckets[bucket] += 1
            report.details.append(AuditDetail(
                doc_id=doc_id,
                short_form=sf,
                long_form=lf,
                bucket=bucket,
                sf_occurrences=meta["sf_occurrences"],
                lf_occurrences=meta["lf_occurrences"],
                has_definition_pattern=meta["has_definition_pattern"],
                in_glossary=meta["in_glossary"],
                min_distance=meta["min_distance"],
                reason=reason,
            ))

        if verbose:
            doc_buckets = {}
            for d in report.details:
                if d.doc_id == doc_id:
                    doc_buckets[d.bucket] = doc_buckets.get(d.bucket, 0) + 1
            print(f"  {len(doc_annos)} annotations: {dict(doc_buckets)}")

    return report


def print_report(report: AuditReport) -> None:
    """Print formatted audit report."""
    print()
    print("=" * 60)
    print("GOLD AUDIT REPORT")
    print("=" * 60)
    print()
    print(f"Total annotations: {report.total}")
    print()
    print("BUCKET BREAKDOWN:")
    for bucket in AuditBucket:
        count = report.buckets[bucket]
        pct = (count / report.total * 100) if report.total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket.value:20} {count:4} ({pct:5.1f}%) {bar}")

    print()
    print("CEILINGS:")
    print(f"  Pair ceiling (EXTRACTABLE_PAIR):  {report.pair_ceiling * 100:.1f}%")
    print(f"  SF-only ceiling (non-PARSE_ISSUE): {report.sf_ceiling * 100:.1f}%")
    print()

    # Show PARSE_ISSUE details
    parse_issues = [d for d in report.details if d.bucket == AuditBucket.PARSE_ISSUE]
    if parse_issues:
        print("PARSE_ISSUE SFs (not found in text):")
        for d in parse_issues[:20]:
            print(f"  {d.short_form:15} | {d.reason}")
        if len(parse_issues) > 20:
            print(f"  ... and {len(parse_issues) - 20} more")

    print()

    # Show UNSUPPORTED_PAIR details
    unsupported = [d for d in report.details if d.bucket == AuditBucket.UNSUPPORTED_PAIR]
    if unsupported:
        print("UNSUPPORTED_PAIR (SF+LF exist but no evidence):")
        for d in unsupported[:20]:
            dist = d.min_distance or "N/A"
            print(f"  {d.short_form:15} → {(d.long_form or '')[:30]:30} | dist={dist}")
        if len(unsupported) > 20:
            print(f"  ... and {len(unsupported) - 20} more")

    print()

    # Show SF_ONLY details
    sf_only = [d for d in report.details if d.bucket == AuditBucket.SF_ONLY]
    if sf_only:
        print("SF_ONLY (SF found, LF not extractable):")
        for d in sf_only[:20]:
            lf_info = f"LF: {d.long_form[:25]}..." if d.long_form else "no LF in gold"
            print(f"  {d.short_form:15} | {lf_info} | SF occ: {d.sf_occurrences}")
        if len(sf_only) > 20:
            print(f"  ... and {len(sf_only) - 20} more")


def save_report(report: AuditReport, output_path: str) -> None:
    """Save audit report to JSON."""
    data = {
        "total": report.total,
        "buckets": {k.value: v for k, v in report.buckets.items()},
        "pair_ceiling": report.pair_ceiling,
        "sf_ceiling": report.sf_ceiling,
        "details": [
            {
                "doc_id": d.doc_id,
                "short_form": d.short_form,
                "long_form": d.long_form,
                "bucket": d.bucket.value,
                "sf_occurrences": d.sf_occurrences,
                "lf_occurrences": d.lf_occurrences,
                "has_definition_pattern": d.has_definition_pattern,
                "in_glossary": d.in_glossary,
                "min_distance": d.min_distance,
                "reason": d.reason,
            }
            for d in report.details
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Report saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    # Default paths
    GOLD_PATH = "/Users/frederictetard/Projects/ese/gold_data/papers_gold.json"
    PDF_FOLDER = "/Users/frederictetard/Projects/ese/gold_data/PAPERS"
    OUTPUT_PATH = "/Users/frederictetard/Projects/ese/gold_data/audit_report.json"

    print("Running Gold Audit...")
    print(f"Gold: {GOLD_PATH}")
    print(f"PDFs: {PDF_FOLDER}")
    print()

    report = audit_gold(
        gold_path=GOLD_PATH,
        pdf_folder=PDF_FOLDER,
        proximity_k=500,
        verbose=True,
    )

    print_report(report)
    save_report(report, OUTPUT_PATH)