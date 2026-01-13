# corpus_metadata/C_generators/C00_strategy_identifiers.py
"""
Identifier Extractor - Extract standardized IDs from scientific/medical documents.

Target: Database identifiers, publication references, and standardized codes.

Supported ID Types:
  - OMIM: Online Mendelian Inheritance in Man (e.g., OMIM #118450, MIM:118450)
  - Orphanet: Rare disease database (e.g., ORPHA:123, Orphanet:123)
  - DOI: Digital Object Identifier (e.g., 10.1000/xyz123)
  - PMID: PubMed ID (e.g., PMID:12345678, PMID 12345678)
  - PMC: PubMed Central (e.g., PMC1234567)
  - NCT: ClinicalTrials.gov (e.g., NCT01234567)
  - ORCID: Researcher identifier (e.g., 0000-0002-1234-5678)
  - ISRCTN: International Standard Randomised Controlled Trial Number
  - EudraCT: European Clinical Trials Database (e.g., 2020-001234-56)
  - UMLS CUI: Unified Medical Language System (e.g., C0123456)
  - MeSH: Medical Subject Headings (e.g., D012345)
  - ICD-10: International Classification of Diseases (e.g., E11.9)
  - SNOMED CT: (e.g., 123456789)

Usage:
    extractor = IdentifierExtractor()
    identifiers = extractor.extract(doc_graph)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from A_core.A01_domain_models import (
    Candidate,
    GeneratorType,
)
from A_core.A02_interfaces import BaseCandidateGenerator
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from B_parsing.B02_doc_graph import DocumentGraph


class IdentifierType(str, Enum):
    """Types of identifiers that can be extracted."""

    OMIM = "OMIM"
    ORPHANET = "Orphanet"
    DOI = "DOI"
    PMID = "PMID"
    PMC = "PMC"
    NCT = "NCT"
    ORCID = "ORCID"
    ISRCTN = "ISRCTN"
    EUDRACT = "EudraCT"
    UMLS_CUI = "UMLS_CUI"
    MESH = "MeSH"
    ICD10 = "ICD-10"
    SNOMED = "SNOMED"
    MONDO = "MONDO"
    HP = "HPO"  # Human Phenotype Ontology
    GENE = "Gene"  # Gene symbols like JAG1, NOTCH2
    UNKNOWN = "Unknown"


@dataclass
class ExtractedIdentifier:
    """Represents an extracted identifier."""

    id_type: IdentifierType
    value: str  # The full identifier (e.g., "OMIM #118450")
    normalized: str  # Normalized form (e.g., "118450")
    context: str  # Surrounding text
    start_pos: int
    end_pos: int
    confidence: float = 0.95
    long_form: Optional[str] = None  # Expanded name (e.g., trial title for NCT IDs)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.id_type.value,
            "value": self.value,
            "normalized": self.normalized,
            "context": self.context[:200] if self.context else "",
            "confidence": self.confidence,
        }
        if self.long_form:
            result["long_form"] = self.long_form
        return result


# Identifier patterns with named groups
IDENTIFIER_PATTERNS: Dict[IdentifierType, List[re.Pattern]] = {
    IdentifierType.OMIM: [
        re.compile(r"(?:OMIM|MIM)\s*[#:]?\s*(\d{6})", re.IGNORECASE),
        re.compile(r"#(\d{6})\b"),  # Standalone # followed by 6 digits
    ],
    IdentifierType.ORPHANET: [
        re.compile(r"(?:ORPHA|Orphanet)\s*[:#]?\s*(\d+)", re.IGNORECASE),
    ],
    IdentifierType.DOI: [
        re.compile(r"(?:doi[:\s]*)?10\.\d{4,9}/[^\s\])<>\"]+", re.IGNORECASE),
        re.compile(r"https?://doi\.org/(10\.\d{4,9}/[^\s\])<>\"]+)", re.IGNORECASE),
    ],
    IdentifierType.PMID: [
        re.compile(r"PMID[:\s]*(\d{7,8})", re.IGNORECASE),
        re.compile(r"PubMed\s*(?:ID)?[:\s]*(\d{7,8})", re.IGNORECASE),
    ],
    IdentifierType.PMC: [
        re.compile(r"PMC\s*(\d{6,8})", re.IGNORECASE),
    ],
    IdentifierType.NCT: [
        re.compile(r"NCT\s*(\d{8})", re.IGNORECASE),
        re.compile(r"ClinicalTrials\.gov[:\s]*NCT(\d{8})", re.IGNORECASE),
    ],
    IdentifierType.ORCID: [
        re.compile(r"(?:ORCID[:\s]*)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", re.IGNORECASE),
    ],
    IdentifierType.ISRCTN: [
        re.compile(r"ISRCTN\s*(\d{8,})", re.IGNORECASE),
    ],
    IdentifierType.EUDRACT: [
        re.compile(r"(\d{4}-\d{6}-\d{2})", re.IGNORECASE),  # EudraCT format
    ],
    IdentifierType.UMLS_CUI: [
        re.compile(r"\b(C\d{7})\b"),  # UMLS CUI format
    ],
    IdentifierType.MESH: [
        re.compile(r"MeSH[:\s]*(D\d{6,9})", re.IGNORECASE),
    ],
    IdentifierType.ICD10: [
        re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b"),  # ICD-10 format
    ],
    IdentifierType.MONDO: [
        re.compile(r"MONDO[:\s]*(\d{7})", re.IGNORECASE),
    ],
    IdentifierType.HP: [
        re.compile(r"HP[:\s]*(\d{7})", re.IGNORECASE),
    ],
}

# Gene symbol patterns (common gene naming conventions)
GENE_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9]{1,5}(?:-[A-Z0-9]+)?)\b"  # e.g., JAG1, NOTCH2, BRCA1
)

# Known gene symbols to validate against (subset of common ones)
KNOWN_GENES = {
    "JAG1",
    "NOTCH2",
    "BRCA1",
    "BRCA2",
    "TP53",
    "EGFR",
    "KRAS",
    "BRAF",
    "PIK3CA",
    "PTEN",
    "APC",
    "MLH1",
    "MSH2",
    "MSH6",
    "PMS2",
    "CFTR",
    "HBB",
    "HBA1",
    "HBA2",
    "DMD",
    "FBN1",
    "COL1A1",
    "COL1A2",
    "NF1",
    "NF2",
    "TSC1",
    "TSC2",
    "VHL",
    "RB1",
    "WT1",
    "MEN1",
    "RET",
    "FGFR1",
    "FGFR2",
    "FGFR3",
    "ACVRL1",
    "ENG",
    "SMAD4",
    "GDF2",  # HHT genes
}


class IdentifierExtractor(BaseCandidateGenerator):
    """
    Extract standardized identifiers from scientific/medical documents.

    This generator identifies database references (OMIM, Orphanet, etc.),
    publication identifiers (DOI, PMID), clinical trial IDs (NCT),
    and other standardized codes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Which identifier types to extract
        self.enabled_types: Set[IdentifierType] = set(
            self.config.get("enabled_types", list(IdentifierType))
        )

        # Whether to extract gene symbols
        self.extract_genes = self.config.get("extract_genes", True)

        # Context window size
        self.context_window = int(self.config.get("context_window", 100))

        # Provenance
        self.pipeline_version = str(
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.run_id = str(self.config.get("run_id") or generate_run_id("ID"))

    @property
    def generator_type(self) -> GeneratorType:
        # Using a custom type for identifiers
        return GeneratorType.LEXICON_MATCH  # Reuse existing type

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        """Extract identifiers from document. Returns empty list (identifiers stored separately)."""
        # This method returns Candidates for compatibility, but identifiers
        # are better handled separately. Use extract_identifiers() for full extraction.
        return []

    def extract_identifiers(
        self, doc_structure: DocumentGraph
    ) -> List[ExtractedIdentifier]:
        """
        Extract all identifiers from the document.

        Returns:
            List of ExtractedIdentifier objects
        """
        doc = doc_structure
        results: List[ExtractedIdentifier] = []
        seen: Set[Tuple[str, str]] = set()  # (type, normalized) dedup

        # Build full text from all blocks
        full_text = ""
        for block in doc.iter_linear_blocks(skip_header_footer=False):
            if block.text:
                full_text += block.text + " "

        full_text = full_text.strip()
        if not full_text:
            return results

        # Extract each identifier type
        for id_type, patterns in IDENTIFIER_PATTERNS.items():
            if id_type not in self.enabled_types:
                continue

            for pattern in patterns:
                for match in pattern.finditer(full_text):
                    # Get the full match and normalized value
                    full_match = match.group(0)

                    # Try to get captured group (normalized), fallback to full match
                    try:
                        normalized = match.group(1) if match.lastindex else full_match
                    except IndexError:
                        normalized = full_match

                    # Dedup
                    key = (id_type.value, normalized)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Get context
                    start = max(0, match.start() - self.context_window)
                    end = min(len(full_text), match.end() + self.context_window)
                    context = full_text[start:end]

                    results.append(
                        ExtractedIdentifier(
                            id_type=id_type,
                            value=full_match,
                            normalized=normalized,
                            context=context,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=0.95,
                        )
                    )

        # Extract gene symbols if enabled
        if self.extract_genes:
            for match in GENE_PATTERN.finditer(full_text):
                symbol = match.group(1)

                # Only include known genes or genes followed by "gene" keyword
                is_known = symbol in KNOWN_GENES
                has_gene_context = bool(
                    re.search(
                        rf"\b{re.escape(symbol)}\s+gene\b", full_text, re.IGNORECASE
                    )
                )

                if not (is_known or has_gene_context):
                    continue

                key = (IdentifierType.GENE.value, symbol)
                if key in seen:
                    continue
                seen.add(key)

                start = max(0, match.start() - self.context_window)
                end = min(len(full_text), match.end() + self.context_window)
                context = full_text[start:end]

                results.append(
                    ExtractedIdentifier(
                        id_type=IdentifierType.GENE,
                        value=symbol,
                        normalized=symbol,
                        context=context,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.90 if is_known else 0.75,
                    )
                )

        return results

    def get_identifier_blacklist(self, doc_structure: DocumentGraph) -> Set[str]:
        """
        Get set of identifier strings that should be excluded from abbreviation extraction.

        Returns:
            Set of strings like "OMIM", "NCT01234567", etc.
        """
        blacklist: Set[str] = set()

        identifiers = self.extract_identifiers(doc_structure)

        for ident in identifiers:
            # Add the full value
            blacklist.add(ident.value)

            # Add common prefixes that shouldn't be abbreviations
            if ident.id_type == IdentifierType.OMIM:
                blacklist.add("OMIM")
                blacklist.add("MIM")
            elif ident.id_type == IdentifierType.ORPHANET:
                blacklist.add("ORPHA")
                blacklist.add("Orphanet")
            elif ident.id_type == IdentifierType.NCT:
                blacklist.add(f"NCT{ident.normalized}")
            elif ident.id_type == IdentifierType.GENE:
                # Don't blacklist gene symbols - they might have valid expansions
                pass

        return blacklist


def extract_identifiers_from_text(
    text: str, config: Optional[Dict] = None
) -> List[ExtractedIdentifier]:
    """
    Convenience function to extract identifiers from raw text.

    Args:
        text: Raw text to extract from
        config: Optional configuration dict

    Returns:
        List of ExtractedIdentifier objects
    """
    results: List[ExtractedIdentifier] = []
    seen: Set[Tuple[str, str]] = set()
    context_window = 100

    for id_type, patterns in IDENTIFIER_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                full_match = match.group(0)
                try:
                    normalized = match.group(1) if match.lastindex else full_match
                except IndexError:
                    normalized = full_match

                key = (id_type.value, normalized)
                if key in seen:
                    continue
                seen.add(key)

                start = max(0, match.start() - context_window)
                end = min(len(text), match.end() + context_window)
                context = text[start:end]

                results.append(
                    ExtractedIdentifier(
                        id_type=id_type,
                        value=full_match,
                        normalized=normalized,
                        context=context,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                )

    return results


def enrich_nct_identifiers(
    identifiers: List[ExtractedIdentifier],
    config: Optional[Dict] = None,
) -> List[ExtractedIdentifier]:
    """
    Enrich NCT identifiers with trial information from ClinicalTrials.gov.

    Args:
        identifiers: List of extracted identifiers
        config: Optional configuration for the enricher

    Returns:
        Same list with NCT identifiers enriched with long_form (trial title)
    """
    from E_normalization.E06_nct_enricher import get_nct_enricher

    # Find NCT identifiers that need enrichment
    nct_identifiers = [
        ident for ident in identifiers
        if ident.id_type == IdentifierType.NCT and not ident.long_form
    ]

    if not nct_identifiers:
        return identifiers

    # Get enricher instance (reuses cached client)
    enricher = get_nct_enricher(config)

    # Enrich each NCT identifier
    for ident in nct_identifiers:
        nct_id = f"NCT{ident.normalized}"
        info = enricher.enrich(nct_id)
        if info and info.long_form:
            ident.long_form = info.long_form

    return identifiers


if __name__ == "__main__":
    # Quick test
    test_text = """
    Alagille syndrome (OMIM #118450) is a rare genetic disorder.
    Most people have mutations in the JAG1 gene. A small percentage
    have mutations in NOTCH2. See DOI: 10.1038/nature12345 and
    PMID: 12345678 for more information. Clinical trial NCT04817618
    is currently recruiting. The disease is also known as ORPHA:52
    in the Orphanet database.
    """

    identifiers = extract_identifiers_from_text(test_text)

    print("Extracted Identifiers:")
    print("-" * 60)
    for ident in identifiers:
        print(
            f"  {ident.id_type.value:12} | {ident.value:30} | norm: {ident.normalized}"
        )

    # Test NCT enrichment
    print("\n" + "=" * 60)
    print("Testing NCT Enrichment:")
    print("-" * 60)
    identifiers = enrich_nct_identifiers(identifiers)
    for ident in identifiers:
        if ident.id_type == IdentifierType.NCT:
            print(f"  NCT ID: {ident.value}")
            print(f"  Long Form: {ident.long_form}")
