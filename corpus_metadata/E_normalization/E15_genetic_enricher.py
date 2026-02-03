"""
Genetic and phenotype enricher for rare disease trial feasibility.

This module extracts genetic variants, gene names, and HPO phenotypes critical
for rare disease trial eligibility assessment. Uses clinically-validated regex
patterns for HGVS notation, dbSNP IDs, and ontology codes.

Key Components:
    - GeneticEnricher: Main enricher using regex pattern extraction
    - Entity types extracted:
        - gene_symbol: Gene names (GBA, CFTR, DMD, SMN1)
        - variant_hgvs: HGVS variants (c.1226A>G, p.N370S)
        - variant_rsid: dbSNP IDs (rs76763715)
        - hpo_term: HPO phenotype codes (HP:0001744)
        - disease_ordo: Orphanet codes (ORPHA:355)
    - Clinically-validated regex patterns:
        - HGVS coding: c.123A>G, c.456_789del, c.100+1G>A
        - HGVS protein: p.N370S, p.Arg123Ter, p.Gly456fs
        - Gene symbols: 2-6 uppercase letters

Example:
    >>> from E_normalization.E15_genetic_enricher import GeneticEnricher
    >>> enricher = GeneticEnricher()
    >>> result = enricher.extract("Patients with GBA c.1226A>G mutation...")
    >>> for variant in result.variants_hgvs:
    ...     print(f"Variant: {variant.text} (gene: {variant.gene})")
    Variant: c.1226A>G (gene: GBA)

Dependencies:
    - re: Regular expressions for pattern matching
    - Optional: HGNC gene list for symbol validation
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from A_core.A00_logging import get_logger

# Module logger
logger = get_logger(__name__)


# =============================================================================
# REGEX PATTERNS FOR GENETIC ENTITIES
# =============================================================================

# HGVS coding DNA variants (c. notation)
# Matches: c.123A>G, c.456del, c.789_790insT, c.100+1G>A (splice), c.-10C>T (5'UTR)
HGVS_CODING_PATTERN = re.compile(
    r"\bc\.(?:"
    r"[-*]?\d+(?:[+-]\d+)?(?:_[-*]?\d+(?:[+-]\d+)?)?"  # Position(s)
    r"(?:"
    r"[ACGT]>[ACGT]|"  # Substitution
    r"del[ACGT]*|"  # Deletion
    r"ins[ACGT]+|"  # Insertion
    r"dup[ACGT]*|"  # Duplication
    r"inv|"  # Inversion
    r"[ACGT]+>[ACGT]+"  # Complex
    r")"
    r")",
    re.IGNORECASE,
)

# HGVS protein variants (p. notation)
# Matches: p.N370S, p.Arg123Ter, p.Gly456fs, p.Met1?, p.Leu123_Val125del
HGVS_PROTEIN_PATTERN = re.compile(
    r"\bp\.(?:"
    r"[A-Z][a-z]{2}\d+(?:_[A-Z][a-z]{2}\d+)?(?:[A-Z][a-z]{2}|Ter|fs|del|ins|dup|\?)|"  # 3-letter
    r"[ACDEFGHIKLMNPQRSTVWY]\d+(?:_[ACDEFGHIKLMNPQRSTVWY]\d+)?(?:[ACDEFGHIKLMNPQRSTVWY]|Ter|fs|del|ins|dup|\*|\?)"  # 1-letter
    r")",
    re.IGNORECASE,
)

# dbSNP rsID pattern
# Matches: rs76763715, rs121908755
RSID_PATTERN = re.compile(r"\brs\d{6,12}\b", re.IGNORECASE)

# HPO term pattern
# Matches: HP:0001744, HP:0000001
HPO_PATTERN = re.compile(r"\bHP[:\s]?\d{7}\b", re.IGNORECASE)

# Orphanet/ORDO disease code pattern
# Matches: ORPHA:355, ORDO:355, Orphanet:355
ORPHA_PATTERN = re.compile(
    r"\b(?:ORPHA|ORDO|Orphanet)[:\s]?\d{1,6}\b",
    re.IGNORECASE,
)

# Gene symbol pattern (validated subset of HGNC)
# Common rare disease genes - expandable
KNOWN_GENE_SYMBOLS: Set[str] = {
    # Lysosomal storage disorders
    "GBA", "GBA1", "GAA", "HEXA", "HEXB", "IDUA", "IDS", "SGSH", "NAGLU",
    "GLA", "GALNS", "GLB1", "ARSB", "GUSB", "HYAL1", "NPC1", "NPC2",
    # Cystic fibrosis
    "CFTR",
    # Muscular dystrophies
    "DMD", "DYSF", "SGCA", "SGCB", "SGCG", "SGCD", "CAPN3", "FKRP", "ANO5",
    # Spinal muscular atrophy
    "SMN1", "SMN2",
    # Huntington's / neurodegeneration
    "HTT", "ATXN1", "ATXN2", "ATXN3", "CACNA1A", "ATXN7", "TBP",
    # Hemophilia / coagulation
    "F8", "F9", "VWF", "FGA", "FGB", "FGG",
    # Complement pathway (C3G, aHUS)
    "CFH", "CFI", "CFB", "C3", "CD46", "MCP", "THBD", "DGKE",
    # Other common rare disease genes
    "SERPINA1", "HBB", "HBA1", "HBA2", "PAH", "GALT", "OTC", "ASS1",
    "PKD1", "PKD2", "COL4A3", "COL4A4", "COL4A5",
    # Oncology (targeted therapies)
    "BRCA1", "BRCA2", "EGFR", "ALK", "KRAS", "NRAS", "BRAF", "TP53",
    "PTEN", "PIK3CA", "HER2", "ERBB2", "MET", "ROS1", "RET", "NTRK1",
}

# Gene symbol regex - matches gene-like patterns
GENE_SYMBOL_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9]{1,5})\b(?:\s+gene|\s+mutation|\s+variant)?",
)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class GeneticEntity:
    """
    Single genetic entity extracted from text.

    Attributes:
        text: The extracted text span.
        entity_type: Category (gene_symbol, variant_hgvs, variant_rsid, hpo_term, disease_ordo).
        normalized: Normalized form (uppercase gene, standardized HPO format).
        gene: Associated gene symbol if applicable.
        score: Confidence score (1.0 for regex matches).
        start: Character start position.
        end: Character end position.
    """

    text: str
    entity_type: str
    normalized: str
    gene: Optional[str] = None
    score: float = 1.0
    start: int = 0
    end: int = 0

    def __repr__(self) -> str:
        """Return concise string representation."""
        gene_str = f" ({self.gene})" if self.gene else ""
        return f"GeneticEntity({self.entity_type}: '{self.text}'{gene_str})"


@dataclass
class GeneticResult:
    """
    Structured result from genetic entity extraction.

    Attributes:
        gene_symbols: Extracted gene names (GBA, CFTR, etc.).
        variants_hgvs: HGVS-formatted variants (c. and p. notation).
        variants_rsid: dbSNP reference IDs.
        hpo_terms: Human Phenotype Ontology terms.
        disease_ordo: Orphanet rare disease codes.
        raw_entities: All extracted entities for debugging.
        extraction_time_seconds: Processing time.
    """

    gene_symbols: List[GeneticEntity] = field(default_factory=list)
    variants_hgvs: List[GeneticEntity] = field(default_factory=list)
    variants_rsid: List[GeneticEntity] = field(default_factory=list)
    hpo_terms: List[GeneticEntity] = field(default_factory=list)
    disease_ordo: List[GeneticEntity] = field(default_factory=list)

    raw_entities: List[Dict[str, Any]] = field(default_factory=list)
    extraction_time_seconds: float = 0.0

    def to_summary(self) -> Dict[str, Any]:
        """
        Convert to summary dictionary for logging/export.

        Returns:
            Dictionary with entity counts per category.
        """
        return {
            "gene_symbols": len(self.gene_symbols),
            "variants_hgvs": len(self.variants_hgvs),
            "variants_rsid": len(self.variants_rsid),
            "hpo_terms": len(self.hpo_terms),
            "disease_ordo": len(self.disease_ordo),
            "total": self.total_entities,
            "extraction_time_seconds": self.extraction_time_seconds,
        }

    @property
    def total_entities(self) -> int:
        """Total number of extracted entities."""
        return (
            len(self.gene_symbols)
            + len(self.variants_hgvs)
            + len(self.variants_rsid)
            + len(self.hpo_terms)
            + len(self.disease_ordo)
        )

    def get_genes_with_variants(self) -> Dict[str, List[GeneticEntity]]:
        """
        Group variants by their associated gene.

        Returns:
            Dictionary mapping gene symbols to their variants.
        """
        result: Dict[str, List[GeneticEntity]] = {}
        for variant in self.variants_hgvs + self.variants_rsid:
            if variant.gene:
                if variant.gene not in result:
                    result[variant.gene] = []
                result[variant.gene].append(variant)
        return result


# =============================================================================
# ENRICHER CLASS
# =============================================================================

class GeneticEnricher:
    """
    Enricher for genetic variant and phenotype extraction.

    Uses regex patterns to extract gene symbols, HGVS variants,
    HPO phenotypes, and Orphanet disease codes from clinical text.

    Attributes:
        validate_genes: Whether to validate gene symbols against known list.
        extract_hpo: Whether to extract HPO phenotype terms.
        extract_ordo: Whether to extract Orphanet disease codes.
        confidence_threshold: Minimum confidence (not used for regex).

    Example:
        >>> enricher = GeneticEnricher()
        >>> result = enricher.extract("Patients with GBA c.1226A>G...")
        >>> print(result.to_summary())
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the genetic enricher.

        Args:
            config: Optional configuration dictionary with keys:
                - validate_genes: Validate against known gene list (default: True)
                - extract_hpo: Extract HPO terms (default: True)
                - extract_ordo: Extract Orphanet codes (default: True)
                - additional_genes: Set of additional gene symbols to recognize
        """
        config = config or {}
        self.validate_genes: bool = config.get("validate_genes", True)
        self.extract_hpo: bool = config.get("extract_hpo", True)
        self.extract_ordo: bool = config.get("extract_ordo", True)

        # Allow extending gene list
        additional_genes = config.get("additional_genes", set())
        self.known_genes: Set[str] = KNOWN_GENE_SYMBOLS | set(additional_genes)

        logger.debug(
            f"GeneticEnricher initialized: {len(self.known_genes)} known genes, "
            f"HPO={self.extract_hpo}, ORDO={self.extract_ordo}"
        )

    def extract(self, text: str) -> GeneticResult:
        """
        Extract genetic entities from text.

        Scans for gene symbols, HGVS variants, rsIDs, HPO terms,
        and Orphanet disease codes.

        Args:
            text: Input text (document, section, or abstract).

        Returns:
            GeneticResult with categorized entities.
        """
        result = GeneticResult()
        start_time = time.time()

        if not text or not text.strip():
            logger.debug("Empty text provided, returning empty result")
            return result

        seen: Set[str] = set()  # Deduplication

        # Extract HGVS coding variants (c. notation)
        for match in HGVS_CODING_PATTERN.finditer(text):
            variant_text = match.group(0)
            normalized = self._normalize_hgvs(variant_text)

            if normalized not in seen:
                seen.add(normalized)
                gene = self._find_nearby_gene(text, match.start(), match.end())

                entity = GeneticEntity(
                    text=variant_text,
                    entity_type="variant_hgvs",
                    normalized=normalized,
                    gene=gene,
                    start=match.start(),
                    end=match.end(),
                )
                result.variants_hgvs.append(entity)
                result.raw_entities.append(self._entity_to_dict(entity))

        # Extract HGVS protein variants (p. notation)
        for match in HGVS_PROTEIN_PATTERN.finditer(text):
            variant_text = match.group(0)
            normalized = self._normalize_hgvs(variant_text)

            if normalized not in seen:
                seen.add(normalized)
                gene = self._find_nearby_gene(text, match.start(), match.end())

                entity = GeneticEntity(
                    text=variant_text,
                    entity_type="variant_hgvs",
                    normalized=normalized,
                    gene=gene,
                    start=match.start(),
                    end=match.end(),
                )
                result.variants_hgvs.append(entity)
                result.raw_entities.append(self._entity_to_dict(entity))

        # Extract rsIDs
        for match in RSID_PATTERN.finditer(text):
            rsid_text = match.group(0)
            normalized = rsid_text.lower()

            if normalized not in seen:
                seen.add(normalized)
                gene = self._find_nearby_gene(text, match.start(), match.end())

                entity = GeneticEntity(
                    text=rsid_text,
                    entity_type="variant_rsid",
                    normalized=normalized,
                    gene=gene,
                    start=match.start(),
                    end=match.end(),
                )
                result.variants_rsid.append(entity)
                result.raw_entities.append(self._entity_to_dict(entity))

        # Extract gene symbols
        for match in GENE_SYMBOL_PATTERN.finditer(text):
            gene_text = match.group(1)
            normalized = gene_text.upper()

            # Validate against known genes if enabled
            if self.validate_genes and normalized not in self.known_genes:
                continue

            if normalized not in seen:
                seen.add(normalized)

                entity = GeneticEntity(
                    text=gene_text,
                    entity_type="gene_symbol",
                    normalized=normalized,
                    start=match.start(),
                    end=match.end(),
                )
                result.gene_symbols.append(entity)
                result.raw_entities.append(self._entity_to_dict(entity))

        # Extract HPO terms
        if self.extract_hpo:
            for match in HPO_PATTERN.finditer(text):
                hpo_text = match.group(0)
                # Normalize to HP:XXXXXXX format
                normalized = re.sub(r"HP[:\s]?", "HP:", hpo_text.upper())

                if normalized not in seen:
                    seen.add(normalized)

                    entity = GeneticEntity(
                        text=hpo_text,
                        entity_type="hpo_term",
                        normalized=normalized,
                        start=match.start(),
                        end=match.end(),
                    )
                    result.hpo_terms.append(entity)
                    result.raw_entities.append(self._entity_to_dict(entity))

        # Extract Orphanet codes
        if self.extract_ordo:
            for match in ORPHA_PATTERN.finditer(text):
                orpha_text = match.group(0)
                # Normalize to ORPHA:XXXXX format
                normalized = re.sub(
                    r"(?:ORPHA|ORDO|Orphanet)[:\s]?",
                    "ORPHA:",
                    orpha_text,
                    flags=re.IGNORECASE,
                )

                if normalized not in seen:
                    seen.add(normalized)

                    entity = GeneticEntity(
                        text=orpha_text,
                        entity_type="disease_ordo",
                        normalized=normalized,
                        start=match.start(),
                        end=match.end(),
                    )
                    result.disease_ordo.append(entity)
                    result.raw_entities.append(self._entity_to_dict(entity))

        result.extraction_time_seconds = time.time() - start_time
        logger.debug(f"Extracted {result.total_entities} genetic entities")

        return result

    def _normalize_hgvs(self, variant: str) -> str:
        """
        Normalize HGVS variant notation.

        Args:
            variant: Raw variant string.

        Returns:
            Normalized variant (lowercase c./p., uppercase bases).
        """
        # Keep c. or p. prefix lowercase, uppercase the rest
        if variant.lower().startswith("c."):
            return "c." + variant[2:].upper()
        elif variant.lower().startswith("p."):
            return "p." + variant[2:]  # Keep amino acid case
        return variant

    def _find_nearby_gene(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 100,
    ) -> Optional[str]:
        """
        Find a gene symbol near a variant mention.

        Args:
            text: Full text.
            start: Variant start position.
            end: Variant end position.
            window: Character window to search.

        Returns:
            Nearest gene symbol, or None.
        """
        # Look in window before and after
        search_start = max(0, start - window)
        search_end = min(len(text), end + window)
        context = text[search_start:search_end]

        for match in GENE_SYMBOL_PATTERN.finditer(context):
            gene = match.group(1).upper()
            if gene in self.known_genes:
                return gene

        return None

    def _entity_to_dict(self, entity: GeneticEntity) -> Dict[str, Any]:
        """Convert entity to dictionary for raw_entities."""
        return {
            "text": entity.text,
            "type": entity.entity_type,
            "normalized": entity.normalized,
            "gene": entity.gene,
            "score": entity.score,
            "start": entity.start,
            "end": entity.end,
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def extract_genetic_entities(
    text: str,
    config: Optional[Dict[str, Any]] = None,
) -> GeneticResult:
    """
    Convenience function for quick genetic entity extraction.

    Args:
        text: Input text to analyze.
        config: Optional configuration dictionary.

    Returns:
        GeneticResult with extracted entities.

    Example:
        >>> result = extract_genetic_entities("GBA c.1226A>G mutation")
        >>> print(f"Found {len(result.variants_hgvs)} variants")
    """
    enricher = GeneticEnricher(config)
    return enricher.extract(text)


__all__ = [
    "KNOWN_GENE_SYMBOLS",
    "GeneticEntity",
    "GeneticResult",
    "GeneticEnricher",
    "extract_genetic_entities",
]
