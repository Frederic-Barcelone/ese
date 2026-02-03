"""
PubTator3 API integration for gene enrichment.

This module enriches extracted genes using the NCBI PubTator3 API, adding
NCBI Gene IDs (Entrez), normalized names, and aliases. Reuses PubTator3Client
from E04 for consistent caching and rate limiting.

Key Components:
    - GeneEnricher: Enricher implementing BaseEnricher interface
    - Enrichment features:
        - NCBI Gene ID / Entrez ID (if missing)
        - Normalized gene name from PubTator
        - Aliases/synonyms

Example:
    >>> from E_normalization.E18_gene_enricher import GeneEnricher
    >>> enricher = GeneEnricher()
    >>> enriched = enricher.enrich(gene_entity)
    >>> print(f"Entrez: {enriched.entrez_id}, Normalized: {enriched.pubtator_normalized_name}")
    Entrez: 672, Normalized: BRCA1

API Reference: https://www.ncbi.nlm.nih.gov/research/pubtator3/api

Dependencies:
    - A_core.A00_logging: Logging utilities
    - A_core.A02_interfaces: BaseEnricher
    - A_core.A19_gene_models: GeneIdentifier, ExtractedGene
    - E_normalization.E04_pubtator_enricher: PubTator3Client
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from A_core.A00_logging import get_logger
from A_core.A02_interfaces import BaseEnricher
from A_core.A19_gene_models import (
    EnrichmentSource,
    GeneIdentifier,
    ExtractedGene,
)
from E_normalization.E04_pubtator_enricher import PubTator3Client

logger = get_logger(__name__)


class GeneEnricher(BaseEnricher[ExtractedGene, ExtractedGene]):
    """
    Enriches ExtractedGene entities with PubTator3 gene data.

    Enrichment includes:
    - NCBI Gene ID / Entrez ID (if missing)
    - Normalized gene name from PubTator
    - Aliases/synonyms

    PubTator3 returns gene entities with:
    - name: Normalized gene symbol
    - db_id: NCBI Gene ID (Entrez ID)
    - db: "ncbi_gene"
    - biotype: "gene"

    Attributes:
        client: PubTator3Client for API access.
        enrich_missing_entrez: Add Entrez IDs to genes missing them.
        add_aliases: Add aliases from PubTator.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.client = PubTator3Client(self.config)

        # Enrichment settings
        enrichment_cfg = self.config.get("enrichment", {})
        self.enrich_missing_entrez = enrichment_cfg.get("enrich_missing_entrez", True)
        self.add_aliases = enrichment_cfg.get("add_aliases", True)

        logger.debug(
            f"GeneEnricher initialized: entrez={self.enrich_missing_entrez}, "
            f"aliases={self.add_aliases}, enabled={self.enabled}"
        )

    @property
    def enricher_name(self) -> str:
        """Return the enricher identifier."""
        return "gene_enricher"

    def enrich(self, gene: ExtractedGene) -> ExtractedGene:
        """
        Enrich a single gene entity with PubTator data.

        Args:
            gene: Validated gene entity

        Returns:
            Enriched gene entity (or original if enrichment fails/unnecessary)
        """
        if not self.enabled:
            return gene

        # Skip if already has Entrez ID and we're not adding aliases
        if gene.entrez_id and not self.add_aliases:
            return gene

        try:
            return self._enrich_impl(gene)
        except Exception as e:
            logger.warning(
                f"Gene enrichment failed for {gene.hgnc_symbol}: {e}. "
                "Returning original gene without enrichment."
            )
            return gene

    def _enrich_impl(self, gene: ExtractedGene) -> ExtractedGene:
        """Internal enrichment implementation (may raise exceptions)."""
        # Query PubTator using HGNC symbol (query as "gene")
        results = self.client.autocomplete(gene.hgnc_symbol, "gene")

        # Try matched text if no results (might be an alias)
        if not results and gene.matched_text != gene.hgnc_symbol:
            results = self.client.autocomplete(gene.matched_text, "gene")

        # Try full name if still no results
        if not results and gene.full_name:
            results = self.client.autocomplete(gene.full_name, "gene")

        if not results:
            return gene

        # Get best match (first result)
        best = results[0]

        # Prepare updates
        updates: Dict[str, Any] = {}
        new_identifiers = list(gene.identifiers)

        # Add Entrez ID if missing
        # PubTator3 returns db_id for NCBI Gene ID (Entrez)
        if self.enrich_missing_entrez and not gene.entrez_id:
            entrez_id = best.get("db_id", "")
            if entrez_id and best.get("db") == "ncbi_gene":
                updates["entrez_id"] = entrez_id

                # Add to identifiers list
                new_identifiers.append(
                    GeneIdentifier(
                        system="ENTREZ",
                        code=entrez_id,
                        display=best.get("name"),
                    )
                )
                updates["identifiers"] = new_identifiers

        # Add normalized name from PubTator
        if self.add_aliases:
            normalized_name = best.get("name")
            if normalized_name:
                updates["pubtator_normalized_name"] = normalized_name

            # Extract potential aliases from related entries
            other_names = [
                r.get("name")
                for r in results[1:4]  # Get up to 3 related names
                if r.get("name") and r.get("name") != normalized_name
            ]
            if other_names:
                updates["pubtator_aliases"] = other_names

        # Mark as enriched
        if updates:
            updates["enrichment_source"] = EnrichmentSource.PUBTATOR3

            # Add flag
            current_flags = list(gene.validation_flags or [])
            if "pubtator_enriched" not in current_flags:
                current_flags.append("pubtator_enriched")
                updates["validation_flags"] = current_flags

            return gene.model_copy(update=updates)

        return gene

    def enrich_batch(
        self, genes: List[ExtractedGene], verbose: bool = True
    ) -> List[ExtractedGene]:
        """
        Enrich a batch of gene entities.

        Args:
            genes: List of validated gene entities.
            verbose: Log progress information.

        Returns:
            List of enriched gene entities.
        """
        if not self.enabled:
            return genes

        enriched = []
        enriched_count = 0

        for gene in genes:
            result = self.enrich(gene)
            enriched.append(result)

            if "pubtator_enriched" in (result.validation_flags or []):
                if "pubtator_enriched" not in (gene.validation_flags or []):
                    enriched_count += 1

        if verbose and enriched_count > 0:
            logger.info(f"PubTator enriched: {enriched_count}/{len(genes)} genes")

        return enriched


__all__ = ["GeneEnricher"]
