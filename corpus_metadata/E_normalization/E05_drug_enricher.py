# corpus_metadata/corpus_metadata/E_normalization/E05_drug_enricher.py
"""
PubTator3 API integration for drug/chemical enrichment.

Enriches extracted drugs with:
- MeSH identifiers (if missing)
- Normalized drug names
- Aliases/synonyms from PubTator

Reuses PubTator3Client from E04_pubtator_enricher.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from A_core.A06_drug_models import (
    DrugIdentifier,
    ExtractedDrug,
)
from E_normalization.E04_pubtator_enricher import PubTator3Client


class DrugEnricher:
    """
    Enriches ExtractedDrug entities with PubTator3 chemical data.

    Enrichment includes:
    - MeSH ID (if missing)
    - Normalized name from PubTator
    - Aliases/synonyms
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.client = PubTator3Client(config)

        # Enrichment settings
        enrichment_cfg = config.get("enrichment", {})
        self.enrich_missing_mesh = enrichment_cfg.get("enrich_missing_mesh", True)
        self.add_aliases = enrichment_cfg.get("add_aliases", True)
        self.enabled = config.get("enabled", True)

    def enrich(self, drug: ExtractedDrug) -> ExtractedDrug:
        """
        Enrich a single drug entity with PubTator data.

        Args:
            drug: Validated drug entity

        Returns:
            Enriched drug entity (or original if enrichment fails/unnecessary)
        """
        if not self.enabled:
            return drug

        # Skip if already has MeSH and we're not adding aliases
        if drug.mesh_id and not self.add_aliases:
            return drug

        # Query PubTator using preferred name (query as "chemical")
        results = self.client.autocomplete(drug.preferred_name, "chemical")

        # Try brand name if no results
        if not results and drug.brand_name:
            results = self.client.autocomplete(drug.brand_name, "chemical")

        # Try compound ID if still no results
        if not results and drug.compound_id:
            results = self.client.autocomplete(drug.compound_id, "chemical")

        if not results:
            return drug

        # Get best match (first result)
        best = results[0]

        # Prepare updates
        updates: Dict[str, Any] = {}
        new_identifiers = list(drug.identifiers)

        # Add MeSH ID if missing
        # PubTator3 returns db_id for MeSH ID
        if self.enrich_missing_mesh and not drug.mesh_id:
            mesh_id = best.get("db_id", "")
            if mesh_id and best.get("db") == "ncbi_mesh":
                updates["mesh_id"] = mesh_id

                # Add to identifiers list
                new_identifiers.append(
                    DrugIdentifier(
                        system="MeSH",
                        code=mesh_id,
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
                updates["mesh_aliases"] = other_names

        # Mark as enriched
        if updates:
            updates["enrichment_source"] = "pubtator3"

            # Add flag
            current_flags = list(drug.validation_flags or [])
            if "pubtator_enriched" not in current_flags:
                current_flags.append("pubtator_enriched")
                updates["validation_flags"] = current_flags

            return drug.model_copy(update=updates)

        return drug

    def enrich_batch(
        self, drugs: List[ExtractedDrug], verbose: bool = True
    ) -> List[ExtractedDrug]:
        """
        Enrich a batch of drug entities.

        Args:
            drugs: List of validated drug entities
            verbose: Print progress

        Returns:
            List of enriched drug entities
        """
        if not self.enabled:
            return drugs

        enriched = []
        enriched_count = 0

        for drug in drugs:
            result = self.enrich(drug)
            enriched.append(result)

            if "pubtator_enriched" in (result.validation_flags or []):
                if "pubtator_enriched" not in (drug.validation_flags or []):
                    enriched_count += 1

        if verbose and enriched_count > 0:
            print(f"    PubTator enriched: {enriched_count}/{len(drugs)} drugs")

        return enriched
