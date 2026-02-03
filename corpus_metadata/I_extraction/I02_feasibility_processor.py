# corpus_metadata/I_extraction/I02_feasibility_processor.py
"""
Feasibility extraction processor.

Handles extraction of clinical trial feasibility information including
eligibility criteria, epidemiology data, patient journey, endpoints,
and site information using LLM and NER-based approaches.

Key Components:
    - FeasibilityProcessor: Main processor coordinating feasibility extraction
    - Base extraction: LLM-based or pattern-based feasibility detection
    - NER enrichers:
        - EpiExtract4GARD-v2: Rare disease epidemiology
        - ZeroShotBioNER: ADE, dosage, frequency, route, duration
        - BiomedicalNER: 84 clinical entity types
        - PatientJourneyNER: Diagnostic delay, treatment lines, care pathway
        - RegistryNER: Registry names, sizes, access policies
        - GeneticNER: Gene symbols, HGVS variants, HPO terms
    - Span deduplication: Removes overlapping NER spans

Example:
    >>> from I_extraction.I02_feasibility_processor import FeasibilityProcessor
    >>> processor = FeasibilityProcessor(
    ...     run_id=run_id,
    ...     feasibility_detector=detector,
    ...     epi_enricher=epi_enricher,
    ... )
    >>> candidates = processor.process(doc, pdf_path, full_text)

Dependencies:
    - A_core.A07_feasibility_models: FeasibilityCandidate, NERCandidate
    - C_generators.C08_strategy_feasibility: FeasibilityDetector
    - C_generators.C11_llm_feasibility: LLMFeasibilityExtractor
    - E_normalization.E08-E15: Various NER enrichers
    - E_normalization.E11_span_deduplicator: deduplicate_feasibility_candidates
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from B_parsing.B02_doc_graph import DocumentGraph
    from C_generators.C08_strategy_feasibility import FeasibilityDetector
    from C_generators.C11_llm_feasibility import LLMFeasibilityExtractor
    from E_normalization.E08_epi_extract_enricher import EpiExtractEnricher
    from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher
    from E_normalization.E10_biomedical_ner_all import BiomedicalNEREnricher
    from E_normalization.E12_patient_journey_enricher import PatientJourneyEnricher
    from E_normalization.E13_registry_enricher import RegistryEnricher
    from E_normalization.E15_genetic_enricher import GeneticEnricher
    from A_core.A07_feasibility_models import FeasibilityCandidate, NERCandidate


def _make_ner_candidate(
    category: str,
    text: str,
    confidence: float,
    source: str,
    evidence_text: Optional[str] = None,
    **kwargs: Any,
) -> "NERCandidate":
    """Create NERCandidate with common defaults.

    Args:
        category: Entity category (e.g., "adverse_event", "gene_symbol")
        text: Primary text for the candidate
        confidence: Confidence score
        source: NER source name (e.g., "ZeroShotBioNER")
        evidence_text: Evidence text (defaults to text)
        **kwargs: Additional NERCandidate fields (e.g., epidemiology_data)
    """
    from A_core.A07_feasibility_models import NERCandidate

    return NERCandidate(
        category=category,
        text=text,
        evidence_text=evidence_text or text,
        confidence=confidence,
        source=source,
        **kwargs,
    )


def _add_ner_entities(
    candidates: list,
    entities: list,
    category: str,
    source: str,
    text_attr: str = "text",
    evidence_attr: Optional[str] = None,
) -> int:
    """Add NER entities as NERCandidates to candidates list.

    Args:
        candidates: Target list to append to
        entities: Source entities with text and score attributes
        category: NER category label
        source: NER source name
        text_attr: Attribute name for primary text (default "text")
        evidence_attr: Attribute name for evidence text (defaults to text_attr)

    Returns:
        Number of entities added
    """
    count = 0
    for entity in entities:
        text = getattr(entity, text_attr, "") or ""
        evidence = getattr(entity, evidence_attr or text_attr, text) or text
        candidates.append(_make_ner_candidate(
            category=category,
            text=text,
            confidence=entity.score,
            source=source,
            evidence_text=evidence,
        ))
        count += 1
    return count


class FeasibilityProcessor:
    """
    Processes clinical trial feasibility information from documents.

    Coordinates multiple extraction strategies including LLM-based extraction,
    pattern-based detection, and specialized NER enrichers for epidemiology,
    drug administration, biomedical entities, patient journey, registry data,
    and genetic information.
    """

    def __init__(
        self,
        run_id: str,
        feasibility_detector: Optional["FeasibilityDetector"] = None,
        llm_feasibility_extractor: Optional["LLMFeasibilityExtractor"] = None,
        epi_enricher: Optional["EpiExtractEnricher"] = None,
        zeroshot_bioner: Optional["ZeroShotBioNEREnricher"] = None,
        biomedical_ner: Optional["BiomedicalNEREnricher"] = None,
        patient_journey_enricher: Optional["PatientJourneyEnricher"] = None,
        registry_enricher: Optional["RegistryEnricher"] = None,
        genetic_enricher: Optional["GeneticEnricher"] = None,
    ) -> None:
        """
        Initialize the feasibility processor.

        Args:
            run_id: Unique identifier for this pipeline run
            feasibility_detector: Pattern-based feasibility detector
            llm_feasibility_extractor: LLM-based feasibility extractor
            epi_enricher: EpiExtract4GARD epidemiology enricher
            zeroshot_bioner: ZeroShot BioNER enricher
            biomedical_ner: Biomedical NER enricher
            patient_journey_enricher: Patient journey enricher
            registry_enricher: Registry enricher
            genetic_enricher: Genetic enricher
        """
        self.run_id = run_id
        self.feasibility_detector = feasibility_detector
        self.llm_feasibility_extractor = llm_feasibility_extractor
        self.epi_enricher = epi_enricher
        self.zeroshot_bioner = zeroshot_bioner
        self.biomedical_ner = biomedical_ner
        self.patient_journey_enricher = patient_journey_enricher
        self.registry_enricher = registry_enricher
        self.genetic_enricher = genetic_enricher

    def process(
        self, doc: "DocumentGraph", pdf_path: Path, full_text: str
    ) -> List[Union["FeasibilityCandidate", "NERCandidate"]]:
        """
        Process document for clinical trial feasibility information.

        Uses LLM-based extraction when available (higher precision),
        falls back to pattern-based extraction otherwise. Enriches results
        with multiple specialized NER models.

        Args:
            doc: Parsed document graph
            pdf_path: Path to the PDF file
            full_text: Full text of the document

        Returns:
            List of feasibility candidates (eligibility, epidemiology, patient journey, etc.)
        """
        if self.feasibility_detector is None:
            return []

        print("\n[10/12] Extracting feasibility information...")
        start = time.time()

        # Extract using LLM or pattern-based approach
        candidates = self._extract_base_candidates(doc, pdf_path, full_text)

        # Apply all enrichers
        candidates = self._enrich_with_epiextract(candidates, full_text)
        candidates = self._enrich_with_zeroshot(candidates, full_text)
        candidates = self._enrich_with_biomedical_ner(candidates, full_text)
        candidates = self._enrich_with_patient_journey(candidates, full_text)
        candidates = self._enrich_with_registry(candidates, full_text)
        candidates = self._enrich_with_genetic(candidates, full_text)

        # Deduplicate overlapping NER spans
        candidates = self._deduplicate_spans(candidates)

        print(f"  Feasibility items: {len(candidates)}")
        print(f"  Time: {time.time() - start:.2f}s")

        return candidates

    def _extract_base_candidates(
        self, doc: "DocumentGraph", pdf_path: Path, full_text: str
    ) -> List[Union["FeasibilityCandidate", "NERCandidate"]]:
        """Extract base feasibility candidates using LLM or patterns."""
        candidates: List[Union["FeasibilityCandidate", "NERCandidate"]] = []

        if self.llm_feasibility_extractor is not None:
            print("  Using LLM-based extraction...")
            candidates = list(self.llm_feasibility_extractor.extract(
                doc_graph=doc,
                doc_id=pdf_path.stem,
                doc_fingerprint=pdf_path.stem,
                full_text=full_text,
            ))
            self.llm_feasibility_extractor.print_summary()
        elif self.feasibility_detector is not None:
            # Fallback to pattern-based extraction
            print("  Using pattern-based extraction...")
            candidates = list(self.feasibility_detector.extract(doc))
            self.feasibility_detector.print_summary()
        else:
            print("  No feasibility extractor available, skipping extraction...")

        return candidates

    def _enrich_with_epiextract(
        self,
        candidates: List[Union["FeasibilityCandidate", "NERCandidate"]],
        full_text: str,
    ) -> List[Union["FeasibilityCandidate", "NERCandidate"]]:
        """Enrich with EpiExtract4GARD-v2 (rare disease epidemiology NER)."""
        if self.epi_enricher is None:
            return candidates

        print("  Running EpiExtract4GARD-v2 enrichment...")
        epi_start = time.time()
        epi_result = self.epi_enricher.extract(full_text)

        epi_data_list = epi_result.to_epidemiology_data()
        if epi_data_list:
            for epi_data in epi_data_list:
                candidates.append(_make_ner_candidate(
                    category="epidemiology",
                    text=epi_data.value,
                    confidence=0.8,
                    source="EpiExtract4GARD-v2",
                    epidemiology_data=epi_data,
                ))
            print(f"    EpiExtract4GARD: {len(epi_data_list)} epidemiology items")
            print(f"      Locations: {len(epi_result.locations)}")
            print(f"      Epi types: {len(epi_result.epi_types)}")
            print(f"      Statistics: {len(epi_result.statistics)}")
        print(f"    EpiExtract time: {time.time() - epi_start:.2f}s")

        return candidates

    def _enrich_with_zeroshot(
        self,
        candidates: List[Union["FeasibilityCandidate", "NERCandidate"]],
        full_text: str,
    ) -> List[Union["FeasibilityCandidate", "NERCandidate"]]:
        """Enrich with ZeroShotBioNER (ADE, dosage, frequency, route, etc.)."""
        if self.zeroshot_bioner is None:
            return candidates

        print("  Running ZeroShotBioNER enrichment...")
        bioner_start = time.time()
        bioner_result = self.zeroshot_bioner.extract(full_text)

        summary = bioner_result.to_summary()
        entity_counts = summary.get("entity_counts", {})
        total_entities = sum(entity_counts.values())

        if total_entities > 0:
            source = "ZeroShotBioNER"
            _add_ner_entities(candidates, bioner_result.adverse_events, "adverse_event", source)
            _add_ner_entities(candidates, bioner_result.dosages, "drug_dosage", source)
            _add_ner_entities(candidates, bioner_result.frequencies, "drug_frequency", source)
            _add_ner_entities(candidates, bioner_result.routes, "drug_route", source)
            _add_ner_entities(candidates, bioner_result.durations, "treatment_duration", source)

            print(f"    ZeroShotBioNER: {total_entities} entities extracted")
            for key in ("ADE", "dosage", "frequency", "route", "duration"):
                print(f"      {key}: {entity_counts.get(key, 0)}")
        print(f"    ZeroShotBioNER time: {time.time() - bioner_start:.2f}s")

        return candidates

    def _enrich_with_biomedical_ner(
        self,
        candidates: List[Union["FeasibilityCandidate", "NERCandidate"]],
        full_text: str,
    ) -> List[Union["FeasibilityCandidate", "NERCandidate"]]:
        """Enrich with d4data/biomedical-ner-all (84 entity types)."""
        if self.biomedical_ner is None:
            return candidates

        print("  Running BiomedicalNER enrichment...")
        biomed_start = time.time()
        biomed_result = self.biomedical_ner.extract(full_text)

        summary = biomed_result.to_summary()
        category_counts = summary.get("category_counts", {})
        total_entities = sum(category_counts.values())

        if total_entities > 0:
            source = "BiomedicalNER"
            # Map clinical entity types to categories
            clinical_type_map = {
                "Sign_symptom": "symptom",
                "Diagnostic_procedure": "diagnostic_procedure",
                "Therapeutic_procedure": "therapeutic_procedure",
                "Lab_value": "lab_value",
                "Outcome": "outcome",
            }
            for entity in biomed_result.clinical:
                category = clinical_type_map.get(entity.entity_type)
                if category:
                    candidates.append(_make_ner_candidate(
                        category=category, text=entity.text,
                        confidence=entity.score, source=source,
                    ))

            # Demographics use dynamic category
            for entity in biomed_result.demographics:
                candidates.append(_make_ner_candidate(
                    category=f"demographics_{entity.entity_type.lower()}",
                    text=entity.text, confidence=entity.score, source=source,
                ))

            print(f"    BiomedicalNER: {total_entities} entities extracted")
            for key in ("clinical", "demographics", "temporal", "anatomical"):
                print(f"      {key.capitalize()}: {category_counts.get(key, 0)}")
        print(f"    BiomedicalNER time: {time.time() - biomed_start:.2f}s")

        return candidates

    def _enrich_with_patient_journey(
        self,
        candidates: List[Union["FeasibilityCandidate", "NERCandidate"]],
        full_text: str,
    ) -> List[Union["FeasibilityCandidate", "NERCandidate"]]:
        """Enrich with PatientJourneyNER (diagnostic delay, treatment lines, care pathway)."""
        if self.patient_journey_enricher is None:
            return candidates

        print("  Running PatientJourneyNER enrichment...")
        pj_start = time.time()
        pj_result = self.patient_journey_enricher.extract(full_text)

        summary = pj_result.to_summary()
        total_entities = summary.get("total", 0)

        if total_entities > 0:
            source = "PatientJourneyNER"
            entity_fields = [
                ("diagnostic_delays", "diagnostic_delay"),
                ("treatment_lines", "treatment_line"),
                ("care_pathway_steps", "care_pathway_step"),
                ("surveillance_frequencies", "surveillance_frequency"),
                ("pain_points", "pain_point"),
                ("recruitment_touchpoints", "recruitment_touchpoint"),
            ]
            for attr, category in entity_fields:
                entities = getattr(pj_result, attr, [])
                _add_ner_entities(candidates, entities, category, source)
                print(f"      {category}: {summary.get(category, 0)}")
        print(f"    PatientJourney: {total_entities} entities extracted")
        print(f"    PatientJourney time: {time.time() - pj_start:.2f}s")

        return candidates

    def _enrich_with_registry(
        self,
        candidates: List[Union["FeasibilityCandidate", "NERCandidate"]],
        full_text: str,
    ) -> List[Union["FeasibilityCandidate", "NERCandidate"]]:
        """Enrich with RegistryNER (registry names, sizes, data types, access policies)."""
        if self.registry_enricher is None:
            return candidates

        print("  Running RegistryNER enrichment...")
        reg_start = time.time()
        reg_result = self.registry_enricher.extract(full_text)

        summary = reg_result.to_summary()
        total_entities = summary.get("total", 0)

        if total_entities > 0:
            source = "RegistryNER"
            entity_fields = [
                ("registry_names", "registry_name"),
                ("registry_sizes", "registry_size"),
                ("geographic_coverages", "geographic_coverage"),
                ("data_types", "data_types"),
                ("access_policies", "access_policy"),
                ("eligibility_criteria", "eligibility_criteria"),
            ]
            for attr, category in entity_fields:
                entities = getattr(reg_result, attr, [])
                _add_ner_entities(candidates, entities, category, source)
                print(f"      {category}: {summary.get(category, 0)}")

            # Report linked registries
            linked = reg_result.get_linked_registries()
            if linked:
                print(f"    Linked to known registries: {len(linked)}")
                for lr in linked[:3]:
                    print(f"      {lr.get('extracted_text')} -> {lr.get('full_name', 'N/A')}")
        print(f"    RegistryNER: {total_entities} entities extracted")
        print(f"    RegistryNER time: {time.time() - reg_start:.2f}s")

        return candidates

    def _enrich_with_genetic(
        self,
        candidates: List[Union["FeasibilityCandidate", "NERCandidate"]],
        full_text: str,
    ) -> List[Union["FeasibilityCandidate", "NERCandidate"]]:
        """Enrich with GeneticNER (gene symbols, HGVS variants, HPO, ORDO)."""
        if self.genetic_enricher is None:
            return candidates

        print("  Running GeneticNER enrichment...")
        gen_start = time.time()
        gen_result = self.genetic_enricher.extract(full_text)

        summary = gen_result.to_summary()
        total_entities = summary.get("total", 0)

        if total_entities > 0:
            source = "GeneticNER"
            # Genetic entities use 'normalized' as primary text and 'text' as evidence
            entity_fields = [
                ("gene_symbols", "gene_symbol"),
                ("variants_hgvs", "variant_hgvs"),
                ("variants_rsid", "variant_rsid"),
                ("hpo_terms", "hpo_term"),
                ("disease_ordo", "disease_ordo"),
            ]
            for attr, category in entity_fields:
                entities = getattr(gen_result, attr, [])
                _add_ner_entities(
                    candidates, entities, category, source,
                    text_attr="normalized", evidence_attr="text",
                )
                print(f"      {attr}: {summary.get(attr, 0)}")

            # Report gene-variant associations
            genes_with_variants = gen_result.get_genes_with_variants()
            if genes_with_variants:
                print(f"    Gene-variant associations: {len(genes_with_variants)} genes")
                for gene, variants in list(genes_with_variants.items())[:3]:
                    print(f"      {gene}: {len(variants)} variant(s)")
        print(f"    GeneticNER: {total_entities} entities extracted")
        print(f"    GeneticNER time: {time.time() - gen_start:.2f}s")

        return candidates

    def _deduplicate_spans(
        self,
        candidates: List[Union["FeasibilityCandidate", "NERCandidate"]],
    ) -> List[Union["FeasibilityCandidate", "NERCandidate"]]:
        """Deduplicate overlapping NER spans (keep highest confidence)."""
        from E_normalization.E11_span_deduplicator import deduplicate_feasibility_candidates

        pre_dedup_count = len(candidates)
        candidates, dedup_result = deduplicate_feasibility_candidates(candidates)

        if dedup_result.merged_count > 0:
            print("  Span deduplication:")
            print(f"    Before: {pre_dedup_count} -> After: {len(candidates)}")
            print(f"    Merged: {dedup_result.merged_count} overlapping spans")
            summary = dedup_result.to_summary()
            print(f"    By source: {summary.get('by_source', {})}")

        return candidates


__all__ = ["FeasibilityProcessor"]
