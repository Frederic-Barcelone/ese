# corpus_metadata/I_extraction/I02_feasibility_processor.py
"""
Feasibility extraction processor.

Handles extraction of clinical trial feasibility information including
eligibility criteria, epidemiology data, patient journey, endpoints,
and site information using LLM and NER-based approaches.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING, Union

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

        from A_core.A07_feasibility_models import NERCandidate

        print("  Running EpiExtract4GARD-v2 enrichment...")
        epi_start = time.time()
        epi_result = self.epi_enricher.extract(full_text)

        # Convert to EpidemiologyData and add as feasibility candidates
        epi_data_list = epi_result.to_epidemiology_data()
        if epi_data_list:
            for epi_data in epi_data_list:
                # Create a NERCandidate for each epidemiology finding
                epi_candidate = NERCandidate(
                    category="epidemiology",
                    text=epi_data.value,
                    evidence_text=epi_data.value,
                    confidence=0.8,
                    source="EpiExtract4GARD-v2",
                    epidemiology_data=epi_data,
                )
                candidates.append(epi_candidate)
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

        from A_core.A07_feasibility_models import NERCandidate

        print("  Running ZeroShotBioNER enrichment...")
        bioner_start = time.time()
        bioner_result = self.zeroshot_bioner.extract(full_text)

        # Add extracted entities as feasibility candidates
        summary = bioner_result.to_summary()
        entity_counts = summary.get("entity_counts", {})
        total_entities = sum(entity_counts.values())

        if total_entities > 0:
            # Add adverse events as candidates
            for ade in bioner_result.adverse_events:
                candidates.append(NERCandidate(
                    category="adverse_event",
                    text=ade.text,
                    evidence_text=ade.text,
                    confidence=ade.score,
                    source="ZeroShotBioNER",
                ))

            # Add drug administration details as candidates
            for dosage in bioner_result.dosages:
                candidates.append(NERCandidate(
                    category="drug_dosage",
                    text=dosage.text,
                    evidence_text=dosage.text,
                    confidence=dosage.score,
                    source="ZeroShotBioNER",
                ))
            for freq in bioner_result.frequencies:
                candidates.append(NERCandidate(
                    category="drug_frequency",
                    text=freq.text,
                    evidence_text=freq.text,
                    confidence=freq.score,
                    source="ZeroShotBioNER",
                ))
            for route in bioner_result.routes:
                candidates.append(NERCandidate(
                    category="drug_route",
                    text=route.text,
                    evidence_text=route.text,
                    confidence=route.score,
                    source="ZeroShotBioNER",
                ))
            for duration in bioner_result.durations:
                candidates.append(NERCandidate(
                    category="treatment_duration",
                    text=duration.text,
                    evidence_text=duration.text,
                    confidence=duration.score,
                    source="ZeroShotBioNER",
                ))

            print(f"    ZeroShotBioNER: {total_entities} entities extracted")
            print(f"      ADE: {entity_counts.get('ADE', 0)}")
            print(f"      Dosage: {entity_counts.get('dosage', 0)}")
            print(f"      Frequency: {entity_counts.get('frequency', 0)}")
            print(f"      Route: {entity_counts.get('route', 0)}")
            print(f"      Duration: {entity_counts.get('duration', 0)}")
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

        from A_core.A07_feasibility_models import NERCandidate

        print("  Running BiomedicalNER enrichment...")
        biomed_start = time.time()
        biomed_result = self.biomedical_ner.extract(full_text)

        # Add extracted entities as feasibility candidates
        summary = biomed_result.to_summary()
        category_counts = summary.get("category_counts", {})
        total_entities = sum(category_counts.values())

        if total_entities > 0:
            # Add clinical entities
            for entity in biomed_result.clinical:
                if entity.entity_type == "Sign_symptom":
                    candidates.append(NERCandidate(
                        category="symptom",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="BiomedicalNER",
                    ))
                elif entity.entity_type == "Diagnostic_procedure":
                    candidates.append(NERCandidate(
                        category="diagnostic_procedure",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="BiomedicalNER",
                    ))
                elif entity.entity_type == "Therapeutic_procedure":
                    candidates.append(NERCandidate(
                        category="therapeutic_procedure",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="BiomedicalNER",
                    ))
                elif entity.entity_type == "Lab_value":
                    candidates.append(NERCandidate(
                        category="lab_value",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="BiomedicalNER",
                    ))
                elif entity.entity_type == "Outcome":
                    candidates.append(NERCandidate(
                        category="outcome",
                        text=entity.text,
                        evidence_text=entity.text,
                        confidence=entity.score,
                        source="BiomedicalNER",
                    ))

            # Add demographics
            for entity in biomed_result.demographics:
                candidates.append(NERCandidate(
                    category=f"demographics_{entity.entity_type.lower()}",
                    text=entity.text,
                    evidence_text=entity.text,
                    confidence=entity.score,
                    source="BiomedicalNER",
                ))

            print(f"    BiomedicalNER: {total_entities} entities extracted")
            print(f"      Clinical: {category_counts.get('clinical', 0)}")
            print(f"      Demographics: {category_counts.get('demographics', 0)}")
            print(f"      Temporal: {category_counts.get('temporal', 0)}")
            print(f"      Anatomical: {category_counts.get('anatomical', 0)}")
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

        from A_core.A07_feasibility_models import NERCandidate

        print("  Running PatientJourneyNER enrichment...")
        pj_start = time.time()
        pj_result = self.patient_journey_enricher.extract(full_text)

        # Add extracted entities as feasibility candidates
        summary = pj_result.to_summary()
        total_entities = summary.get("total", 0)

        if total_entities > 0:
            # Add diagnostic delays
            for pj_entity in pj_result.diagnostic_delays:
                candidates.append(NERCandidate(
                    category="diagnostic_delay",
                    text=pj_entity.text,
                    evidence_text=pj_entity.text,
                    confidence=pj_entity.score,
                    source="PatientJourneyNER",
                ))

            # Add treatment lines
            for pj_entity in pj_result.treatment_lines:
                candidates.append(NERCandidate(
                    category="treatment_line",
                    text=pj_entity.text,
                    evidence_text=pj_entity.text,
                    confidence=pj_entity.score,
                    source="PatientJourneyNER",
                ))

            # Add care pathway steps
            for pj_entity in pj_result.care_pathway_steps:
                candidates.append(NERCandidate(
                    category="care_pathway_step",
                    text=pj_entity.text,
                    evidence_text=pj_entity.text,
                    confidence=pj_entity.score,
                    source="PatientJourneyNER",
                ))

            # Add surveillance frequencies
            for pj_entity in pj_result.surveillance_frequencies:
                candidates.append(NERCandidate(
                    category="surveillance_frequency",
                    text=pj_entity.text,
                    evidence_text=pj_entity.text,
                    confidence=pj_entity.score,
                    source="PatientJourneyNER",
                ))

            # Add pain points
            for pj_entity in pj_result.pain_points:
                candidates.append(NERCandidate(
                    category="pain_point",
                    text=pj_entity.text,
                    evidence_text=pj_entity.text,
                    confidence=pj_entity.score,
                    source="PatientJourneyNER",
                ))

            # Add recruitment touchpoints
            for pj_entity in pj_result.recruitment_touchpoints:
                candidates.append(NERCandidate(
                    category="recruitment_touchpoint",
                    text=pj_entity.text,
                    evidence_text=pj_entity.text,
                    confidence=pj_entity.score,
                    source="PatientJourneyNER",
                ))

            print(f"      diagnostic_delay: {summary.get('diagnostic_delay', 0)}")
            print(f"      treatment_line: {summary.get('treatment_line', 0)}")
            print(f"      care_pathway_step: {summary.get('care_pathway_step', 0)}")
            print(f"      surveillance_frequency: {summary.get('surveillance_frequency', 0)}")
            print(f"      pain_point: {summary.get('pain_point', 0)}")
            print(f"      recruitment_touchpoint: {summary.get('recruitment_touchpoint', 0)}")
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

        from A_core.A07_feasibility_models import NERCandidate

        print("  Running RegistryNER enrichment...")
        reg_start = time.time()
        reg_result = self.registry_enricher.extract(full_text)

        # Add extracted entities as feasibility candidates
        summary = reg_result.to_summary()
        total_entities = summary.get("total", 0)

        if total_entities > 0:
            # Add registry names
            for reg_entity in reg_result.registry_names:
                candidates.append(NERCandidate(
                    category="registry_name",
                    text=reg_entity.text,
                    evidence_text=reg_entity.text,
                    confidence=reg_entity.score,
                    source="RegistryNER",
                ))

            # Add registry sizes
            for reg_entity in reg_result.registry_sizes:
                candidates.append(NERCandidate(
                    category="registry_size",
                    text=reg_entity.text,
                    evidence_text=reg_entity.text,
                    confidence=reg_entity.score,
                    source="RegistryNER",
                ))

            # Add geographic coverage
            for reg_entity in reg_result.geographic_coverages:
                candidates.append(NERCandidate(
                    category="geographic_coverage",
                    text=reg_entity.text,
                    evidence_text=reg_entity.text,
                    confidence=reg_entity.score,
                    source="RegistryNER",
                ))

            # Add data types
            for reg_entity in reg_result.data_types:
                candidates.append(NERCandidate(
                    category="data_types",
                    text=reg_entity.text,
                    evidence_text=reg_entity.text,
                    confidence=reg_entity.score,
                    source="RegistryNER",
                ))

            # Add access policies
            for reg_entity in reg_result.access_policies:
                candidates.append(NERCandidate(
                    category="access_policy",
                    text=reg_entity.text,
                    evidence_text=reg_entity.text,
                    confidence=reg_entity.score,
                    source="RegistryNER",
                ))

            # Add eligibility criteria
            for reg_entity in reg_result.eligibility_criteria:
                candidates.append(NERCandidate(
                    category="eligibility_criteria",
                    text=reg_entity.text,
                    evidence_text=reg_entity.text,
                    confidence=reg_entity.score,
                    source="RegistryNER",
                ))

            print(f"      registry_name: {summary.get('registry_name', 0)}")
            print(f"      registry_size: {summary.get('registry_size', 0)}")
            print(f"      geographic_coverage: {summary.get('geographic_coverage', 0)}")
            print(f"      data_types: {summary.get('data_types', 0)}")
            print(f"      access_policy: {summary.get('access_policy', 0)}")
            print(f"      eligibility_criteria: {summary.get('eligibility_criteria', 0)}")

            # Report linked registries
            linked = reg_result.get_linked_registries()
            if linked:
                print(f"    Linked to known registries: {len(linked)}")
                for lr in linked[:3]:  # Show first 3
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

        from A_core.A07_feasibility_models import NERCandidate

        print("  Running GeneticNER enrichment...")
        gen_start = time.time()
        gen_result = self.genetic_enricher.extract(full_text)

        # Add extracted entities as feasibility candidates
        summary = gen_result.to_summary()
        total_entities = summary.get("total", 0)

        if total_entities > 0:
            # Add gene symbols
            for gen_entity in gen_result.gene_symbols:
                candidates.append(NERCandidate(
                    category="gene_symbol",
                    text=gen_entity.normalized,
                    evidence_text=gen_entity.text,
                    confidence=gen_entity.score,
                    source="GeneticNER",
                ))

            # Add HGVS variants
            for gen_entity in gen_result.variants_hgvs:
                candidates.append(NERCandidate(
                    category="variant_hgvs",
                    text=gen_entity.normalized,
                    evidence_text=gen_entity.text,
                    confidence=gen_entity.score,
                    source="GeneticNER",
                ))

            # Add rsID variants
            for gen_entity in gen_result.variants_rsid:
                candidates.append(NERCandidate(
                    category="variant_rsid",
                    text=gen_entity.normalized,
                    evidence_text=gen_entity.text,
                    confidence=gen_entity.score,
                    source="GeneticNER",
                ))

            # Add HPO terms
            for gen_entity in gen_result.hpo_terms:
                candidates.append(NERCandidate(
                    category="hpo_term",
                    text=gen_entity.normalized,
                    evidence_text=gen_entity.text,
                    confidence=gen_entity.score,
                    source="GeneticNER",
                ))

            # Add ORDO disease codes
            for gen_entity in gen_result.disease_ordo:
                candidates.append(NERCandidate(
                    category="disease_ordo",
                    text=gen_entity.normalized,
                    evidence_text=gen_entity.text,
                    confidence=gen_entity.score,
                    source="GeneticNER",
                ))

            print(f"      gene_symbols: {summary.get('gene_symbols', 0)}")
            print(f"      variants_hgvs: {summary.get('variants_hgvs', 0)}")
            print(f"      variants_rsid: {summary.get('variants_rsid', 0)}")
            print(f"      hpo_terms: {summary.get('hpo_terms', 0)}")
            print(f"      disease_ordo: {summary.get('disease_ordo', 0)}")

            # Report gene-variant associations
            genes_with_variants = gen_result.get_genes_with_variants()
            if genes_with_variants:
                print(f"    Gene-variant associations: {len(genes_with_variants)} genes")
                for gene, variants in list(genes_with_variants.items())[:3]:  # Show first 3
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
