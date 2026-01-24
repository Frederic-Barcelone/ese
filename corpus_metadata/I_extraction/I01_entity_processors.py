# corpus_metadata/I_extraction/I01_entity_processors.py
"""
Entity processors for extracting and validating domain-specific entities.

Handles processing of diseases, genes, drugs, pharma companies, authors,
and citations from parsed documents.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from B_parsing.B02_doc_graph import DocumentGraph
    from A_core.A01_domain_models import (
        Candidate,
        ExtractedEntity,
        FieldType,
        ValidationStatus,
    )
    from A_core.A05_disease_models import DiseaseCandidate, ExtractedDisease
    from A_core.A06_drug_models import DrugCandidate, ExtractedDrug
    from A_core.A12_gene_models import GeneCandidate, ExtractedGene
    from A_core.A09_pharma_models import ExtractedPharma
    from A_core.A10_author_models import ExtractedAuthor
    from A_core.A11_citation_models import ExtractedCitation
    from C_generators.C06_strategy_disease import DiseaseDetector
    from C_generators.C07_strategy_drug import DrugDetector
    from C_generators.C16_strategy_gene import GeneDetector
    from C_generators.C12_strategy_pharma import PharmaCompanyDetector
    from C_generators.C13_strategy_author import AuthorDetector
    from C_generators.C14_strategy_citation import CitationDetector
    from E_normalization.E03_disease_normalizer import DiseaseNormalizer

from A_core.A03_provenance import hash_string
from Z_utils.Z02_text_helpers import extract_context_snippet


class EntityProcessor:
    """
    Processes domain-specific entities from parsed documents.

    Coordinates detection, validation, and entity creation for diseases,
    genes, drugs, pharma companies, authors, and citations.
    """

    def __init__(
        self,
        run_id: str,
        pipeline_version: str,
        disease_detector: Optional["DiseaseDetector"] = None,
        disease_normalizer: Optional["DiseaseNormalizer"] = None,
        drug_detector: Optional["DrugDetector"] = None,
        gene_detector: Optional["GeneDetector"] = None,
        pharma_detector: Optional["PharmaCompanyDetector"] = None,
        author_detector: Optional["AuthorDetector"] = None,
        citation_detector: Optional["CitationDetector"] = None,
    ) -> None:
        """
        Initialize the entity processor.

        Args:
            run_id: Unique identifier for this pipeline run
            pipeline_version: Version string for the pipeline
            disease_detector: Disease detection component
            disease_normalizer: Disease normalization component
            drug_detector: Drug detection component
            gene_detector: Gene detection component
            pharma_detector: Pharma company detection component
            author_detector: Author detection component
            citation_detector: Citation detection component
        """
        self.run_id = run_id
        self.pipeline_version = pipeline_version
        self.disease_detector = disease_detector
        self.disease_normalizer = disease_normalizer
        self.drug_detector = drug_detector
        self.gene_detector = gene_detector
        self.pharma_detector = pharma_detector
        self.author_detector = author_detector
        self.citation_detector = citation_detector

        # Optional enrichers (can be set after init)
        self.disease_enricher = None
        self.drug_enricher = None

    # =========================================================================
    # ENTITY CREATION HELPERS
    # =========================================================================

    def create_entity_from_candidate(
        self,
        candidate: "Candidate",
        status: "ValidationStatus",
        confidence: float,
        reason: str,
        flags: List[str],
        raw_response: Dict,
        long_form_override: Optional[str] = None,
    ) -> "ExtractedEntity":
        """Create ExtractedEntity from a Candidate (for auto-approve/reject)."""
        from A_core.A01_domain_models import EvidenceSpan, ExtractedEntity, ValidationStatus

        context = (candidate.context_text or "").strip()
        ctx_hash = hash_string(context) if context else "no_context"
        primary = EvidenceSpan(
            text=context,
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=len(context),
        )
        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            field_type=candidate.field_type,
            short_form=candidate.short_form.strip(),
            long_form=long_form_override
            or (candidate.long_form.strip() if candidate.long_form else None),
            primary_evidence=primary,
            supporting_evidence=[],
            status=status,
            confidence_score=confidence,
            rejection_reason=reason if status == ValidationStatus.REJECTED else None,
            validation_flags=flags,
            provenance=candidate.provenance,
            raw_llm_response=raw_response,
        )

    def create_entity_from_search(
        self,
        doc_id: str,
        full_text: str,
        match: re.Match,
        long_form: Optional[str],
        field_type: "FieldType",
        confidence: float,
        flags: List[str],
        rule_version: str,
        lexicon_source: str,
    ) -> "ExtractedEntity":
        """Create ExtractedEntity from a text search match."""
        from A_core.A01_domain_models import (
            Coordinate,
            EvidenceSpan,
            ExtractedEntity,
            GeneratorType,
            ProvenanceMetadata,
            ValidationStatus,
        )

        context_snippet = extract_context_snippet(full_text, match.start(), match.end())
        ctx_hash = hash_string(context_snippet)

        primary = EvidenceSpan(
            text=context_snippet,
            location=Coordinate(page_num=1),
            scope_ref=ctx_hash,
            start_char_offset=match.start() - max(0, match.start() - 100),
            end_char_offset=match.end() - max(0, match.start() - 100),
        )

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=lexicon_source,
            generator_name=GeneratorType.LEXICON_MATCH,
            rule_version=rule_version,
            lexicon_source=f"orchestrator:{lexicon_source}",
        )

        return ExtractedEntity(
            candidate_id=uuid.uuid4(),
            doc_id=doc_id,
            field_type=field_type,
            short_form=match.group(),
            long_form=long_form,
            primary_evidence=primary,
            supporting_evidence=[],
            status=ValidationStatus.VALIDATED,
            confidence_score=confidence,
            rejection_reason=None,
            validation_flags=flags,
            provenance=prov,
            raw_llm_response={"auto": lexicon_source},
        )

    # =========================================================================
    # DISEASE PROCESSING
    # =========================================================================

    def process_diseases(self, doc: "DocumentGraph", pdf_path: Path) -> List["ExtractedDisease"]:
        """
        Process document for disease mentions.

        Returns validated disease entities.
        """
        if self.disease_detector is None:
            return []

        print("\n[5/12] Detecting disease mentions...")
        start = time.time()

        # Generate disease candidates
        candidates = self.disease_detector.extract(doc)
        print(f"  Disease candidates: {len(candidates)}")

        if not candidates:
            print(f"  Time: {time.time() - start:.2f}s")
            return []

        # Auto-validate based on source (specialized lexicons are high trust)
        results: List["ExtractedDisease"] = []
        for candidate in candidates:
            # Specialized lexicons (PAH, ANCA, IgAN) are auto-validated
            is_specialized = "specialized" in candidate.generator_type.value

            entity = self._create_disease_entity(
                candidate=candidate,
                status_validated=True,
                confidence=candidate.initial_confidence + candidate.confidence_boost,
                flags=["auto_validated_lexicon"]
                if is_specialized
                else ["lexicon_match"],
            )
            results.append(entity)

        # Normalize diseases
        if self.disease_normalizer is not None:
            results = self.disease_normalizer.normalize_batch(results)

        # PubTator enrichment
        if self.disease_enricher is not None:
            print("  Enriching with PubTator3...")
            results = self.disease_enricher.enrich_batch(results, verbose=True)

        validated_count = sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
        print(f"  Validated diseases: {validated_count}")
        print(f"  Time: {time.time() - start:.2f}s")

        return results

    def _create_disease_entity(
        self,
        candidate: "DiseaseCandidate",
        status_validated: bool,
        confidence: float,
        flags: List[str],
    ) -> "ExtractedDisease":
        """Create ExtractedDisease from a DiseaseCandidate."""
        from A_core.A01_domain_models import EvidenceSpan, ValidationStatus
        from A_core.A05_disease_models import ExtractedDisease

        context = (candidate.context_text or "").strip()
        ctx_hash = hash_string(context) if context else "no_context"

        primary = EvidenceSpan(
            text=context,
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=len(context),
        )

        # Extract primary codes from identifiers
        icd10 = None
        icd11 = None
        snomed = None
        mondo = None
        orpha = None
        umls = None
        mesh = None

        for ident in candidate.identifiers:
            if ident.system in ("ICD-10", "ICD-10-CM") and not icd10:
                icd10 = ident.code
            elif ident.system == "ICD-11" and not icd11:
                icd11 = ident.code
            elif ident.system in ("SNOMED-CT", "SNOMED") and not snomed:
                snomed = ident.code
            elif ident.system == "MONDO" and not mondo:
                mondo = ident.code
            elif ident.system in ("ORPHA", "Orphanet") and not orpha:
                orpha = ident.code
            elif ident.system == "UMLS" and not umls:
                umls = ident.code
            elif ident.system == "MeSH" and not mesh:
                mesh = ident.code

        return ExtractedDisease(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            matched_text=candidate.matched_text,
            preferred_label=candidate.preferred_label,
            abbreviation=candidate.abbreviation,
            identifiers=candidate.identifiers,
            icd10_code=icd10,
            icd11_code=icd11,
            snomed_code=snomed,
            mondo_id=mondo,
            orpha_code=orpha,
            umls_cui=umls,
            mesh_id=mesh,
            primary_evidence=primary,
            supporting_evidence=[],
            status=ValidationStatus.VALIDATED if status_validated else ValidationStatus.REJECTED,
            confidence_score=min(1.0, confidence),
            validation_flags=flags,
            is_rare_disease=candidate.is_rare_disease,
            disease_category=candidate.disease_category,
            provenance=candidate.provenance,
        )

    # =========================================================================
    # GENE PROCESSING
    # =========================================================================

    def process_genes(self, doc: "DocumentGraph", pdf_path: Path) -> List["ExtractedGene"]:
        """
        Process document for gene mentions (rare disease-associated).

        Returns validated gene entities.
        """
        if self.gene_detector is None:
            return []

        print("\n[5b/12] Detecting gene mentions...")
        start = time.time()

        # Generate gene candidates
        candidates = self.gene_detector.detect(doc)
        print(f"  Gene candidates: {len(candidates)}")

        if not candidates:
            print(f"  Time: {time.time() - start:.2f}s")
            return []

        # Auto-validate based on source
        results: List["ExtractedGene"] = []
        for candidate in candidates:
            entity = self._create_gene_entity(
                candidate=candidate,
                status_validated=True,
                confidence=candidate.initial_confidence,
                flags=["auto_validated_lexicon"],
            )
            results.append(entity)

        validated_count = sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
        print(f"  Validated genes: {validated_count}")
        print(f"  Time: {time.time() - start:.2f}s")

        return results

    def _create_gene_entity(
        self,
        candidate: "GeneCandidate",
        status_validated: bool,
        confidence: float,
        flags: List[str],
    ) -> "ExtractedGene":
        """Create ExtractedGene from a GeneCandidate."""
        from A_core.A01_domain_models import EvidenceSpan, ValidationStatus
        from A_core.A12_gene_models import ExtractedGene

        context = (candidate.context_text or "").strip()
        ctx_hash = hash_string(context) if context else "no_context"

        primary = EvidenceSpan(
            text=context,
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=len(context),
        )

        # Extract primary codes from identifiers
        hgnc_id = None
        entrez_id = None
        ensembl_id = None
        omim_id = None
        uniprot_id = None

        for ident in candidate.identifiers:
            if ident.system == "HGNC" and not hgnc_id:
                hgnc_id = ident.code
            elif ident.system == "ENTREZ" and not entrez_id:
                entrez_id = ident.code
            elif ident.system == "ENSEMBL" and not ensembl_id:
                ensembl_id = ident.code
            elif ident.system == "OMIM" and not omim_id:
                omim_id = ident.code
            elif ident.system == "UNIPROT" and not uniprot_id:
                uniprot_id = ident.code

        return ExtractedGene(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            matched_text=candidate.matched_text,
            hgnc_symbol=candidate.hgnc_symbol,
            full_name=candidate.full_name,
            is_alias=candidate.is_alias,
            alias_of=candidate.alias_of,
            identifiers=candidate.identifiers,
            hgnc_id=hgnc_id,
            entrez_id=entrez_id,
            ensembl_id=ensembl_id,
            omim_id=omim_id,
            uniprot_id=uniprot_id,
            primary_evidence=primary,
            supporting_evidence=[],
            status=ValidationStatus.VALIDATED if status_validated else ValidationStatus.REJECTED,
            confidence_score=min(1.0, confidence),
            validation_flags=flags,
            locus_type=candidate.locus_type,
            chromosome=candidate.chromosome,
            associated_diseases=candidate.associated_diseases,
            provenance=candidate.provenance,
        )

    # =========================================================================
    # DRUG PROCESSING
    # =========================================================================

    def process_drugs(self, doc: "DocumentGraph", pdf_path: Path) -> List["ExtractedDrug"]:
        """
        Process document for drug mentions.

        Returns validated drug entities.
        """
        if self.drug_detector is None:
            return []

        print("\n[6/12] Detecting drug mentions...")
        start = time.time()

        # Run drug detection
        candidates = self.drug_detector.detect(doc)
        print(f"  Drug candidates: {len(candidates)}")

        # Convert candidates to ExtractedDrug (auto-validated for lexicon matches)
        from A_core.A06_drug_models import DrugGeneratorType

        results: List["ExtractedDrug"] = []
        for candidate in candidates:
            # Determine if auto-validated (Alexion, investigational, FDA)
            is_specialized = candidate.generator_type in {
                DrugGeneratorType.LEXICON_ALEXION,
                DrugGeneratorType.LEXICON_INVESTIGATIONAL,
                DrugGeneratorType.PATTERN_COMPOUND_ID,
            }

            entity = self._candidate_to_extracted_drug(
                candidate,
                status_validated=True,
                confidence=candidate.initial_confidence if is_specialized else 0.7,
                flags=["auto_validated_lexicon"]
                if is_specialized
                else ["lexicon_match"],
            )
            results.append(entity)

        # PubTator enrichment
        if self.drug_enricher is not None:
            print("  Enriching with PubTator3...")
            results = self.drug_enricher.enrich_batch(results, verbose=True)

        validated_count = sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
        print(f"  Validated drugs: {validated_count}")
        print(f"  Time: {time.time() - start:.2f}s")

        return results

    def _candidate_to_extracted_drug(
        self,
        candidate: "DrugCandidate",
        status_validated: bool,
        confidence: float,
        flags: List[str],
    ) -> "ExtractedDrug":
        """Convert DrugCandidate to ExtractedDrug."""
        from A_core.A01_domain_models import EvidenceSpan, ValidationStatus
        from A_core.A06_drug_models import ExtractedDrug

        return ExtractedDrug(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            matched_text=candidate.matched_text,
            preferred_name=candidate.preferred_name,
            brand_name=candidate.brand_name,
            compound_id=candidate.compound_id,
            identifiers=candidate.identifiers,
            primary_evidence=EvidenceSpan(
                text=candidate.context_text,
                location=candidate.context_location,
                scope_ref="drug_detection",
                start_char_offset=0,
                end_char_offset=len(candidate.context_text),
            ),
            status=ValidationStatus.VALIDATED if status_validated else ValidationStatus.REJECTED,
            confidence_score=confidence,
            validation_flags=flags,
            drug_class=candidate.drug_class,
            mechanism=candidate.mechanism,
            development_phase=candidate.development_phase,
            is_investigational=candidate.is_investigational,
            sponsor=candidate.sponsor,
            conditions=candidate.conditions,
            nct_id=candidate.nct_id,
            dosage_form=candidate.dosage_form,
            route=candidate.route,
            marketing_status=candidate.marketing_status,
            provenance=candidate.provenance,
        )

    # =========================================================================
    # PHARMA PROCESSING
    # =========================================================================

    def process_pharma(self, doc: "DocumentGraph", pdf_path: Path) -> List["ExtractedPharma"]:
        """
        Process document for pharma company mentions.

        Returns validated pharma company entities.
        """
        if self.pharma_detector is None:
            return []

        print("\n[7/12] Detecting pharma company mentions...")
        start = time.time()

        # Build full text for detection
        full_text = " ".join(
            block.text
            for block in doc.iter_linear_blocks()
            if block.text
        )

        doc_fingerprint = hash_string(full_text[:5000])

        # Run pharma detection
        candidates = self.pharma_detector.detect(
            doc_graph=doc,
            doc_id=doc.doc_id,
            doc_fingerprint=doc_fingerprint,
            full_text=full_text,
        )
        print(f"  Pharma candidates: {len(candidates)}")

        # Validate candidates (auto-validated for lexicon matches)
        results = self.pharma_detector.validate_candidates(candidates)

        validated_count = sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
        print(f"  Validated pharma companies: {validated_count}")
        print(f"  Time: {time.time() - start:.2f}s")

        return results

    # =========================================================================
    # AUTHOR PROCESSING
    # =========================================================================

    def process_authors(
        self, doc: "DocumentGraph", pdf_path: Path, full_text: str
    ) -> List["ExtractedAuthor"]:
        """
        Process document for author/investigator mentions.

        Returns validated author entities.
        """
        if self.author_detector is None:
            return []

        print("\n[8/12] Detecting author/investigator mentions...")
        start = time.time()

        # Build document fingerprint
        doc_fingerprint = pdf_path.stem

        # Run author detection
        candidates = self.author_detector.detect(
            doc_graph=doc,
            doc_id=doc.doc_id if doc else pdf_path.stem,
            doc_fingerprint=doc_fingerprint,
            full_text=full_text,
        )
        print(f"  Author candidates found: {len(candidates)}")

        # Validate candidates
        results = self.author_detector.validate_candidates(candidates)

        validated_count = sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
        print(f"  Validated authors: {validated_count}")
        print(f"  Author detection took {time.time() - start:.2f}s")

        return results

    # =========================================================================
    # CITATION PROCESSING
    # =========================================================================

    def process_citations(
        self, doc: "DocumentGraph", pdf_path: Path, full_text: str
    ) -> List["ExtractedCitation"]:
        """
        Process document for citation/reference mentions.

        Returns validated citation entities.
        """
        if self.citation_detector is None:
            return []

        print("\n[9/12] Detecting citation/reference mentions...")
        start = time.time()

        # Build document fingerprint
        doc_fingerprint = pdf_path.stem

        # Run citation detection
        candidates = self.citation_detector.detect(
            doc_graph=doc,
            doc_id=doc.doc_id if doc else pdf_path.stem,
            doc_fingerprint=doc_fingerprint,
            full_text=full_text,
        )
        print(f"  Citation candidates found: {len(candidates)}")

        # Validate candidates
        results = self.citation_detector.validate_candidates(candidates)

        validated_count = sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
        print(f"  Validated citations: {validated_count}")
        print(f"  Citation detection took {time.time() - start:.2f}s")

        return results


__all__ = ["EntityProcessor"]
