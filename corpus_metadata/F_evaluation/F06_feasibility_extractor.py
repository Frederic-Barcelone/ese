# corpus_metadata/F_evaluation/F06_feasibility_extractor.py
"""
Feasibility Profile Extractor - Orchestrates all feasibility extraction components.

This module combines:
- C13: Screening yield extraction
- C14: Eligibility criteria extraction
- C15: Operational burden extraction
- C16: Recruitment footprint extraction

And computes derived feasibility metrics.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from A_core.A04_feasibility_models import (
    FeasibilityMetrics,
    FeasibilityProfile,
    TrialIdentifiers,
)
from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
from B_parsing.B02_doc_graph import DocumentGraph
from C_generators.C13_strategy_screening import ScreeningYieldExtractor
from C_generators.C14_strategy_eligibility import EligibilityExtractor
from C_generators.C15_strategy_burden import OperationalBurdenExtractor
from C_generators.C16_strategy_footprint import RecruitmentFootprintExtractor


class FeasibilityExtractor:
    """
    Orchestrates feasibility extraction from clinical trial documents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize extractors
        self.screening_extractor = ScreeningYieldExtractor(config)
        self.eligibility_extractor = EligibilityExtractor(config)
        self.burden_extractor = OperationalBurdenExtractor(config)
        self.footprint_extractor = RecruitmentFootprintExtractor(config)

    def extract_from_pdf(self, pdf_path: str) -> FeasibilityProfile:
        """
        Extract feasibility profile from a PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Complete FeasibilityProfile with all extracted data
        """
        # Parse PDF to document graph
        print(f"[INFO] Parsing PDF: {pdf_path}")
        parser = PDFToDocGraphParser()
        doc = parser.parse(pdf_path)

        return self.extract_from_docgraph(doc, pdf_path)

    def extract_from_docgraph(
        self,
        doc: DocumentGraph,
        source_path: Optional[str] = None
    ) -> FeasibilityProfile:
        """
        Extract feasibility profile from a DocumentGraph.

        Args:
            doc: Parsed document graph
            source_path: Optional source file path

        Returns:
            Complete FeasibilityProfile with all extracted data
        """
        # Initialize profile
        profile = FeasibilityProfile(
            doc_id=doc.doc_id,
            extraction_timestamp=datetime.utcnow().isoformat(),
        )

        # Detect document type
        profile.doc_type = self._detect_doc_type(doc)

        # Extract trial identifiers
        print("[INFO] Extracting trial identifiers...")
        profile.trial_id = self._extract_trial_ids(doc)

        # Extract screening yield
        print("[INFO] Extracting screening yield...")
        profile.screening = self.screening_extractor.extract(doc)

        # Extract eligibility criteria
        print("[INFO] Extracting eligibility criteria...")
        profile.eligibility = self.eligibility_extractor.extract(doc)

        # Extract operational burden
        print("[INFO] Extracting operational burden...")
        profile.burden = self.burden_extractor.extract(doc)

        # Extract recruitment footprint
        print("[INFO] Extracting recruitment footprint...")
        profile.footprint = self.footprint_extractor.extract(doc)

        # Compute derived metrics
        print("[INFO] Computing feasibility metrics...")
        profile.compute_metrics()

        # Compute quality indicators
        profile.extraction_completeness = self._compute_completeness(profile)
        profile.fields_with_evidence = self._count_fields_with_evidence(profile)
        profile.total_fields_extracted = self._count_total_fields(profile)

        return profile

    def _detect_doc_type(self, doc: DocumentGraph) -> str:
        """Detect document type from content."""
        # Sample first few blocks
        text_sample = ""
        for i, block in enumerate(doc.iter_linear_blocks()):
            if i > 10:
                break
            text_sample += " " + (block.text or "")

        text_lower = text_sample.lower()

        if "protocol" in text_lower:
            return "protocol"
        elif "clinical study report" in text_lower or "csr" in text_lower:
            return "csr"
        elif "synopsis" in text_lower:
            return "synopsis"
        elif "trial" in text_lower or "randomized" in text_lower:
            return "article"
        else:
            return "unknown"

    def _extract_trial_ids(self, doc: DocumentGraph) -> TrialIdentifiers:
        """Extract trial registry identifiers."""
        from A_core.A04_feasibility_models import EvidenceSnippet

        ids = TrialIdentifiers()

        # NCT pattern
        nct_pattern = r"NCT\d{8}"
        # EudraCT pattern
        eudract_pattern = r"\d{4}-\d{6}-\d{2}"

        import re

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = block.text or ""

            # NCT ID
            if not ids.nct_id:
                match = re.search(nct_pattern, text)
                if match:
                    ids.nct_id = match.group(0)
                    ids.evidence.append(EvidenceSnippet(
                        text=match.group(0),
                        page=block.page_num
                    ))

            # EudraCT ID
            if not ids.eudract_id:
                match = re.search(eudract_pattern, text)
                if match:
                    ids.eudract_id = match.group(0)
                    ids.evidence.append(EvidenceSnippet(
                        text=match.group(0),
                        page=block.page_num
                    ))

            # Stop early if found both
            if ids.nct_id and ids.eudract_id:
                break

        return ids

    def _compute_completeness(self, profile: FeasibilityProfile) -> float:
        """Compute extraction completeness score."""
        fields_present = 0
        total_fields = 10  # Key fields we care about

        # Check key fields
        if profile.screening.screened:
            fields_present += 1
        if profile.screening.randomized:
            fields_present += 1
        if profile.eligibility.age:
            fields_present += 1
        if profile.eligibility.lab_thresholds:
            fields_present += 1
        if profile.eligibility.diagnosis_requirements:
            fields_present += 1
        if profile.burden.invasive_procedures:
            fields_present += 1
        if profile.burden.visit_schedule:
            fields_present += 1
        if profile.footprint.num_sites:
            fields_present += 1
        if profile.footprint.num_countries:
            fields_present += 1
        if profile.trial_id.nct_id:
            fields_present += 1

        return round(fields_present / total_fields, 2)

    def _count_fields_with_evidence(self, profile: FeasibilityProfile) -> int:
        """Count fields that have evidence attached."""
        count = 0

        if profile.screening.evidence:
            count += 1
        if profile.eligibility.evidence:
            count += 1
        if profile.burden.evidence:
            count += 1
        if profile.footprint.evidence:
            count += 1
        if profile.trial_id.evidence:
            count += 1

        # Count sub-fields with evidence
        for lab in profile.eligibility.lab_thresholds:
            if lab.evidence:
                count += 1
        for proc in profile.burden.invasive_procedures:
            if proc.evidence:
                count += 1
        for vax in profile.burden.vaccination_requirements:
            if vax.evidence:
                count += 1

        return count

    def _count_total_fields(self, profile: FeasibilityProfile) -> int:
        """Count total fields extracted (non-None)."""
        count = 0

        # Screening
        if profile.screening.screened:
            count += 1
        if profile.screening.randomized:
            count += 1
        if profile.screening.screen_failed:
            count += 1
        if profile.screening.completed:
            count += 1
        count += len(profile.screening.screen_fail_reasons)

        # Eligibility
        if profile.eligibility.age:
            count += 1
        count += len(profile.eligibility.lab_thresholds)
        count += len(profile.eligibility.diagnosis_requirements)
        count += len(profile.eligibility.medication_requirements)
        count += len(profile.eligibility.comorbidity_exclusions)

        # Burden
        if profile.burden.visit_schedule:
            count += 1
        if profile.burden.run_in_duration_days:
            count += 1
        count += len(profile.burden.invasive_procedures)
        count += len(profile.burden.vaccination_requirements)
        count += len(profile.burden.special_samples)
        count += len(profile.burden.patient_reported_outcomes)

        # Footprint
        if profile.footprint.num_sites:
            count += 1
        if profile.footprint.num_countries:
            count += 1
        count += len(profile.footprint.countries)

        return count

    def to_json(self, profile: FeasibilityProfile) -> str:
        """Convert profile to JSON string."""
        return profile.model_dump_json(indent=2, exclude_none=True)

    def to_dict(self, profile: FeasibilityProfile) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return profile.model_dump(exclude_none=True)


def extract_feasibility(pdf_path: str) -> FeasibilityProfile:
    """
    Convenience function to extract feasibility profile from PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        FeasibilityProfile with all extracted data
    """
    extractor = FeasibilityExtractor()
    return extractor.extract_from_pdf(pdf_path)


def print_feasibility_report(profile: FeasibilityProfile) -> None:
    """Print a formatted feasibility report."""
    print("\n" + "=" * 80)
    print("FEASIBILITY EXTRACTION REPORT")
    print("=" * 80)

    print(f"\nDocument: {profile.doc_id}")
    print(f"Type: {profile.doc_type}")
    print(f"Extracted: {profile.extraction_timestamp}")

    # Trial IDs
    print("\n" + "-" * 40)
    print("TRIAL IDENTIFIERS")
    print("-" * 40)
    if profile.trial_id.nct_id:
        print(f"  NCT ID: {profile.trial_id.nct_id}")
    if profile.trial_id.eudract_id:
        print(f"  EudraCT: {profile.trial_id.eudract_id}")

    # Footprint
    print("\n" + "-" * 40)
    print("RECRUITMENT FOOTPRINT")
    print("-" * 40)
    if profile.footprint.num_sites:
        print(f"  Sites: {profile.footprint.num_sites}")
    if profile.footprint.num_countries:
        print(f"  Countries: {profile.footprint.num_countries}")
    if profile.footprint.countries:
        print(f"  Country list: {', '.join(profile.footprint.countries[:10])}")

    # Screening
    print("\n" + "-" * 40)
    print("SCREENING YIELD")
    print("-" * 40)
    if profile.screening.screened:
        print(f"  Screened: {profile.screening.screened}")
    if profile.screening.randomized:
        print(f"  Randomized: {profile.screening.randomized}")
    if profile.screening.screening_yield_pct:
        print(f"  Yield: {profile.screening.screening_yield_pct}%")
    if profile.screening.screen_fail_reasons:
        print("  Screen fail reasons:")
        for reason in profile.screening.screen_fail_reasons[:5]:
            count_str = f" (n={reason.count})" if reason.count else ""
            print(f"    - {reason.reason}{count_str}")

    # Eligibility
    print("\n" + "-" * 40)
    print("ELIGIBILITY CRITERIA")
    print("-" * 40)
    if profile.eligibility.age:
        age = profile.eligibility.age
        age_str = f"{age.min_age or '?'}-{age.max_age or '?'} years"
        print(f"  Age: {age_str}")
    if profile.eligibility.lab_thresholds:
        print("  Lab thresholds:")
        for lab in profile.eligibility.lab_thresholds[:5]:
            print(f"    - {lab.analyte} {lab.operator.value} {lab.value} {lab.unit}")
    if profile.eligibility.diagnosis_requirements:
        print("  Diagnosis requirements:")
        for diag in profile.eligibility.diagnosis_requirements[:3]:
            method = f" ({diag.confirmation_method.value})" if diag.confirmation_method else ""
            print(f"    - {diag.condition}{method}")

    # Burden
    print("\n" + "-" * 40)
    print("OPERATIONAL BURDEN")
    print("-" * 40)
    if profile.burden.visit_schedule:
        vs = profile.burden.visit_schedule
        if vs.total_visits:
            print(f"  Total visits: {vs.total_visits}")
        if vs.study_duration_weeks:
            print(f"  Duration: {vs.study_duration_weeks} weeks")
    if profile.burden.run_in_duration_days:
        print(f"  Run-in period: {profile.burden.run_in_duration_days} days")
    if profile.burden.invasive_procedures:
        print("  Invasive procedures:")
        for proc in profile.burden.invasive_procedures:
            print(f"    - {proc.procedure}")
    if profile.burden.vaccination_requirements:
        print("  Vaccinations required:")
        for vax in profile.burden.vaccination_requirements:
            print(f"    - {vax.vaccine}")

    # Feasibility Metrics
    print("\n" + "-" * 40)
    print("FEASIBILITY METRICS")
    print("-" * 40)
    metrics = profile.feasibility_metrics
    if metrics.patient_burden_score is not None:
        print(f"  Patient burden score: {metrics.patient_burden_score:.1f}/10")
    if metrics.site_complexity_score is not None:
        print(f"  Site complexity score: {metrics.site_complexity_score:.1f}/10")
    print(f"  Biopsy required: {'Yes' if metrics.biopsy_required else 'No'}")
    print(f"  Vaccination required: {'Yes' if metrics.vaccination_required else 'No'}")
    if metrics.hard_gates:
        print("  Hard gates:")
        for gate in metrics.hard_gates[:5]:
            print(f"    - {gate}")

    # Quality
    print("\n" + "-" * 40)
    print("EXTRACTION QUALITY")
    print("-" * 40)
    print(f"  Completeness: {profile.extraction_completeness:.0%}")
    print(f"  Fields with evidence: {profile.fields_with_evidence}")
    print(f"  Total fields extracted: {profile.total_fields_extracted}")

    print("\n" + "=" * 80)


# -------------------------
# CLI Entry Point
# -------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract feasibility profile from PDF")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    # Extract
    profile = extract_feasibility(args.pdf_path)

    if args.json:
        output = FeasibilityExtractor().to_json(profile)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"[INFO] Written to {args.output}")
        else:
            print(output)
    else:
        print_feasibility_report(profile)
        if args.output:
            with open(args.output, "w") as f:
                f.write(FeasibilityExtractor().to_json(profile))
            print(f"\n[INFO] JSON written to {args.output}")
