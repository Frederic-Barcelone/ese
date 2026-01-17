# corpus_metadata/C_generators/C15_strategy_burden.py
"""
Operational Burden Extractor - Extract site and patient burden factors.

Targets:
- Visit schedule (number, frequency, duration)
- Run-in period requirements
- Invasive procedures (biopsies, lumbar puncture, etc.)
- Vaccination requirements
- Special sample handling
- PRO burden (diaries, wearables)
- Background therapy constraints
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from A_core.A04_feasibility_models import (
    EvidenceSnippet,
    InvasiveProcedure,
    OperationalBurden,
    ProcedureType,
    SpecialSampleRequirement,
    VaccinationRequirement,
    VisitSchedule,
)
from B_parsing.B02_doc_graph import DocumentGraph


class OperationalBurdenExtractor:
    """
    Extract operational burden factors from clinical trial documents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Procedure patterns
        self.procedure_patterns = {
            ProcedureType.BIOPSY: [
                r"(?:renal|kidney|liver|skin|muscle|bone\s+marrow|tumor)\s+biops(?:y|ies)",
                r"biops(?:y|ies)\s+(?:of\s+)?(?:the\s+)?(?:renal|kidney|liver|skin|tumor)",
                r"tissue\s+biops(?:y|ies)",
            ],
            ProcedureType.LUMBAR_PUNCTURE: [
                r"lumbar\s+puncture",
                r"spinal\s+tap",
                r"CSF\s+(?:collection|sampling|analysis)",
                r"cerebrospinal\s+fluid\s+(?:collection|sampling)",
            ],
            ProcedureType.BONE_MARROW: [
                r"bone\s+marrow\s+(?:aspiration|biopsy|sampling)",
            ],
            ProcedureType.IMAGING_MRI: [
                r"MRI\s+(?:scan|imaging|examination)",
                r"magnetic\s+resonance\s+imaging",
            ],
            ProcedureType.IMAGING_CT: [
                r"CT\s+(?:scan|imaging|examination)",
                r"computed\s+tomography",
            ],
            ProcedureType.IMAGING_PET: [
                r"PET\s+(?:scan|imaging)",
                r"positron\s+emission\s+tomography",
            ],
            ProcedureType.ECHOCARDIOGRAM: [
                r"echocardiogra(?:m|phy)",
                r"cardiac\s+ultrasound",
            ],
            ProcedureType.ECG: [
                r"ECG",
                r"electrocardiogra(?:m|phy)",
                r"EKG",
            ],
            ProcedureType.ENDOSCOPY: [
                r"endoscop(?:y|ic)",
                r"colonoscop(?:y|ic)",
                r"gastroscop(?:y|ic)",
            ],
            ProcedureType.BRONCHOSCOPY: [
                r"bronchoscop(?:y|ic)",
                r"BAL",
                r"bronchoalveolar\s+lavage",
            ],
        }

        # Vaccination patterns
        self.vaccine_patterns = [
            (r"vaccin(?:e|ation)\s+(?:against|for)\s+(.+?)(?:\s+(?:at\s+least|prior\s+to)|\.|,|;)", "vaccine_for"),
            (r"(?:Neisseria\s+)?meningitidis\s+vaccin(?:e|ation)", "meningitis"),
            (r"(?:Streptococcus\s+)?pneumoni(?:ae|a)\s+vaccin(?:e|ation)", "pneumonia"),
            (r"pneumococcal\s+vaccin(?:e|ation)", "pneumonia"),
            (r"(?:Haemophilus\s+)?influenzae\s+(?:type\s+b\s+)?vaccin(?:e|ation)", "hib"),
            (r"meningococcal\s+vaccin(?:e|ation)", "meningitis"),
            (r"encapsulated\s+bacteria\s+vaccin(?:e|ation)", "encapsulated"),
            (r"hepatitis\s+[AB]\s+vaccin(?:e|ation)", "hepatitis"),
        ]

        # Visit schedule patterns
        self.visit_patterns = [
            # "visits at days 1, 14, 30, 60, 90"
            r"visits?\s+(?:at|on)\s+days?\s+([\d,\s]+(?:and\s+\d+)?)",
            # "day 1, 14, 30, 60, 90, and 180"
            r"(?:day|week)\s+([\d,\s]+(?:and\s+\d+)?)",
            # "X visits over Y weeks/months"
            r"(\d+)\s+visits?\s+over\s+(\d+)\s+(weeks?|months?)",
            # "visit schedule" table
            r"schedule\s+of\s+(?:assessments?|visits?|procedures?)",
        ]

        # Run-in patterns
        self.run_in_patterns = [
            r"run[- ]?in\s+(?:period|phase)\s+(?:of\s+)?(\d+)\s*(days?|weeks?|months?)",
            r"(\d+)[- ]?(day|week|month)\s+run[- ]?in",
            r"screening\s+period\s+(?:of\s+)?(?:up\s+to\s+)?(\d+)\s*(days?|weeks?|months?)",
        ]

        # PRO patterns
        self.pro_patterns = [
            r"patient[- ]?reported\s+outcome",
            r"PRO\s+(?:questionnaire|assessment|measure)",
            r"(?:daily|weekly)\s+diary",
            r"electronic\s+diary",
            r"ePRO",
            r"quality\s+of\s+life\s+(?:questionnaire|assessment)",
            r"QoL\s+(?:questionnaire|assessment)",
            r"SF-?36",
            r"EQ-?5D",
        ]

        # Sample handling patterns
        self.sample_patterns = [
            (r"24[- ]?(?:hour|h|hr)\s+urine", "24h urine"),
            (r"central\s+lab(?:oratory)?", "central_lab"),
            (r"biomarker\s+(?:sample|analysis|assessment)", "biomarker"),
            (r"pharmacokinetic\s+(?:sample|sampling|analysis)", "PK samples"),
            (r"PK\s+(?:sample|sampling)", "PK samples"),
            (r"genetic\s+(?:sample|testing|analysis)", "genetic"),
            (r"DNA\s+(?:sample|collection)", "genetic"),
        ]

    def extract(self, doc: DocumentGraph) -> OperationalBurden:
        """Extract operational burden data from document."""
        result = OperationalBurden()

        # Process document
        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = block.text
            if not text:
                continue

            page_num = block.page_num

            # Extract procedures
            for proc in self._extract_procedures(text, page_num):
                if not self._procedure_exists(proc, result.invasive_procedures):
                    result.invasive_procedures.append(proc)

            # Extract vaccinations
            for vax in self._extract_vaccinations(text, page_num):
                if not self._vaccination_exists(vax, result.vaccination_requirements):
                    result.vaccination_requirements.append(vax)

            # Extract visit schedule
            if not result.visit_schedule or not result.visit_schedule.visit_days:
                schedule = self._extract_visit_schedule(text, page_num)
                if schedule:
                    result.visit_schedule = schedule

            # Extract run-in period
            if not result.run_in_duration_days:
                run_in = self._extract_run_in(text)
                if run_in:
                    result.run_in_duration_days = run_in

            # Extract PROs
            for pro in self._extract_pros(text):
                if pro not in result.patient_reported_outcomes:
                    result.patient_reported_outcomes.append(pro)

            # Check for diary/wearable
            text_lower = text.lower()
            if any(kw in text_lower for kw in ["diary", "daily log", "patient log"]):
                result.diary_required = True
            if any(kw in text_lower for kw in ["wearable", "actigraph", "activity monitor", "smartwatch"]):
                result.wearable_required = True

            # Extract special samples
            for sample in self._extract_special_samples(text, page_num):
                if not self._sample_exists(sample, result.special_samples):
                    result.special_samples.append(sample)

        # Extract from tables (schedule of assessments)
        self._extract_from_tables(doc, result)

        return result

    def _extract_procedures(
        self,
        text: str,
        page_num: int
    ) -> List[InvasiveProcedure]:
        """Extract invasive procedures from text."""
        results = []

        for proc_type, patterns in self.procedure_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Extract timing if present
                    timing = self._extract_procedure_timing(text, match.end())

                    results.append(InvasiveProcedure(
                        procedure=match.group(0),
                        procedure_type=proc_type,
                        timing=timing,
                        evidence=[EvidenceSnippet(
                            text=text[max(0, match.start()-20):match.end()+50],
                            page=page_num
                        )]
                    ))
                    break  # Only one match per procedure type

        return results

    def _extract_procedure_timing(
        self,
        text: str,
        match_end: int
    ) -> List[str]:
        """Extract timing for a procedure."""
        timing = []

        # Look for timing patterns after the match
        context = text[match_end:match_end + 100]

        # "at screening", "at baseline", "at month 6"
        timing_match = re.search(
            r"(?:at|during)\s+(screening|baseline|day\s+\d+|week\s+\d+|month\s+\d+)",
            context,
            re.IGNORECASE
        )
        if timing_match:
            timing.append(timing_match.group(1))

        # Multiple timepoints: "at screening and month 6"
        multi_match = re.search(
            r"(?:at|during)\s+(screening|baseline)(?:\s+and\s+(?:at\s+)?)(month\s+\d+|week\s+\d+)",
            context,
            re.IGNORECASE
        )
        if multi_match:
            timing = [multi_match.group(1), multi_match.group(2)]

        return timing

    def _extract_vaccinations(
        self,
        text: str,
        page_num: int
    ) -> List[VaccinationRequirement]:
        """Extract vaccination requirements from text."""
        results = []

        for pattern, vax_type in self.vaccine_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Extract pathogen/vaccine name
                if vax_type == "vaccine_for":
                    pathogen = match.group(1).strip()
                    vaccine = f"Vaccine against {pathogen}"
                elif vax_type == "meningitis":
                    pathogen = "Neisseria meningitidis"
                    vaccine = "Meningococcal vaccine"
                elif vax_type == "pneumonia":
                    pathogen = "Streptococcus pneumoniae"
                    vaccine = "Pneumococcal vaccine"
                elif vax_type == "hib":
                    pathogen = "Haemophilus influenzae type b"
                    vaccine = "Hib vaccine"
                elif vax_type == "encapsulated":
                    pathogen = "Encapsulated bacteria"
                    vaccine = "Vaccines against encapsulated bacteria"
                else:
                    pathogen = None
                    vaccine = match.group(0)

                # Extract timing
                timing = None
                timing_match = re.search(
                    r"(?:at\s+least\s+)?(\d+)\s*(days?|weeks?)\s+(?:before|prior\s+to)",
                    text[match.end():match.end() + 80],
                    re.IGNORECASE
                )
                if timing_match:
                    timing = f"at least {timing_match.group(1)} {timing_match.group(2)} before first dose"

                results.append(VaccinationRequirement(
                    vaccine=vaccine,
                    pathogen=pathogen,
                    timing=timing,
                    evidence=[EvidenceSnippet(
                        text=text[max(0, match.start()-10):match.end()+50],
                        page=page_num
                    )]
                ))

        return results

    def _extract_visit_schedule(
        self,
        text: str,
        page_num: int
    ) -> Optional[VisitSchedule]:
        """Extract visit schedule from text."""
        # Look for visit days
        days_match = re.search(
            r"(?:visits?\s+(?:at|on)\s+)?days?\s+([\d,\s]+(?:and\s+\d+)?)",
            text,
            re.IGNORECASE
        )

        if days_match:
            days_str = days_match.group(1)
            # Parse day numbers
            days = re.findall(r"\d+", days_str)
            if days:
                return VisitSchedule(
                    visit_days=[int(d) for d in days],
                    total_visits=len(days),
                    evidence=[EvidenceSnippet(text=days_match.group(0), page=page_num)]
                )

        # Look for "X visits over Y weeks"
        duration_match = re.search(
            r"(\d+)\s+visits?\s+over\s+(\d+)\s+(weeks?|months?)",
            text,
            re.IGNORECASE
        )
        if duration_match:
            num_visits = int(duration_match.group(1))
            duration = int(duration_match.group(2))
            unit = duration_match.group(3)
            weeks = duration if "week" in unit else duration * 4

            return VisitSchedule(
                total_visits=num_visits,
                study_duration_weeks=weeks,
                evidence=[EvidenceSnippet(text=duration_match.group(0), page=page_num)]
            )

        return None

    def _extract_run_in(self, text: str) -> Optional[int]:
        """Extract run-in period duration in days."""
        for pattern in self.run_in_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                unit = match.group(2).lower()

                if "day" in unit:
                    return value
                elif "week" in unit:
                    return value * 7
                elif "month" in unit:
                    return value * 30

        return None

    def _extract_pros(self, text: str) -> List[str]:
        """Extract patient-reported outcomes."""
        results = []

        # Named PRO instruments
        pro_instruments = [
            ("SF-36", r"SF[- ]?36"),
            ("EQ-5D", r"EQ[- ]?5D"),
            ("FACIT-Fatigue", r"FACIT[- ]?(?:F|Fatigue)"),
            ("PROMIS", r"PROMIS"),
            ("PHQ-9", r"PHQ[- ]?9"),
            ("GAD-7", r"GAD[- ]?7"),
            ("HADS", r"HADS"),
            ("VAS", r"(?:visual\s+analog(?:ue)?\s+scale|VAS)"),
        ]

        for name, pattern in pro_instruments:
            if re.search(pattern, text, re.IGNORECASE):
                if name not in results:
                    results.append(name)

        # Generic PRO mentions
        if re.search(r"patient[- ]?reported\s+outcome", text, re.IGNORECASE):
            if "PRO questionnaire" not in results:
                results.append("PRO questionnaire")

        return results

    def _extract_special_samples(
        self,
        text: str,
        page_num: int
    ) -> List[SpecialSampleRequirement]:
        """Extract special sample requirements."""
        results = []

        for pattern, sample_type in self.sample_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if sample_type == "central_lab":
                    results.append(SpecialSampleRequirement(
                        sample_type="Blood/urine",
                        central_lab=True,
                        evidence=[EvidenceSnippet(text=match.group(0), page=page_num)]
                    ))
                else:
                    results.append(SpecialSampleRequirement(
                        sample_type=sample_type,
                        central_lab=False,
                        evidence=[EvidenceSnippet(text=match.group(0), page=page_num)]
                    ))

        return results

    def _extract_from_tables(
        self,
        doc: DocumentGraph,
        result: OperationalBurden
    ) -> None:
        """Extract from schedule of assessments tables."""
        for table in doc.iter_tables():
            # Check if this looks like a schedule table
            if not self._is_schedule_table(table):
                continue

            # Extract visit count from columns
            if table.headers:
                # Count visit columns (often named "Visit 1", "Day 1", "Week 1", etc.)
                visit_cols = [
                    h for h in table.headers
                    if re.search(r"(?:visit|day|week|month)\s*\d+", str(h), re.IGNORECASE)
                ]
                if visit_cols and (not result.visit_schedule or not result.visit_schedule.total_visits):
                    result.visit_schedule = VisitSchedule(
                        total_visits=len(visit_cols),
                        visit_windows=", ".join(str(v) for v in visit_cols),
                        evidence=[EvidenceSnippet(
                            text=f"Schedule table with {len(visit_cols)} visits",
                            page=table.page_num
                        )]
                    )

            # Look for procedures in table rows
            for row in table.logical_rows:
                if not row:
                    continue
                row_text = " ".join(str(cell) for cell in row).lower()

                # Check for biopsies
                if "biopsy" in row_text and "renal" in row_text:
                    if not any(p.procedure_type == ProcedureType.BIOPSY for p in result.invasive_procedures):
                        result.invasive_procedures.append(InvasiveProcedure(
                            procedure="Renal biopsy",
                            procedure_type=ProcedureType.BIOPSY,
                            evidence=[EvidenceSnippet(
                                text=" ".join(str(cell) for cell in row),
                                page=table.page_num
                            )]
                        ))

    def _is_schedule_table(self, table) -> bool:
        """Check if table is a schedule of assessments."""
        keywords = [
            "schedule", "assessment", "visit", "procedure",
            "day", "week", "month", "screening", "baseline"
        ]

        if table.headers:
            header_text = " ".join(str(h).lower() for h in table.headers)
            if any(kw in header_text for kw in keywords):
                return True

        return False

    def _procedure_exists(
        self,
        new_proc: InvasiveProcedure,
        existing: List[InvasiveProcedure]
    ) -> bool:
        """Check if procedure already exists."""
        for proc in existing:
            if proc.procedure_type == new_proc.procedure_type:
                return True
        return False

    def _vaccination_exists(
        self,
        new_vax: VaccinationRequirement,
        existing: List[VaccinationRequirement]
    ) -> bool:
        """Check if vaccination already exists."""
        for vax in existing:
            if vax.pathogen and new_vax.pathogen:
                if vax.pathogen.lower() == new_vax.pathogen.lower():
                    return True
            if vax.vaccine.lower() == new_vax.vaccine.lower():
                return True
        return False

    def _sample_exists(
        self,
        new_sample: SpecialSampleRequirement,
        existing: List[SpecialSampleRequirement]
    ) -> bool:
        """Check if sample requirement already exists."""
        for sample in existing:
            if sample.sample_type.lower() == new_sample.sample_type.lower():
                return True
        return False


def extract_operational_burden(doc: DocumentGraph) -> OperationalBurden:
    """Convenience function for operational burden extraction."""
    extractor = OperationalBurdenExtractor()
    return extractor.extract(doc)
