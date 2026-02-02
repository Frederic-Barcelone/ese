# corpus_metadata/K_tests/K57_test_clinical_criteria.py
"""
Tests for A_core.A21_clinical_criteria module.

Tests clinical criteria models, lab criterion evaluation, and severity grades.
"""

from __future__ import annotations


from A_core.A21_clinical_criteria import (
    EntityNormalization,
    LabTimepoint,
    LabCriterion,
    DiagnosisConfirmation,
    SeverityGradeType,
    SeverityGrade,
    SEVERITY_GRADE_MAPPINGS,
)


class TestEntityNormalization:
    """Tests for EntityNormalization class."""

    def test_create_minimal(self):
        norm = EntityNormalization(system="LOINC", code="2160-0")
        assert norm.system == "LOINC"
        assert norm.code == "2160-0"
        assert norm.label is None

    def test_create_with_label(self):
        norm = EntityNormalization(
            system="RxNorm",
            code="1049502",
            label="metformin 500 MG Oral Tablet",
        )
        assert norm.label == "metformin 500 MG Oral Tablet"

    def test_various_systems(self):
        systems = ["LOINC", "RxNorm", "ATC", "Orphanet", "SNOMED", "ICD-10"]
        for sys in systems:
            norm = EntityNormalization(system=sys, code="TEST")
            assert norm.system == sys


class TestLabTimepoint:
    """Tests for LabTimepoint class."""

    def test_create_minimal(self):
        tp = LabTimepoint()
        assert tp.day is None
        assert tp.visit_name is None
        assert tp.window_days is None

    def test_create_with_day(self):
        tp = LabTimepoint(day=-14, window_days=3)
        assert tp.day == -14
        assert tp.window_days == 3

    def test_create_with_visit(self):
        tp = LabTimepoint(visit_name="screening", window_days=7)
        assert tp.visit_name == "screening"


class TestLabCriterion:
    """Tests for LabCriterion class."""

    def test_create_minimal(self):
        crit = LabCriterion(
            analyte="eGFR",
            operator=">=",
            value=30.0,
            unit="mL/min/1.73m2",
        )
        assert crit.analyte == "eGFR"
        assert crit.operator == ">="

    def test_create_full(self):
        crit = LabCriterion(
            analyte="UPCR",
            operator=">=",
            value=1.0,
            unit="g/g",
            specimen="first_morning_void",
            timepoints=[LabTimepoint(visit_name="screening")],
            normalization=EntityNormalization(system="LOINC", code="9318-7"),
        )
        assert crit.specimen == "first_morning_void"
        assert len(crit.timepoints) == 1

    def test_create_with_range(self):
        crit = LabCriterion(
            analyte="eGFR",
            operator="between",
            value=0,  # Not used for between
            min_value=30.0,
            max_value=89.0,
            unit="mL/min/1.73m2",
        )
        assert crit.min_value == 30.0
        assert crit.max_value == 89.0


class TestLabCriterionEvaluate:
    """Tests for LabCriterion.evaluate method."""

    def test_evaluate_gte(self):
        crit = LabCriterion(analyte="eGFR", operator=">=", value=30.0, unit="mL/min")
        assert crit.evaluate(30.0) is True
        assert crit.evaluate(45.0) is True
        assert crit.evaluate(29.9) is False

    def test_evaluate_lte(self):
        crit = LabCriterion(analyte="UPCR", operator="<=", value=3.5, unit="g/g")
        assert crit.evaluate(3.5) is True
        assert crit.evaluate(2.0) is True
        assert crit.evaluate(4.0) is False

    def test_evaluate_gt(self):
        crit = LabCriterion(analyte="C3", operator=">", value=0.9, unit="g/L")
        assert crit.evaluate(1.0) is True
        assert crit.evaluate(0.9) is False
        assert crit.evaluate(0.5) is False

    def test_evaluate_lt(self):
        crit = LabCriterion(analyte="albumin", operator="<", value=3.0, unit="g/dL")
        assert crit.evaluate(2.5) is True
        assert crit.evaluate(3.0) is False
        assert crit.evaluate(3.5) is False

    def test_evaluate_eq(self):
        crit = LabCriterion(analyte="stage", operator="==", value=3.0, unit="")
        assert crit.evaluate(3.0) is True
        assert crit.evaluate(2.0) is False

    def test_evaluate_between(self):
        crit = LabCriterion(
            analyte="eGFR",
            operator="between",
            value=0,
            min_value=30.0,
            max_value=89.0,
            unit="mL/min",
        )
        assert crit.evaluate(30.0) is True
        assert crit.evaluate(60.0) is True
        assert crit.evaluate(89.0) is True
        assert crit.evaluate(29.9) is False
        assert crit.evaluate(90.0) is False

    def test_evaluate_unknown_operator(self):
        crit = LabCriterion(analyte="test", operator="!=", value=1.0, unit="")
        assert crit.evaluate(2.0) is False  # Unknown operator returns False


class TestDiagnosisConfirmation:
    """Tests for DiagnosisConfirmation class."""

    def test_create_minimal(self):
        diag = DiagnosisConfirmation(method="biopsy")
        assert diag.method == "biopsy"

    def test_create_full(self):
        diag = DiagnosisConfirmation(
            method="biopsy",
            window_months=12,
            assessor="central review",
            findings="mesangial IgA deposits",
        )
        assert diag.window_months == 12
        assert diag.assessor == "central review"
        assert diag.findings == "mesangial IgA deposits"

    def test_various_methods(self):
        methods = ["biopsy", "genetic_testing", "clinical_criteria"]
        for method in methods:
            diag = DiagnosisConfirmation(method=method)
            assert diag.method == method


class TestSeverityGradeType:
    """Tests for SeverityGradeType enum."""

    def test_all_values(self):
        assert SeverityGradeType.NYHA.value == "nyha"
        assert SeverityGradeType.ECOG.value == "ecog"
        assert SeverityGradeType.CKD.value == "ckd"
        assert SeverityGradeType.CHILD_PUGH.value == "child_pugh"
        assert SeverityGradeType.MELD.value == "meld"
        assert SeverityGradeType.BCLC.value == "bclc"
        assert SeverityGradeType.TNM.value == "tnm"
        assert SeverityGradeType.EDSS.value == "edss"


class TestSeverityGrade:
    """Tests for SeverityGrade class."""

    def test_create_with_numeric_value(self):
        grade = SeverityGrade(
            grade_type=SeverityGradeType.NYHA,
            raw_value="Class II",
            numeric_value=2,
            operator="<=",
        )
        assert grade.numeric_value == 2
        assert grade.operator == "<="

    def test_create_with_range(self):
        grade = SeverityGrade(
            grade_type=SeverityGradeType.ECOG,
            raw_value="ECOG 0-1",
            min_value=0,
            max_value=1,
            operator="between",
        )
        assert grade.min_value == 0
        assert grade.max_value == 1


class TestSeverityGradeEvaluate:
    """Tests for SeverityGrade.evaluate method."""

    def test_evaluate_lte(self):
        grade = SeverityGrade(
            grade_type=SeverityGradeType.NYHA,
            raw_value="NYHA <= II",
            numeric_value=2,
            operator="<=",
        )
        assert grade.evaluate(1) is True
        assert grade.evaluate(2) is True
        assert grade.evaluate(3) is False

    def test_evaluate_gte(self):
        grade = SeverityGrade(
            grade_type=SeverityGradeType.CKD,
            raw_value="CKD >= 3",
            numeric_value=3,
            operator=">=",
        )
        assert grade.evaluate(3) is True
        assert grade.evaluate(4) is True
        assert grade.evaluate(2) is False

    def test_evaluate_eq(self):
        grade = SeverityGrade(
            grade_type=SeverityGradeType.ECOG,
            raw_value="ECOG 0",
            numeric_value=0,
            operator="==",
        )
        assert grade.evaluate(0) is True
        assert grade.evaluate(1) is False

    def test_evaluate_between(self):
        grade = SeverityGrade(
            grade_type=SeverityGradeType.ECOG,
            raw_value="ECOG 0-1",
            min_value=0,
            max_value=1,
            operator="between",
        )
        assert grade.evaluate(0) is True
        assert grade.evaluate(1) is True
        assert grade.evaluate(2) is False

    def test_evaluate_no_operator(self):
        grade = SeverityGrade(
            grade_type=SeverityGradeType.NYHA,
            raw_value="NYHA II",
            numeric_value=2,
        )
        assert grade.evaluate(2) is False  # No operator, returns False


class TestSeverityGradeMappings:
    """Tests for SEVERITY_GRADE_MAPPINGS dictionary."""

    def test_nyha_mappings(self):
        mappings = SEVERITY_GRADE_MAPPINGS[SeverityGradeType.NYHA]
        assert mappings["i"] == 1
        assert mappings["class ii"] == 2
        assert mappings["iii"] == 3
        assert mappings["4"] == 4

    def test_ecog_mappings(self):
        mappings = SEVERITY_GRADE_MAPPINGS[SeverityGradeType.ECOG]
        assert mappings["0"] == 0
        assert mappings["3"] == 3
        assert mappings["5"] == 5

    def test_ckd_mappings(self):
        mappings = SEVERITY_GRADE_MAPPINGS[SeverityGradeType.CKD]
        assert mappings["stage 3a"] == 3
        assert mappings["stage 3b"] == 3
        assert mappings["5"] == 5

    def test_child_pugh_mappings(self):
        mappings = SEVERITY_GRADE_MAPPINGS[SeverityGradeType.CHILD_PUGH]
        assert mappings["a"] == 1
        assert mappings["class b"] == 2
        assert mappings["c"] == 3
        # Score-based mappings
        assert mappings["5"] == 1  # A
        assert mappings["9"] == 2  # B
        assert mappings["15"] == 3  # C
