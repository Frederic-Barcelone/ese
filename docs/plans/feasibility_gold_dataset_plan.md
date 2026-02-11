# Plan: Feasibility Gold Standard Dataset for Rare Diseases

## Goal
Create a 20-document gold standard dataset for evaluating the pipeline's feasibility extraction. Documents simulate real clinical trial feasibility assessments and treatment guidelines for rare diseases across different countries, with rich annotations covering patient funnel, local guidelines, investigator/site presence, and patient populations.

---

## Step 1: Design the 20 Documents

### Document Mix
- **10 clinical trial feasibility reports / study publications** (paper-style)
- **10 clinical practice guidelines / country-specific guideline summaries** (guideline-style)

### Rare Diseases (one per document)
Each document features a distinct rare disease with country-specific context:

| # | Disease | Country/Region | Style | Key Feasibility Focus |
|---|---------|---------------|-------|----------------------|
| 1 | Fabry Disease | Germany | Paper | Phase 3, enzyme replacement, renal biopsy gate |
| 2 | Huntington Disease | UK | Guideline | Treatment pathway, specialist centres, genetic testing |
| 3 | Pulmonary Arterial Hypertension | France | Paper | Phase 2/3, right heart catheterization, 6MWD endpoint |
| 4 | Wilson Disease | Japan | Guideline | Copper monitoring, chelation therapy, hepatologist network |
| 5 | Hereditary Angioedema | USA | Paper | Phase 3, prophylaxis trial, attack rate endpoint |
| 6 | Gaucher Disease | Brazil | Guideline | ERT access, regional centres, diagnostic delay |
| 7 | Cystic Fibrosis | Canada | Paper | Phase 2, CFTR modulator, sweat chloride endpoint |
| 8 | Myasthenia Gravis | South Korea | Paper | Phase 3, complement inhibitor, QMG score |
| 9 | Primary Biliary Cholangitis | Italy | Guideline | UDCA response, liver biopsy, specialist referral |
| 10 | Duchenne Muscular Dystrophy | Australia | Paper | Phase 2, gene therapy, ambulatory inclusion criterion |
| 11 | Sickle Cell Disease | Nigeria | Guideline | Hydroxyurea access, HbSS confirmation, rural screening |
| 12 | Systemic Mastocytosis | Spain | Paper | Phase 2, KIT mutation, bone marrow biopsy gate |
| 13 | Niemann-Pick Type C | Netherlands | Guideline | Miglustat pathway, filipin staining, expert centres |
| 14 | IgA Nephropathy | China | Paper | Phase 3, UPCR endpoint, renal biopsy requirement |
| 15 | Epidermolysis Bullosa | Argentina | Guideline | Wound care pathway, genetic confirmation, dermatology network |
| 16 | Phenylketonuria | Turkey | Paper | Phase 2, PAH enzyme, Phe level monitoring |
| 17 | Pompe Disease | India | Guideline | ERT access, GAA assay, newborn screening |
| 18 | Paroxysmal Nocturnal Hemoglobinuria | Sweden | Paper | Phase 3, complement C5 inhibitor, LDH endpoint |
| 19 | Tuberous Sclerosis Complex | Mexico | Guideline | mTOR inhibitor pathway, epilepsy monitoring, genetic panel |
| 20 | Hereditary Transthyretin Amyloidosis | Portugal | Paper | Phase 3, TTR stabilizer, neuropathy staging |

### Content Structure per Document

**Paper-style documents** (~2-3 pages each) will contain:
- Title, authors, abstract
- Introduction (disease background, epidemiology, unmet need)
- Methods (study design, eligibility criteria, endpoints, sites/countries, visit schedule)
- Results (patient funnel/CONSORT flow, screen failure reasons, demographics)
- Discussion (feasibility challenges, recruitment barriers)

**Guideline-style documents** (~2-3 pages each) will contain:
- Title, issuing body, date
- Disease overview (epidemiology, prevalence by region)
- Diagnostic pathway (required tests, specialist referral)
- Treatment recommendations (first-line, second-line, monitoring)
- Centre of expertise requirements (investigator qualifications, infrastructure)
- Patient registry/population data
- Access and screening considerations

---

## Step 2: Define Gold Annotation Schema

Align with `FeasibilityExportDocument` from A07 but structured for gold evaluation.

### Gold JSON Structure (per document)
```json
{
  "doc_id": "fabry_disease_germany.pdf",
  "doc_type": "paper",
  "disease": "Fabry Disease",
  "country": "Germany",

  "study_design": {
    "phase": "3",
    "design_type": "parallel",
    "blinding": "double-blind",
    "randomization_ratio": "1:1",
    "sample_size": 120,
    "actual_enrollment": 112,
    "duration_months": 18,
    "treatment_arms": [
      {"name": "Pegunigalsidase alfa", "n": 56, "dose": "1 mg/kg", "frequency": "every 2 weeks", "route": "IV"},
      {"name": "Agalsidase beta", "n": 56, "dose": "1 mg/kg", "frequency": "every 2 weeks", "route": "IV"}
    ],
    "control_type": "active_comparator",
    "setting": "multi-centre",
    "sites_total": 28,
    "countries_total": 12
  },

  "eligibility_inclusion": [
    {"text": "Age 18-65 years", "category": "age"},
    {"text": "Confirmed Fabry disease by alpha-galactosidase A enzyme assay", "category": "disease_definition"},
    {"text": "eGFR >= 40 mL/min/1.73m2", "category": "lab_value"},
    ...
  ],
  "eligibility_exclusion": [
    {"text": "Dialysis or renal transplant", "category": "organ_function"},
    ...
  ],

  "epidemiology": [
    {"data_type": "prevalence", "value": "1 in 40,000", "geography": "Germany", "source": "Fabry Registry"},
    {"data_type": "incidence", "value": "1 in 80,000 live births", "geography": "worldwide"}
  ],

  "screening_flow": {
    "screened": 198,
    "screen_failures": 86,
    "randomized": 112,
    "treated": 110,
    "completed": 101,
    "discontinued": 9,
    "screen_failure_rate": 43.4,
    "screen_fail_reasons": [
      {"reason": "eGFR below threshold", "count": 32, "percentage": 37.2},
      {"reason": "Withdrew consent", "count": 18, "percentage": 20.9},
      {"reason": "Prior renal transplant", "count": 14, "percentage": 16.3}
    ]
  },

  "operational_burden": {
    "invasive_procedures": [
      {"name": "renal biopsy", "timing": ["screening"], "purpose": "diagnosis_confirmation", "is_eligibility_requirement": true},
      {"name": "IV infusion", "timing": ["every 2 weeks"], "purpose": "treatment_administration"}
    ],
    "visit_schedule": {
      "total_visits": 22,
      "frequency": "every 2 weeks for infusion, every 3 months for assessment"
    },
    "central_lab_required": true,
    "central_lab_analytes": ["alpha-galactosidase A", "plasma Gb3", "lyso-Gb3", "eGFR", "proteinuria"],
    "background_therapy": [
      {"therapy_class": "ACEi/ARB", "requirement_type": "required_stable", "stable_duration_days": 90}
    ],
    "hard_gates": ["confirmed Fabry genotype", "renal biopsy", "eGFR threshold"]
  },

  "endpoints": [
    {"type": "primary", "name": "Annualized rate of change in eGFR", "timepoint": "18 months"},
    {"type": "secondary", "name": "Change in plasma lyso-Gb3", "timepoint": "18 months"},
    {"type": "safety", "name": "Infusion-associated reactions", "timepoint": "18 months"}
  ],

  "sites_and_investigators": {
    "total_sites": 28,
    "total_countries": 12,
    "countries": ["Germany", "France", "UK", "Italy", "Spain", "Netherlands", "USA", "Canada", "Brazil", "Japan", "Australia", "South Korea"],
    "investigator_requirements": "Board-certified nephrologist or metabolic disease specialist with >=3 years Fabry disease experience"
  },

  "local_guidelines": {
    "guideline_name": "German Society for Nephrology Fabry Disease Guidelines 2023",
    "key_recommendations": [
      "ERT initiation recommended when eGFR < 90 or proteinuria > 300 mg/day",
      "Genetic confirmation mandatory before ERT start",
      "Monitoring every 6 months with cardiac MRI annually"
    ],
    "impact_on_feasibility": "Guideline-mandated genetic testing aligns with trial eligibility; existing monitoring infrastructure supports trial visit schedule"
  },

  "patient_population": {
    "estimated_diagnosed_patients": 1200,
    "estimated_eligible_patients": 340,
    "registry_name": "Fabry Registry (Sanofi Genzyme)",
    "registry_size": 4800,
    "diagnostic_delay_years": 13.7,
    "referral_centres": 18,
    "geographic_distribution": "Concentrated in university hospitals in Berlin, Munich, Hamburg, Cologne"
  }
}
```

### Top-level Gold File
```json
{
  "corpus": "RareDis-Feasibility",
  "description": "20 synthetic documents for rare disease clinical trial feasibility evaluation",
  "version": "1.0",
  "documents": [ ...20 annotated document objects... ]
}
```

---

## Step 3: Create the Files

### 3a. Document Content (20 text files → PDFs)
- Location: `gold_data/feasibility/docs/`
- Format: Plain text with clear section structure (will be converted to simple PDFs)
- Each ~800-1500 words with realistic clinical content
- Specific numbers, criteria, and quotes that the pipeline can extract

### 3b. Gold Annotations
- Location: `gold_data/feasibility_gold.json`
- One JSON file containing all 20 document annotations
- Schema as defined in Step 2

### 3c. PDF Generation Script
- Location: `gold_data/feasibility/generate_feasibility_gold.py`
- Converts text files to single-column PDFs (like other gold generators)
- Outputs PDFs to `gold_data/feasibility/pdfs/`

---

## Step 4: Add Evaluation Support to F03

### 4a. Gold Loader
Add `load_feasibility_gold()` function to F03 that:
- Loads `feasibility_gold.json`
- Returns `dict[doc_id → FeasibilityGoldEntry]`

### 4b. Comparison Functions
Add feasibility-specific comparison logic:

| Field | Matching Strategy |
|-------|------------------|
| `study_design.phase` | Exact match |
| `study_design.sample_size` | Numeric tolerance ±5% |
| `study_design.blinding` | Normalized string match |
| `eligibility_inclusion` | Text fuzzy match (0.8) + category match |
| `eligibility_exclusion` | Text fuzzy match (0.8) + category match |
| `epidemiology` | data_type match + value presence |
| `screening_flow.screened/randomized/etc` | Numeric exact match |
| `screen_fail_reasons` | Fuzzy reason text + count match |
| `operational_burden.invasive_procedures` | Name fuzzy match |
| `endpoints` | Type + name fuzzy match |
| `sites_and_investigators.total_sites` | Numeric exact |
| `sites_and_investigators.countries` | Set overlap (Jaccard) |

### 4c. Scoring Metrics
- **Per-field F1**: For list fields (eligibility, endpoints, screen_fail_reasons)
- **Per-field accuracy**: For scalar fields (phase, sample_size, blinding)
- **Per-document composite score**: Weighted average across all fields
- **Corpus-level aggregates**: Macro-averaged F1 across all 20 docs

### 4d. Integration
- Add `RUN_FEASIBILITY = False` flag
- Add `FEASIBILITY_GOLD` and `FEASIBILITY_PATH` constants
- Add `DATASET_PRESETS["FEASIBILITY"]` with `feasibility: True`
- Wire into the main evaluation loop

---

## Step 5: Implementation Order

| Order | Task | Files | Est. Lines |
|-------|------|-------|-----------|
| 1 | Write 20 document text files | `gold_data/feasibility/docs/*.txt` | ~20K words |
| 2 | Write gold annotations JSON | `gold_data/feasibility_gold.json` | ~3K lines |
| 3 | Write PDF generation script | `gold_data/feasibility/generate_feasibility_gold.py` | ~120 lines |
| 4 | Generate PDFs | `gold_data/feasibility/pdfs/*.pdf` | (generated) |
| 5 | Add feasibility gold loader to F03 | `F03_evaluation_runner.py` | ~80 lines |
| 6 | Add feasibility comparison functions | `F03_evaluation_runner.py` | ~200 lines |
| 7 | Add feasibility scoring + integration | `F03_evaluation_runner.py` | ~100 lines |
| 8 | Run pipeline on 20 docs + evaluate | — | (verification) |

---

## Key Design Decisions

1. **Synthetic documents, not real papers**: Avoids copyright issues, allows precise control of annotation density, ensures all feasibility fields are present.

2. **Country diversity**: 20 documents × 20 different countries ensures evaluation covers geographic variation in guidelines, patient populations, and investigator networks.

3. **Both styles**: Guidelines test epidemiology/pathway extraction; papers test CONSORT flow/study design extraction. Real-world feasibility assessments draw from both.

4. **Rich patient funnel**: Every paper-style document includes full CONSORT flow with specific numbers and screen failure reasons — the hardest part of feasibility extraction.

5. **Local guidelines section**: Each document references country-specific treatment guidelines and their impact on trial feasibility — tests the pipeline's ability to extract guideline context.

6. **Hard gates**: Each document specifies concrete feasibility barriers (biopsy requirements, genetic testing, specialist access) that the operational_burden extractor should capture.

7. **Scoring by field type**: Scalar fields (phase, sample_size) use accuracy. List fields (eligibility, endpoints) use F1. This matches the heterogeneous nature of feasibility data.
