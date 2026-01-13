"""
FDA Therapeutic Areas and Disease Keywords Configuration - UPDATED VERSION
===========================================================================
Canonical names with ALIASES mapping for comprehensive search coverage.

KEY IMPROVEMENTS:
- Canonical names only in disease/drug lists (no duplicates)
- All synonyms, abbreviations, and variants mapped via ALIASES
- Corrected classifications (e.g., caplacizumab → anti-vWF agent)
- Better search coverage through systematic alias expansion

Used by: sync.py, labels.py, and other downloaders
"""

# ============================================================================
# THERAPEUTIC AREAS - CANONICAL NAMES ONLY
# ============================================================================

THERAPEUTIC_AREAS = {
    'nephrology': {
        'rare_diseases': [
            "IgA nephropathy",
            "focal segmental glomerulosclerosis",
            "membranous nephropathy",
            "minimal change disease",
            "lupus nephritis",
            "ANCA-associated vasculitis",
            "granulomatosis with polyangiitis",
            "microscopic polyangiitis",
            "atypical hemolytic uremic syndrome",
            "C3 glomerulopathy",
            "paroxysmal nocturnal hemoglobinuria",
            "Alport syndrome",
            "Fabry disease",
            "cystinosis",
            "primary hyperoxaluria",
            "Dent disease",
            "nephronophthisis",
            "Bartter syndrome",
            "Gitelman syndrome",
            "autosomal dominant polycystic kidney disease",
            "autosomal recessive polycystic kidney disease",
            "cystinuria",
            "renal tubular acidosis",
            "nephrotic syndrome",
            "steroid-resistant nephrotic syndrome",
            "congenital nephrotic syndrome",
            "proteinuria",
            "chronic kidney disease",
            "end-stage renal disease",
            "diabetic kidney disease",
            "anti-GBM disease",
            "Fanconi syndrome",
        ],
        'drug_classes': [
            "complement inhibitor",
            "C5 inhibitor",
            "C3 inhibitor",
            "factor D inhibitor",
            "endothelin receptor antagonist",
            "SGLT2 inhibitor",
            "mineralocorticoid receptor antagonist",
            "phosphate binder",
            "calcimimetic",
            "erythropoiesis-stimulating agent",
            "iron supplement",
            "vitamin D analog",
            "immunosuppressant (kidney)",
            "calcineurin inhibitor",
        ],
    },

    'hematology': {
        'rare_diseases': [
            "hemophilia A",
            "hemophilia B",
            "von Willebrand disease",
            "factor VII deficiency",
            "factor X deficiency",
            "factor XI deficiency",
            "factor XIII deficiency",
            "factor V deficiency",
            "factor II deficiency",
            "Glanzmann thrombasthenia",
            "Bernard-Soulier syndrome",
            "sickle cell disease",
            "beta thalassemia",
            "alpha thalassemia",
            "hemoglobin H disease",
            "paroxysmal nocturnal hemoglobinuria",
            "aplastic anemia",
            "pure red cell aplasia",
            "Diamond-Blackfan anemia",
            "Fanconi anemia",
            "congenital dyserythropoietic anemia",
            "pyruvate kinase deficiency",
            "glucose-6-phosphate dehydrogenase deficiency",
            "hereditary spherocytosis",
            "hereditary elliptocytosis",
            "myelodysplastic syndromes",
            "primary myelofibrosis",
            "polycythemia vera",
            "essential thrombocythemia",
            "chronic myelomonocytic leukemia",
            "thrombotic thrombocytopenic purpura",
            "heparin-induced thrombocytopenia",
            "immune thrombocytopenia",
            "hemolytic uremic syndrome",
            "antiphospholipid syndrome",
            "factor V Leiden",
            "protein C deficiency",
            "protein S deficiency",
            "antithrombin deficiency",
            "thrombophilia",
            "acute myeloid leukemia",
            "acute lymphoblastic leukemia",
            "chronic myeloid leukemia",
            "chronic lymphocytic leukemia",
            "multiple myeloma",
            "Hodgkin lymphoma",
            "non-Hodgkin lymphoma",
            "diffuse large B-cell lymphoma",
            "follicular lymphoma",
            "mantle cell lymphoma",
            "marginal zone lymphoma",
            "Burkitt lymphoma",
            "T-cell lymphoma",
            "cutaneous T-cell lymphoma",
            "mycosis fungoides",
            "Sezary syndrome",
            "Waldenström macroglobulinemia",
            "hairy cell leukemia",
            "hereditary hemochromatosis",
            "iron overload",
            "transfusional iron overload",
            "porphyria",
            "acute intermittent porphyria",
            "porphyria cutanea tarda",
            "systemic mastocytosis",
            "hypereosinophilic syndrome",
            "cold agglutinin disease",
            "warm autoimmune hemolytic anemia",
            "autoimmune hemolytic anemia",
            "severe chronic neutropenia",
            "congenital neutropenia",
            "cyclic neutropenia",
        ],
        'drug_classes': [
            "factor replacement therapy",
            "gene therapy",
            "bispecific antibody (hemophilia)",
            "monoclonal antibody",
            "complement inhibitor",
            "C5 inhibitor",
            "thrombopoietin receptor agonist",
            "erythropoiesis-stimulating agent",
            "iron chelator",
            "BCL-2 inhibitor",
            "BTK inhibitor",
            "JAK inhibitor",
            "FLT3 inhibitor",
            "IDH inhibitor",
            "anti-CD20 antibody",
            "anti-CD38 antibody",
            "proteasome inhibitor",
            "IMiD",
            "hypomethylating agent",
            "anti-vWF agent",
        ],
    }
}

# ============================================================================
# ALIASES - MAPS SYNONYMS/ABBREVIATIONS TO CANONICAL NAMES
# ============================================================================

ALIASES = {
    # Nephrology — diseases
    "FSGS": "focal segmental glomerulosclerosis",
    "Wegener granulomatosis": "granulomatosis with polyangiitis",
    "aHUS": "atypical hemolytic uremic syndrome",
    "C3 glomerulonephritis": "C3 glomerulopathy",
    "dense deposit disease": "C3 glomerulopathy",
    "PNH": "paroxysmal nocturnal hemoglobinuria",
    "hyperoxaluria type 1": "primary hyperoxaluria",
    "hyperoxaluria type 2": "primary hyperoxaluria",
    "ADPKD": "autosomal dominant polycystic kidney disease",
    "ARPKD": "autosomal recessive polycystic kidney disease",
    "CKD": "chronic kidney disease",
    "ESRD": "end-stage renal disease",
    "end stage renal disease": "end-stage renal disease",
    "diabetic nephropathy": "diabetic kidney disease",
    "Goodpasture syndrome": "anti-GBM disease",
    "renal Fanconi syndrome": "Fanconi syndrome",
    "ANCA vasculitis": "ANCA-associated vasculitis",
    "nephrotic syndrome steroid resistant": "steroid-resistant nephrotic syndrome",

    # Nephrology — drug classes
    "ERA": "endothelin receptor antagonist",
    "sodium glucose cotransporter 2 inhibitor": "SGLT2 inhibitor",
    "MRA": "mineralocorticoid receptor antagonist",
    "erythropoiesis stimulating agent": "erythropoiesis-stimulating agent",
    "ESA": "erythropoiesis-stimulating agent",
    "erythropoietin": "erythropoiesis-stimulating agent",
    "immunosuppressant kidney": "immunosuppressant (kidney)",

    # Hematology — diseases
    "Christmas disease": "hemophilia B",
    "VWD": "von Willebrand disease",
    "hemophilia C": "factor XI deficiency",
    "prothrombin deficiency": "factor II deficiency",
    "SCD": "sickle cell disease",
    "sickle cell anemia": "sickle cell disease",
    "thalassemia major": "beta thalassemia",
    "thalassemia intermedia": "beta thalassemia",
    "HbH disease": "hemoglobin H disease",
    "severe aplastic anemia": "aplastic anemia",
    "PRCA": "pure red cell aplasia",
    "DBA": "Diamond-Blackfan anemia",
    "CDA": "congenital dyserythropoietic anemia",
    "PK deficiency": "pyruvate kinase deficiency",
    "G6PD deficiency": "glucose-6-phosphate dehydrogenase deficiency",
    "MDS": "myelodysplastic syndromes",
    "myelodysplastic syndrome": "myelodysplastic syndromes",
    "myelofibrosis": "primary myelofibrosis",
    "PMF": "primary myelofibrosis",
    "PV": "polycythemia vera",
    "ET": "essential thrombocythemia",
    "CMML": "chronic myelomonocytic leukemia",
    "TTP": "thrombotic thrombocytopenic purpura",
    "acquired TTP": "thrombotic thrombocytopenic purpura",
    "congenital TTP": "thrombotic thrombocytopenic purpura",
    "HIT": "heparin-induced thrombocytopenia",
    "ITP": "immune thrombocytopenia",
    "immune thrombocytopenic purpura": "immune thrombocytopenia",
    "HUS": "hemolytic uremic syndrome",
    "Shiga toxin HUS": "hemolytic uremic syndrome",
    "APS": "antiphospholipid syndrome",
    "AML": "acute myeloid leukemia",
    "ALL": "acute lymphoblastic leukemia",
    "acute lymphocytic leukemia": "acute lymphoblastic leukemia",
    "CML": "chronic myeloid leukemia",
    "CLL": "chronic lymphocytic leukemia",
    "plasma cell myeloma": "multiple myeloma",
    "NHL": "non-Hodgkin lymphoma",
    "DLBCL": "diffuse large B-cell lymphoma",
    "MCL": "mantle cell lymphoma",
    "CTCL": "cutaneous T-cell lymphoma",
    "WM": "Waldenström macroglobulinemia",
    "mastocytosis": "systemic mastocytosis",
    "eosinophilic disorders": "hypereosinophilic syndrome",
    "HES": "hypereosinophilic syndrome",
    "CAD": "cold agglutinin disease",
    "WAIHA": "warm autoimmune hemolytic anemia",
    "AIHA": "autoimmune hemolytic anemia",
    "Kostmann syndrome": "congenital neutropenia",
    "hemochromatosis": "hereditary hemochromatosis",

    # Hematology — drug classes
    "factor VIII": "factor replacement therapy",
    "factor IX": "factor replacement therapy",
    "factor VII": "factor replacement therapy",
    "recombinant factor": "factor replacement therapy",
    "factor replacement": "factor replacement therapy",
    "gene therapy hemophilia": "gene therapy",
    "bispecific antibody": "bispecific antibody (hemophilia)",
    "bispecific antibody hemophilia": "bispecific antibody (hemophilia)",
    "emicizumab": "bispecific antibody (hemophilia)",
    "C5 inhibitor hematology": "C5 inhibitor",
    "TPO receptor agonist": "thrombopoietin receptor agonist",
    "romiplostim": "thrombopoietin receptor agonist",
    "eltrombopag": "thrombopoietin receptor agonist",
    "ESA hematology": "erythropoiesis-stimulating agent",
    "deferasirox": "iron chelator",
    "deferiprone": "iron chelator",
    "deferoxamine": "iron chelator",
    "BCL2 inhibitor": "BCL-2 inhibitor",
    "venetoclax": "BCL-2 inhibitor",
    "Bruton tyrosine kinase inhibitor": "BTK inhibitor",
    "ibrutinib": "BTK inhibitor",
    "acalabrutinib": "BTK inhibitor",
    "Janus kinase inhibitor": "JAK inhibitor",
    "ruxolitinib": "JAK inhibitor",
    "IDH1 inhibitor": "IDH inhibitor",
    "IDH2 inhibitor": "IDH inhibitor",
    "CD20 antibody": "anti-CD20 antibody",
    "rituximab": "anti-CD20 antibody",
    "CD38 antibody": "anti-CD38 antibody",
    "daratumumab": "anti-CD38 antibody",
    "bortezomib": "proteasome inhibitor",
    "carfilzomib": "proteasome inhibitor",
    "immunomodulatory drug": "IMiD",
    "lenalidomide": "IMiD",
    "pomalidomide": "IMiD",
    "azacitidine": "hypomethylating agent",
    "decitabine": "hypomethylating agent",
    "anti-ADAMTS13 antibody": "anti-vWF agent",
    "caplacizumab": "anti-vWF agent",
}

# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def normalize_term(term):
    """Map any synonym/abbreviation to the canonical label."""
    return ALIASES.get(term, term)


def get_all_therapeutic_areas():
    """Get list of all therapeutic area names"""
    return list(THERAPEUTIC_AREAS.keys())


def get_disease_count(therapeutic_area):
    """Get count of diseases for a therapeutic area"""
    if therapeutic_area in THERAPEUTIC_AREAS:
        return len(THERAPEUTIC_AREAS[therapeutic_area]['rare_diseases'])
    return 0


def get_drug_class_count(therapeutic_area):
    """Get count of drug classes for a therapeutic area"""
    if therapeutic_area in THERAPEUTIC_AREAS:
        return len(THERAPEUTIC_AREAS[therapeutic_area]['drug_classes'])
    return 0


def get_expanded_keywords(therapeutic_area):
    """
    Get all keywords including aliases for a therapeutic area.
    This expands canonical terms with their aliases for comprehensive searching.
    """
    if therapeutic_area not in THERAPEUTIC_AREAS:
        return []
    
    diseases = THERAPEUTIC_AREAS[therapeutic_area]['rare_diseases']
    drug_classes = THERAPEUTIC_AREAS[therapeutic_area]['drug_classes']
    canonical_terms = diseases + drug_classes
    
    # Create reverse mapping: canonical -> list of aliases
    reverse_aliases = {}
    for alias, canonical in ALIASES.items():
        if canonical in canonical_terms:
            if canonical not in reverse_aliases:
                reverse_aliases[canonical] = []
            reverse_aliases[canonical].append(alias)
    
    # Build expanded list: canonical + all its aliases
    expanded = []
    for term in canonical_terms:
        expanded.append(term)  # Add canonical
        if term in reverse_aliases:
            expanded.extend(reverse_aliases[term])  # Add all aliases
    
    return expanded


def print_configuration_summary():
    """Print summary of therapeutic areas configuration"""
    print("\n" + "="*70)
    print("THERAPEUTIC AREAS CONFIGURATION - UPDATED")
    print("="*70)
    
    for area in get_all_therapeutic_areas():
        disease_count = get_disease_count(area)
        class_count = get_drug_class_count(area)
        canonical_count = disease_count + class_count
        
        # Count aliases for this area
        diseases = THERAPEUTIC_AREAS[area]['rare_diseases']
        drug_classes = THERAPEUTIC_AREAS[area]['drug_classes']
        all_canonical = set(diseases + drug_classes)
        alias_count = sum(1 for v in ALIASES.values() if v in all_canonical)
        
        print(f"\n{area.upper()}:")
        print(f"  Canonical Diseases: {disease_count}")
        print(f"  Canonical Drug Classes: {class_count}")
        print(f"  Total Canonical: {canonical_count}")
        print(f"  Aliases: {alias_count}")
        print(f"  Total Keywords (with aliases): {canonical_count + alias_count}")
    
    total_diseases = sum(get_disease_count(a) for a in get_all_therapeutic_areas())
    total_classes = sum(get_drug_class_count(a) for a in get_all_therapeutic_areas())
    total_aliases = len(ALIASES)
    
    print("\nTOTAL:")
    print(f"  Therapeutic Areas: {len(get_all_therapeutic_areas())}")
    print(f"  Canonical Diseases: {total_diseases}")
    print(f"  Canonical Drug Classes: {total_classes}")
    print(f"  Total Canonical: {total_diseases + total_classes}")
    print(f"  Total Aliases: {total_aliases}")
    print(f"  Total Keywords (with aliases): {total_diseases + total_classes + total_aliases}")
    print("="*70 + "\n")


# ============================================================================
# MAIN - For testing this file directly
# ============================================================================

if __name__ == "__main__":
    print_configuration_summary()
    
    # Example: Show canonical vs expanded
    print("\nEXAMPLE: Nephrology Keywords")
    print("-" * 70)
    print("Canonical only:", len(THERAPEUTIC_AREAS['nephrology']['rare_diseases'] + 
                                   THERAPEUTIC_AREAS['nephrology']['drug_classes']))
    print("With aliases:", len(get_expanded_keywords('nephrology')))
    
    # Show some examples of aliases
    print("\nExample Aliases:")
    examples = [
        ("FSGS", normalize_term("FSGS")),
        ("aHUS", normalize_term("aHUS")),
        ("ESA", normalize_term("ESA")),
        ("AML", normalize_term("AML")),
        ("SCD", normalize_term("SCD")),
    ]
    for alias, canonical in examples:
        print(f"  '{alias}' → '{canonical}'")