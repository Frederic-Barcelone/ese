~/Projects/ese main* ⇡ ese ❯ /Users/frederictetard/Projects/ese/venv/bin/python /Users/frederictetard/Projects/ese/corpus_metadata/orchestrator.py                                                                                                                                           11:55:19
Starting pipeline... (loading NLP models, this may take a moment)
Loaded scispacy en_core_sci_lg with abbreviation detector + UMLS linker
    [INFO] Skipped 2 bad rare disease entries

Lexicons loaded: 21 sources, 616,586 entries
----------------------------------------------------------------------
  Abbreviation (122,637 entries)
    * Abbreviations                 5,392  2025_08_abbreviation_general.json
    * UMLS biological              97,308  2025_08_umls_biological_abbreviations_v5.tsv
    * UMLS clinical                19,937  2025_08_umls_clinical_abbreviations_v5.tsv
  Drug (142,239 entries)
    * Alexion drugs                    11  2025_08_alexion_drugs.json
    * Investigational drugs         7,067  2025_08_investigational_drugs.json
    * FDA approved drugs            3,200  2025_08_fda_approved_drugs.json
    * RxNorm terms                131,961  2025_08_lexicon_drug.json
    * scispacy NER                enabled  en_core_sci_lg
  Disease (40,792 entries)
    * Rare disease acronyms         1,631  2025_08_rare_disease_acronyms.json
    * (PAH)                             8  disease_lexicon_pah.json
    * (ANCA)                            5  disease_lexicon_anca.json
    * (IGAN)                            6  disease_lexicon_igan.json
    * (C3G)                             5  disease_lexicon_c3g.json
    * General diseases             29,669  2025_08_lexicon_disease.json
    * Orphanet diseases             9,468  2025_08_orphanet_diseases.json
    * scispacy NER                enabled  en_core_sci_lg
  Other (310,918 entries)
    * Trial acronyms              125,454  trial_acronyms_lexicon.json
    * PRO scales                      301  pro_scales_lexicon.json
    * Meta-Inventory               65,048  2025_meta_inventory_abbreviations.json
    * MONDO diseases               97,313  2025_mondo_diseases.json
    * ChEMBL drugs                 22,802  2025_chembl_drugs.json


Orchestrator v0.8 initialized
  Run ID: RUN_20260131_105519_e89fa32c888e
  Config: /Users/frederictetard/Projects/ese/corpus_metadata/G_config/config.yaml
  Model:  claude-sonnet-4-20250514
  Logs:   /Users/frederictetard/Projects/ese/corpus_log

  Extraction Pipeline Configuration:
  ----------------------------------------
  PRESET: standard

  EXTRACTORS:
    drugs                ON
    diseases             ON
    genes                ON
    abbreviations        ON
    feasibility          ON
    pharma_companies     OFF
    authors              OFF
    citations            OFF
    document_metadata    OFF
    tables               ON
    care_pathways        ON
    recommendations      ON
    visuals              ON

  OPTIONS:
    use_llm_validation   ON
    use_llm_feasibility  ON
    use_vlm_tables       ON
    use_normalization    ON
    use_epi_enricher     ON
    use_zeroshot_bioner  ON
    use_biomedical_ner   ON
    use_patient_journey  ON
    use_registry_extraction ON
    use_genetic_extraction ON
  ----------------------------------------

############################################################
BATCH PROCESSING: 1 PDFs
Folder: /Users/frederictetard/Projects/ese/Pdfs
############################################################

[1/1] 01_Article_Iptacopan C3G Trial.pdf

============================================================
Processing: 01_Article_Iptacopan C3G Trial.pdf
============================================================

[1/12] Parsing PDF...
Consider using the pymupdf_layout package for a greatly improved page layout analysis.
  Pages: 12
  Blocks: 246
  Tables: 3
  Time: 90.73s
  ⏱  90.7s
  Extracted text: 01_Article_Iptacopan C3G Trial_20260131_120112.txt

[2/12] Generating candidates...
  gen:syntax_pattern: 37 candidates
  gen:glossary_table: 0 candidates
  gen:rigid_pattern: 2 candidates
  gen:table_layout: 2 candidates
  gen:lexicon_match: 708 candidates
  Total unique: 741
  Time: 167.90s
  ⏱  167.9s
  LEXICON_MATCH reduction: 228 -> 117 (dedup: 49, form filter: 62)

[3/12] Validating candidates with Claude...
  Corroborated SFs: 46
  Frequent SFs (2+): 174
  Filtered (lexicon-only, rare): 572
  Auto-approved stats: 4
  Auto-approved country: 0
  Auto-rejected blacklist: 30
  Auto-rejected context: 0
  Auto-rejected trial IDs: 1
  Auto-rejected common words: 16
  Candidates for LLM: 114
  Batch (explicit pairs): 28
  Individual (lexicon): 86
  ⏱  151.6s
    Enriched trial 'APPEAR-C3G': A Multicenter, Randomized, Double-blind, Parallel Group, Pla...
  Hyphenated detected (PASO C): 5
  Direct search detected: 3

  Running LLM SF-only extractor (PASO D)...
    LLM chunks: 5, errors: 0, candidates: 16
  LLM extracted (PASO D): 1

[4/12] Normalizing, disambiguating & deduplicating...
  Normalized: 0
  NCT enriched: 1
  Disambiguated: 0
  Deduplicated: 147 duplicates merged
  ⏱  0.0s

[5/12] Detecting disease mentions...
  Disease candidates: 42
  Validated diseases: 42
  Time: 12.56s
  ⏱  12.6s

[5b/12] Detecting gene mentions...
  Gene candidates: 12
  Validated genes: 12
  Time: 0.02s
  ⏱  0.0s

[6/12] Detecting drug mentions...
  Drug candidates: 7
  Validated drugs: 7
  Time: 10.27s
  ⏱  10.3s

[Pharma detection] SKIPPED (disabled in config)

[Author detection] SKIPPED (disabled in config)

[Citation detection] SKIPPED (disabled in config)

[10/12] Extracting feasibility information...
  Using LLM-based extraction...
  Running EpiExtract4GARD-v2 enrichment...
Invalid model-index. Not loading eval results into CardData.
    EpiExtract4GARD: 1 epidemiology items
      Locations: 51
      Epi types: 5
      Statistics: 1
    EpiExtract time: 24.95s
  Running ZeroShotBioNER enrichment...
    ZeroShotBioNER time: 2.19s
  Running BiomedicalNER enrichment...
    BiomedicalNER: 22 entities extracted
      Clinical: 13
      Demographics: 0
      Temporal: 0
      Anatomical: 0
    BiomedicalNER time: 1.18s
  Running PatientJourneyNER enrichment...
      diagnostic_delay: 0
      treatment_line: 0
      care_pathway_step: 0
      surveillance_frequency: 4
      pain_point: 2
      recruitment_touchpoint: 2
    PatientJourney: 8 entities extracted
    PatientJourney time: 0.08s
  Running RegistryNER enrichment...
    RegistryNER: 0 entities extracted
    RegistryNER time: 0.59s
  Running GeneticNER enrichment...
      gene_symbols: 1
      variants_hgvs: 0
      variants_rsid: 0
      hpo_terms: 0
      disease_ordo: 0
    GeneticNER: 1 entities extracted
    GeneticNER time: 0.00s
  Span deduplication:
    Before: 35 -> After: 32
    Merged: 3 overlapping spans
    By source: {'EpiExtract4GARD-v2': 1, 'BiomedicalNER': 4, 'PatientJourneyNER': 5}
  Feasibility items: 32
  Time: 44.35s
  ⏱  44.3s

[10a/12] Extracting care pathways from flowchart figures...
  Extracted 0 care pathways
  ⏱  0.0s

[10b/12] Extracting guideline recommendations...
    + Text extraction: 3 recommendations
  Extracted 1 recommendation sets (3 recommendations)
  ⏱  12.2s

[10c/12] Extracting visuals (tables and figures)...