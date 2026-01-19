# corpus_metadata/G_config/__init__.py
"""
Configuration module for corpus_metadata extraction pipeline.

All extraction settings are in config.yaml under extraction_pipeline:

    extraction_pipeline:
      extractors:
        drugs: true
        diseases: true
        abbreviations: true
        feasibility: true
        pharma_companies: false
        authors: false
        citations: false
        document_metadata: false
        tables: true
      options:
        use_llm_validation: true
        use_llm_feasibility: true
        use_vlm_tables: false
        use_normalization: true
"""
