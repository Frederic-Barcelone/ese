#!/usr/bin/env python3
"""
ERN Content Curator v2 - Optimized Single-Call Version
=======================================================

OPTIMIZATIONS vs v1:
- SINGLE LLM CALL: Classification + Entity Extraction in one prompt (50% fewer API calls)
- INCREMENTAL SAVING: Results saved every 10 files (not just at the end)
- BETTER CHECKPOINTING: Tracks entities in checkpoint too
- COST TRACKING: Estimates API usage
- SMART DEDUPLICATION: Merges partial names, handles synonyms, combines info from multiple sources

Features:
- Claude-powered classification (USE/NOT_USE)
- Entity extraction (experts, institutions, diseases, registries, projects, etc.)
- Copies selected files to curated folder
- JSON output of all extracted entities
- Checkpoint/resume capability

Requirements:
    pip install anthropic

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python ern_content_curator_v2.py

Author: ERN RAG Pipeline
Date: 2025-11
"""

import os
import json
import shutil
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from difflib import SequenceMatcher

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Folders
    "source_folder": "EU_ERN_DATA/rag_content",
    "curated_folder": "EU_ERN_DATA/rag_content_curated",
    "output_folder": "EU_ERN_DATA/curation_output",
    
    # Claude API
    "api_key": None,  # Set here OR use ANTHROPIC_API_KEY env var
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 6000,
    
    # Processing
    "dry_run": False,
    "skip_empty_files": True,
    "min_content_chars": 100,
    "save_interval": 10,  # Save results every N files
    "verbose": True,
}


# ============================================================================
# COMBINED PROMPT (Classification + Extraction in ONE call)
# ============================================================================

COMBINED_PROMPT = '''You are analyzing medical content about rare diseases for a RAG knowledge base.

Your task: 
1. CLASSIFY the document (USE or NOT_USE)
2. If USE: EXTRACT all valuable entities

## Classification Criteria

**USE** - Include if the document contains ANY of these:
- Medical/clinical information about rare diseases
- Expert/staff names with roles or credentials
- Healthcare institution details
- Disease information, guidelines, care pathways, protocols
- Research projects, patient registries, databases
- Educational content (courses, webinars, training)
- Organizational structure, governance, network information

**NOT_USE** - Exclude if the document:
- Is empty or contains only headers/navigation
- Contains only past event dates without scientific content
- Is purely administrative (travel, hotels, registration)
- Contains only media links without transcripts
- Shows "No matches" or empty state messages

## Entity Extraction (only if USE)

Extract these entities if present:

1. **experts** - Doctors, researchers, coordinators, staff, speakers
   - IMPORTANT: Always extract FULL NAME if available (first + last name)
   - Include: name, title (Dr/Prof/PhD/MD), role, institution, expertise_areas

2. **institutions** - Hospitals, universities, research centers
   - Include: name, city, country, type

3. **diseases** - Medical conditions, syndromes, disorders
   - Use the most specific/common name
   - Include: name, category (glomerulopathy/tubulopathy/ciliopathy/metabolic/tma/developmental/general)

4. **projects** - Research projects, EU initiatives, clinical trials
   - Include: name, acronym, type, description

5. **registries** - Patient registries, databases, biobanks
   - Include: name, acronym, type, focus_area

6. **guidelines** - Clinical guidelines, protocols, pathways
   - Include: title, type, disease_area

7. **educational_resources** - Courses, webinars, workshops
   - Include: title, type, topics

---

## Document to Analyze

**Filename:** {filename}
**Source URL:** {url}
**Title:** {title}

**Content:**
{content}

---

## Response Format

Respond with ONLY valid JSON:

{{
  "classification": {{
    "decision": "USE" or "NOT_USE",
    "confidence": "HIGH" or "MEDIUM" or "LOW",
    "reason": "1-2 sentence explanation",
    "content_type": "expert_directory|educational|research|organizational|clinical|guidelines|event|empty|administrative",
    "quality_score": 1-5
  }},
  "entities": {{
    "experts": [
      {{"name": "...", "title": "...", "role": "...", "institution": "...", "expertise_areas": ["..."]}}
    ],
    "institutions": [
      {{"name": "...", "city": "...", "country": "...", "type": "..."}}
    ],
    "diseases": [
      {{"name": "...", "category": "..."}}
    ],
    "projects": [
      {{"name": "...", "acronym": "...", "type": "...", "description": "..."}}
    ],
    "registries": [
      {{"name": "...", "acronym": "...", "type": "...", "focus_area": "..."}}
    ],
    "guidelines": [
      {{"title": "...", "type": "...", "disease_area": "..."}}
    ],
    "educational_resources": [
      {{"title": "...", "type": "...", "topics": ["..."]}}
    ]
  }}
}}

IMPORTANT:
- If decision is NOT_USE, return empty arrays [] for all entity categories
- If decision is USE, extract ALL entities found in the document
- Use empty arrays [] for entity categories with nothing found'''


# ============================================================================
# SMART DEDUPLICATION
# ============================================================================

class SmartDeduplicator:
    """
    Smart deduplication that:
    - Matches partial names (e.g., "Tanja" matches "Tanja Wlodkowski")
    - Handles case-insensitive matching
    - Merges information from multiple sources
    - Handles synonyms and abbreviations
    """
    
    # Known disease synonyms (comprehensive list based on ERN data)
    DISEASE_SYNONYMS = {
        # Rare kidney disease variants
        "rare kidney diseases": ["rare renal diseases", "rare kidney disease", "rare renal diagnoses", 
                                  "rare disorders of the kidneys", "rare and ultra-rare kidney diseases",
                                  "rare renal disorders", "inherited kidney diseases", 
                                  "hereditary kidney disorders", "hereditary kidney diseases",
                                  "rare kidney disorders", "complex kidney diseases", "kidney disease"],
        "rare diseases": ["rare and complex diseases", "ultra-rare diseases"],
        
        # CKD variants
        "chronic kidney disease": ["paediatric chronic kidney disease", "pediatric chronic kidney disease",
                                   "pediatric ckd", "ckd"],
        
        # Nephrotic syndrome variants  
        "nephrotic syndrome": ["genetic steroid-resistant nephrotic syndrome", 
                               "steroid resistant nephrotic syndrome",
                               "steroid-resistant nephrotic syndrome", "srns"],
        
        # Collagen IV / Alport variants
        "alport syndrome": ["collagen iv glomerulopathies", "collagen iv nephropathies", 
                           "col4a nephropathies"],
        
        # Tubulopathy variants
        "bartter syndrome": ["bartter"],
        "gitelman syndrome": ["gitelman"],
        "renal tubular acidosis": ["distal renal tubular acidosis", "drta", "rta"],
        "tubulopathies": ["tubulopathy", "tubular disorders"],
        
        # Metabolic diseases
        "cystinosis": [],
        "cystinuria": [],
        "fabry disease": ["fabry", "anderson-fabry disease"],
        "hyperoxaluria": ["primary hyperoxaluria", "ph1", "ph2", "ph3"],
        "hypophosphatemic rickets": ["x-linked hypophosphatemic rickets", "xlh"],
        
        # PKD variants
        "adpkd": ["autosomal dominant polycystic kidney disease", "pediatric adpkd", 
                  "pediatric autosomal dominant polycystic kidney disease"],
        "arpkd": ["autosomal recessive polycystic kidney disease"],
        "polycystic kidney disease": ["polycystic kidney diseases", "pkd", 
                                       "hereditary cystic renal diseases", "cystic kidney disease",
                                       "early onset cystic kidney disease"],
        
        # Glomerulopathies
        "iga nephropathy": ["iga vasculitis", "iga vasculitis with nephritis", "igan"],
        "glomerulopathies": ["hereditary glomerulopathies", "immune glomerulopathies", 
                            "glomerular diseases"],
        "c3 glomerulopathy": ["c3g", "c3 glomerulonephritis"],
        "membranous nephropathy": ["mn", "membranous glomerulonephritis"],
        "lupus nephritis": ["sle nephritis", "systemic lupus erythematosus nephritis"],
        
        # CAKUT variants
        "cakut": ["congenital anomalies of kidney and urinary tract", 
                  "congenital anomalies of the kidney and urinary tract",
                  "renal or urinary tract malformations", "congenital kidney disorders"],
        
        # TMA variants
        "hemolytic uremic syndrome": ["hus", "ahus", "atypical hemolytic uremic syndrome"],
        "thrombotic microangiopathies": ["tma", "thrombotic microangiopathy"],
        
        # Other
        "podocyte disorders": ["podocytopathy", "podocytopathies"],
        "nephrogenic diabetes insipidus": ["ndi"],
        "siadh": ["syndrome of inappropriate antidiuretic hormone secretion"],
        "lowe syndrome": ["oculocerebrorenal syndrome", "lowe-bickel syndrome"],
        "dent disease": ["dent disease 1", "dent disease 2"],
    }
    
    # Known institution synonyms (comprehensive list)
    INSTITUTION_SYNONYMS = {
        # ERKNet variants
        "erknet": ["european reference network for rare kidney diseases", 
                   "european rare kidney disease reference network",
                   "erknet - european reference network for rare kidney diseases"],
        
        # EU institutions
        "european commission": ["ec", "european commission, dg health"],
        "european union": ["eu"],
        
        # Heidelberg
        "heidelberg university": ["university of heidelberg"],
        "university hospital heidelberg": ["university hospital of heidelberg", 
                                            "heidelberg university hospital",
                                            "universitätsklinikum heidelberg"],
        
        # Leuven
        "university hospitals leuven": ["ku leuven", "leuven"],
        
        # Professional societies
        "european society for paediatric nephrology": ["espn"],
        "international pediatric nephrology association": ["ipna"],
        "european renal association": ["era"],
        
        # Amsterdam
        "amsterdam umc": ["amsterdam medical center"],
        
        # Other ERNs
        "ern bone": ["ern-bond", "ern bond"],
        "ern epicare": ["ern-epicare"],
        "ern eye": ["ern-eye"],
        "eurobloodnet": ["ern-eurobloodnet"],
        "metabern": ["ern-metabern"],
    }
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize a name for comparison."""
        if not name:
            return ""
        # Remove titles, lowercase, strip whitespace
        name = name.lower().strip()
        # Remove common titles
        for title in ["prof.", "prof", "dr.", "dr", "phd", "md", "msc", "bsc"]:
            name = name.replace(title, "")
        # Remove extra spaces
        name = " ".join(name.split())
        return name
    
    @staticmethod
    def similarity_ratio(s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    @staticmethod
    def is_partial_match(short_name: str, full_name: str) -> bool:
        """Check if short_name is a partial match of full_name (e.g., first name only)."""
        short_norm = SmartDeduplicator.normalize_name(short_name)
        full_norm = SmartDeduplicator.normalize_name(full_name)
        
        if not short_norm or not full_norm:
            return False
        
        # Check if short name is contained in full name
        if short_norm in full_norm:
            return True
        
        # Check if short name matches first or last name
        full_parts = full_norm.split()
        if short_norm in full_parts:
            return True
        
        return False
    
    @staticmethod
    def merge_expert_info(existing: Dict, new: Dict) -> Dict:
        """Merge information from two expert records, preferring fuller information."""
        merged = existing.copy()
        
        # Prefer longer/fuller name
        if len(new.get("name", "")) > len(merged.get("name", "")):
            merged["name"] = new["name"]
        
        # Prefer title with more info (e.g., "Prof. Dr." over "Dr")
        new_title = new.get("title", "")
        existing_title = merged.get("title", "")
        if len(new_title) > len(existing_title) or ("prof" in new_title.lower() and "prof" not in existing_title.lower()):
            merged["title"] = new_title
        
        # Prefer longer role description
        if len(new.get("role", "")) > len(merged.get("role", "")):
            merged["role"] = new["role"]
        
        # Prefer non-empty institution with more detail
        new_inst = new.get("institution", "")
        existing_inst = merged.get("institution", "")
        if new_inst and (not existing_inst or len(new_inst) > len(existing_inst)):
            if existing_inst.lower() not in ["erknet", ""]:  # Don't replace specific with generic
                merged["institution"] = new_inst
        
        # Merge expertise areas
        existing_areas = set(merged.get("expertise_areas", []))
        new_areas = set(new.get("expertise_areas", []))
        merged["expertise_areas"] = list(existing_areas | new_areas)
        
        # Track all source files
        existing_sources = merged.get("_source_files", [merged.get("_source_file", "")])
        new_source = new.get("_source_file", "")
        if new_source and new_source not in existing_sources:
            existing_sources.append(new_source)
        merged["_source_files"] = [s for s in existing_sources if s]
        if "_source_file" in merged:
            del merged["_source_file"]
        
        return merged
    
    @staticmethod
    def merge_entity_info(existing: Dict, new: Dict, key_field: str = "name") -> Dict:
        """Generic merge for entities, keeping the most complete information."""
        merged = existing.copy()
        
        for field, value in new.items():
            if field == "_source_file":
                continue
            if field == "_source_files":
                continue
            
            existing_val = merged.get(field, "")
            
            # For lists, merge them
            if isinstance(value, list) and isinstance(existing_val, list):
                merged[field] = list(set(existing_val) | set(value))
            # For strings, prefer longer/more complete
            elif isinstance(value, str) and isinstance(existing_val, str):
                if len(value) > len(existing_val):
                    merged[field] = value
            # For empty values, fill in
            elif not existing_val and value:
                merged[field] = value
        
        # Track source files
        existing_sources = merged.get("_source_files", [merged.get("_source_file", "")])
        new_source = new.get("_source_file", "")
        if new_source and new_source not in existing_sources:
            existing_sources.append(new_source)
        merged["_source_files"] = [s for s in existing_sources if s]
        if "_source_file" in merged:
            del merged["_source_file"]
        
        return merged
    
    @classmethod
    def find_synonym_group(cls, name: str, synonym_dict: Dict) -> Optional[str]:
        """Find if a name belongs to a synonym group."""
        name_lower = name.lower().strip()
        
        # Check if it's a canonical name
        if name_lower in synonym_dict:
            return name_lower
        
        # Check if it's in any synonym list
        for canonical, synonyms in synonym_dict.items():
            if name_lower in [s.lower() for s in synonyms]:
                return canonical
            # Check partial match for longer names
            for syn in synonyms:
                if cls.similarity_ratio(name_lower, syn.lower()) > 0.85:
                    return canonical
        
        return None
    
    @classmethod
    def deduplicate_experts(cls, experts: List[Dict]) -> List[Dict]:
        """Deduplicate experts with smart name matching."""
        if not experts:
            return []
        
        # Group by normalized name, handling partial matches
        merged = {}  # normalized_name -> merged_dict
        
        for expert in experts:
            name = expert.get("name", "").strip()
            if not name:
                continue
            
            norm_name = cls.normalize_name(name)
            
            # Try to find an existing match
            matched_key = None
            for existing_key in merged.keys():
                # Exact match
                if norm_name == existing_key:
                    matched_key = existing_key
                    break
                # Partial match (first name only)
                if cls.is_partial_match(norm_name, existing_key) or cls.is_partial_match(existing_key, norm_name):
                    matched_key = existing_key
                    break
                # High similarity
                if cls.similarity_ratio(norm_name, existing_key) > 0.85:
                    matched_key = existing_key
                    break
            
            if matched_key:
                # Merge with existing
                merged[matched_key] = cls.merge_expert_info(merged[matched_key], expert)
                # Update key if new name is longer
                if len(norm_name) > len(matched_key):
                    merged[norm_name] = merged.pop(matched_key)
            else:
                # New expert
                merged[norm_name] = expert.copy()
        
        return list(merged.values())
    
    @classmethod
    def deduplicate_diseases(cls, diseases: List[Dict]) -> List[Dict]:
        """Deduplicate diseases with synonym handling."""
        if not diseases:
            return []
        
        merged = {}  # canonical_name -> merged_dict
        
        for disease in diseases:
            name = disease.get("name", "").strip()
            if not name:
                continue
            
            name_lower = name.lower()
            
            # Find synonym group
            canonical = cls.find_synonym_group(name, cls.DISEASE_SYNONYMS)
            
            if canonical:
                # Use canonical name
                if canonical in merged:
                    merged[canonical] = cls.merge_entity_info(merged[canonical], disease)
                else:
                    # Create new entry with canonical name (keep original capitalization if first)
                    disease_copy = disease.copy()
                    merged[canonical] = disease_copy
            else:
                # Check for case-insensitive duplicates
                if name_lower in merged:
                    merged[name_lower] = cls.merge_entity_info(merged[name_lower], disease)
                else:
                    merged[name_lower] = disease.copy()
        
        return list(merged.values())
    
    @classmethod
    def deduplicate_institutions(cls, institutions: List[Dict]) -> List[Dict]:
        """Deduplicate institutions with synonym handling."""
        if not institutions:
            return []
        
        merged = {}  # normalized_name -> merged_dict
        
        for inst in institutions:
            name = inst.get("name", "").strip()
            if not name:
                continue
            
            name_lower = name.lower()
            
            # Find synonym group
            canonical = cls.find_synonym_group(name, cls.INSTITUTION_SYNONYMS)
            key = canonical if canonical else name_lower
            
            if key in merged:
                merged[key] = cls.merge_entity_info(merged[key], inst)
            else:
                merged[key] = inst.copy()
        
        return list(merged.values())
    
    @classmethod
    def deduplicate_generic(cls, items: List[Dict], key_field: str = "name") -> List[Dict]:
        """Generic deduplication for other entity types."""
        if not items:
            return []
        
        merged = {}  # normalized_key -> merged_dict
        
        for item in items:
            key = item.get(key_field, item.get("title", "")).strip()
            if not key:
                continue
            
            key_lower = key.lower()
            
            # Check for similar existing entries
            matched_key = None
            for existing_key in merged.keys():
                if cls.similarity_ratio(key_lower, existing_key) > 0.85:
                    matched_key = existing_key
                    break
            
            if matched_key:
                merged[matched_key] = cls.merge_entity_info(merged[matched_key], item, key_field)
            else:
                merged[key_lower] = item.copy()
        
        return list(merged.values())
    
    @classmethod
    def deduplicate_all(cls, entities: Dict) -> Dict:
        """Deduplicate all entity categories."""
        return {
            "experts": cls.deduplicate_experts(entities.get("experts", [])),
            "institutions": cls.deduplicate_institutions(entities.get("institutions", [])),
            "diseases": cls.deduplicate_diseases(entities.get("diseases", [])),
            "projects": cls.deduplicate_generic(entities.get("projects", []), "acronym"),
            "registries": cls.deduplicate_generic(entities.get("registries", []), "acronym"),
            "guidelines": cls.deduplicate_generic(entities.get("guidelines", []), "title"),
            "educational_resources": cls.deduplicate_generic(entities.get("educational_resources", []), "title"),
        }


# ============================================================================
# CLAUDE CLIENT
# ============================================================================

class ClaudeClient:
    """Client for Anthropic Claude API with usage tracking."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
    
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        return self._client
    
    def complete(self, prompt: str, max_tokens: int = 6000) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Track usage
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.total_calls += 1
        
        return response.content[0].text
    
    def get_usage_stats(self) -> Dict:
        """Return usage statistics."""
        # Approximate costs (Claude Sonnet pricing as of 2024)
        input_cost = (self.total_input_tokens / 1_000_000) * 3.0  # $3 per 1M input tokens
        output_cost = (self.total_output_tokens / 1_000_000) * 15.0  # $15 per 1M output tokens
        
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4)
        }
    
    @staticmethod
    def parse_json(text: str) -> Optional[Dict]:
        """Parse JSON from Claude's response."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        return None


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProcessingResult:
    filename: str
    filepath: str
    decision: str
    confidence: str
    reason: str
    content_type: str
    quality_score: int
    entities: Dict = field(default_factory=dict)
    skipped_llm: bool = False


@dataclass
class ProcessingStats:
    total_files: int = 0
    use_count: int = 0
    not_use_count: int = 0
    llm_calls: int = 0
    skipped_empty: int = 0
    errors: List[Dict] = field(default_factory=list)


# ============================================================================
# FILE UTILITIES
# ============================================================================

def read_markdown_file(filepath: Path) -> Dict:
    """Read markdown file with YAML frontmatter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    
    result = {
        "filename": filepath.name,
        "filepath": str(filepath),
        "url": "",
        "title": "",
        "content": raw_content,
    }
    
    if raw_content.startswith("---"):
        parts = raw_content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            result["content"] = parts[2].strip()
            
            url_match = re.search(r'^url:\s*(.+)$', frontmatter, re.MULTILINE)
            if url_match:
                result["url"] = url_match.group(1).strip()
            
            title_match = re.search(r'^title:\s*(.+)$', frontmatter, re.MULTILINE)
            if title_match:
                result["title"] = title_match.group(1).strip()
    
    return result


def get_content_length(content: str) -> int:
    """Get clean content length (excluding frontmatter)."""
    if content.strip().startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2]
    
    cleaned = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return len(cleaned)


def truncate_content(content: str, max_chars: int = 8000) -> str:
    """Truncate content for LLM prompt."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[... content truncated ...]"


# ============================================================================
# MAIN CURATOR CLASS
# ============================================================================

class ERNCuratorV2:
    """Optimized curator with single LLM call per file."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.source_folder = Path(config["source_folder"])
        self.curated_folder = Path(config["curated_folder"])
        self.output_folder = Path(config["output_folder"])
        
        # Get API key
        api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found.\n"
                "Set ANTHROPIC_API_KEY environment variable:\n"
                "  export ANTHROPIC_API_KEY=your_key_here"
            )
        
        self.claude = ClaudeClient(api_key=api_key, model=config.get("model", "claude-sonnet-4-20250514"))
        
        # Results
        self.results: List[ProcessingResult] = []
        self.entities = {
            "experts": [],
            "institutions": [],
            "diseases": [],
            "projects": [],
            "registries": [],
            "guidelines": [],
            "educational_resources": []
        }
        self.stats = ProcessingStats()
        
        # Checkpoint
        self.checkpoint_file = self.output_folder / "checkpoint_v2.json"
        self.processed_files: set = set()
    
    def setup_folders(self):
        self.curated_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self):
        """Load checkpoint for resume capability."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                
                self.processed_files = set(checkpoint.get("processed_files", []))
                self.stats.use_count = checkpoint.get("use_count", 0)
                self.stats.not_use_count = checkpoint.get("not_use_count", 0)
                self.stats.skipped_empty = checkpoint.get("skipped_empty", 0)
                self.stats.llm_calls = checkpoint.get("llm_calls", 0)
                
                # Load results
                for r in checkpoint.get("results", []):
                    self.results.append(ProcessingResult(
                        filename=r["filename"],
                        filepath=r["filepath"],
                        decision=r["decision"],
                        confidence=r["confidence"],
                        reason=r["reason"],
                        content_type=r["content_type"],
                        quality_score=r["quality_score"],
                        entities=r.get("entities", {}),
                        skipped_llm=r.get("skipped_llm", False)
                    ))
                
                # Load entities
                self.entities = checkpoint.get("entities", self.entities)
                
                if self.processed_files:
                    print(f"\n*** RESUMING: {len(self.processed_files)} files already processed ***")
                    print(f"    USE: {self.stats.use_count} | NOT_USE: {self.stats.not_use_count} | Skipped: {self.stats.skipped_empty}\n")
                return True
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        return False
    
    def save_checkpoint(self):
        """Save checkpoint with all data."""
        checkpoint = {
            "processed_files": list(self.processed_files),
            "results": [asdict(r) for r in self.results],
            "entities": self.entities,
            "use_count": self.stats.use_count,
            "not_use_count": self.stats.not_use_count,
            "skipped_empty": self.stats.skipped_empty,
            "llm_calls": self.stats.llm_calls,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    
    def save_results(self, deduplicate: bool = False):
        """Save all results to JSON files (incremental)."""
        timestamp = datetime.now().isoformat()
        
        # Optionally deduplicate before saving
        entities_to_save = self.entities
        if deduplicate:
            print("\n  Deduplicating entities...")
            entities_to_save = SmartDeduplicator.deduplicate_all(self.entities)
            
            # Print dedup stats
            for category in entities_to_save:
                before = len(self.entities.get(category, []))
                after = len(entities_to_save.get(category, []))
                if before != after:
                    print(f"    {category}: {before} -> {after} (merged {before - after})")
        
        # Classification results
        classification_data = {
            "metadata": {
                "date": timestamp,
                "source_folder": str(self.source_folder),
                "model": self.config["model"],
                "total_files": self.stats.total_files,
                "processed_files": len(self.processed_files),
                "use_count": self.stats.use_count,
                "not_use_count": self.stats.not_use_count,
                "skipped_empty": self.stats.skipped_empty,
                "llm_calls": self.stats.llm_calls,
                "api_usage": self.claude.get_usage_stats(),
                "deduplicated": deduplicate
            },
            "files": [asdict(r) for r in self.results]
        }
        
        with open(self.output_folder / "classification_results.json", 'w', encoding='utf-8') as f:
            json.dump(classification_data, f, indent=2, ensure_ascii=False)
        
        # Entity results
        entity_data = {
            "metadata": {
                "date": timestamp,
                "files_with_entities": self.stats.use_count,
                "model": self.config["model"],
                "deduplicated": deduplicate
            },
            "entities": entities_to_save
        }
        
        with open(self.output_folder / "extracted_entities.json", 'w', encoding='utf-8') as f:
            json.dump(entity_data, f, indent=2, ensure_ascii=False)
        
        # Individual entity files
        for category, items in entities_to_save.items():
            with open(self.output_folder / f"{category}.json", 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
    
    def get_source_files(self) -> List[Path]:
        """Get list of markdown files."""
        return sorted(self.source_folder.glob("*.md"))
    
    def process_file(self, file_data: Dict) -> ProcessingResult:
        """Process a single file with combined classification + extraction."""
        filename = file_data["filename"]
        content = file_data["content"]
        
        # Quick check for empty files (no LLM call needed)
        if self.config["skip_empty_files"]:
            content_len = get_content_length(content)
            if content_len < self.config["min_content_chars"]:
                self.stats.skipped_empty += 1
                return ProcessingResult(
                    filename=filename,
                    filepath=file_data["filepath"],
                    decision="NOT_USE",
                    confidence="HIGH",
                    reason=f"Content too short ({content_len} chars)",
                    content_type="empty",
                    quality_score=1,
                    entities={},
                    skipped_llm=True
                )
        
        # Build combined prompt
        prompt = COMBINED_PROMPT.format(
            filename=filename,
            url=file_data.get("url", ""),
            title=file_data.get("title", ""),
            content=truncate_content(content)
        )
        
        # Single LLM call for both classification and extraction
        response = self.claude.complete(prompt, self.config["max_tokens"])
        self.stats.llm_calls += 1
        
        parsed = self.claude.parse_json(response)
        
        if parsed:
            classification = parsed.get("classification", {})
            entities = parsed.get("entities", {})
            
            return ProcessingResult(
                filename=filename,
                filepath=file_data["filepath"],
                decision=classification.get("decision", "NOT_USE"),
                confidence=classification.get("confidence", "LOW"),
                reason=classification.get("reason", ""),
                content_type=classification.get("content_type", "unknown"),
                quality_score=classification.get("quality_score", 0),
                entities=entities,
                skipped_llm=False
            )
        else:
            return ProcessingResult(
                filename=filename,
                filepath=file_data["filepath"],
                decision="NOT_USE",
                confidence="LOW",
                reason="Failed to parse LLM response",
                content_type="error",
                quality_score=0,
                entities={},
                skipped_llm=False
            )
    
    def copy_file(self, source_path: Path) -> bool:
        """Copy file to curated folder."""
        if self.config["dry_run"]:
            return True
        try:
            dest_path = self.curated_folder / source_path.name
            shutil.copy2(source_path, dest_path)
            return True
        except Exception as e:
            print(f"    Error copying: {e}")
            return False
    
    def merge_entities(self, entities: Dict, source_file: str):
        """Merge extracted entities into master lists."""
        if not entities:
            return
        
        for category in self.entities.keys():
            items = entities.get(category, [])
            for item in items:
                if isinstance(item, dict) and item:
                    # Check if it has meaningful content
                    has_content = any(v for k, v in item.items() if v and k != "_source_file")
                    if has_content:
                        item["_source_file"] = source_file
                        self.entities[category].append(item)
    
    def process(self):
        """Main processing method."""
        self.setup_folders()
        self.load_checkpoint()
        
        files = self.get_source_files()
        if not files:
            print("No files found!")
            return
        
        self.stats.total_files = len(files)
        
        print("\n" + "=" * 60)
        print("ERN Content Curator v2 (Optimized + Smart Deduplication)")
        print("=" * 60)
        print(f"  Total files: {len(files)}")
        print(f"  Already processed: {len(self.processed_files)}")
        print(f"  Remaining: {len(files) - len(self.processed_files)}")
        print("=" * 60)
        
        for i, filepath in enumerate(files, 1):
            filename = filepath.name
            
            # Skip already processed files
            if filename in self.processed_files:
                continue
            
            print(f"\n[{i}/{len(files)}] {filename}")
            
            try:
                file_data = read_markdown_file(filepath)
                result = self.process_file(file_data)
                self.results.append(result)
                self.processed_files.add(filename)
                
                if result.decision == "USE":
                    self.stats.use_count += 1
                    self.copy_file(filepath)
                    self.merge_entities(result.entities, filename)
                    
                    # Count entities found
                    entity_counts = {k: len(v) for k, v in result.entities.items() 
                                    if isinstance(v, list) and v}
                    if entity_counts:
                        print(f"  ✓ USE [{result.confidence}] ({result.content_type}) - Entities: {entity_counts}")
                    else:
                        print(f"  ✓ USE [{result.confidence}] ({result.content_type})")
                else:
                    self.stats.not_use_count += 1
                    skip_note = " [skipped LLM]" if result.skipped_llm else ""
                    print(f"  ✗ NOT_USE [{result.confidence}] ({result.content_type}){skip_note}")
                
                # Save checkpoint and results periodically
                if i % self.config["save_interval"] == 0:
                    self.save_checkpoint()
                    self.save_results(deduplicate=False)  # Don't dedupe on interim saves
                    usage = self.claude.get_usage_stats()
                    print(f"\n  [Saved] LLM calls: {usage['total_calls']} | Est. cost: ${usage['estimated_cost_usd']:.4f}")
                    
            except KeyboardInterrupt:
                print("\n\n*** INTERRUPTED - Saving progress... ***")
                self.save_checkpoint()
                self.save_results(deduplicate=True)  # Dedupe on interrupt
                self.print_summary()
                return
            except Exception as e:
                print(f"  ERROR: {e}")
                self.stats.errors.append({"file": filename, "error": str(e)})
        
        # Final save with deduplication
        self.save_checkpoint()
        self.save_results(deduplicate=True)
        self.print_summary()
        
        # Remove checkpoint on successful completion
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("\n[Checkpoint cleaned up]")
    
    def print_summary(self):
        """Print summary."""
        usage = self.claude.get_usage_stats()
        
        # Get deduplicated counts
        deduped = SmartDeduplicator.deduplicate_all(self.entities)
        
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"  Total files:      {self.stats.total_files}")
        print(f"  Processed:        {len(self.processed_files)}")
        print(f"  Selected (USE):   {self.stats.use_count}")
        print(f"  Excluded:         {self.stats.not_use_count}")
        print(f"  Skipped (empty):  {self.stats.skipped_empty}")
        print(f"  Errors:           {len(self.stats.errors)}")
        
        print("\n  API Usage:")
        print(f"    LLM calls:      {usage['total_calls']}")
        print(f"    Input tokens:   {usage['total_input_tokens']:,}")
        print(f"    Output tokens:  {usage['total_output_tokens']:,}")
        print(f"    Est. cost:      ${usage['estimated_cost_usd']:.4f}")
        
        print("\n  Entity counts (after deduplication):")
        for category, items in deduped.items():
            original = len(self.entities.get(category, []))
            deduped_count = len(items)
            if original != deduped_count:
                print(f"    {category}: {deduped_count} (merged {original - deduped_count} duplicates)")
            elif items:
                print(f"    {category}: {deduped_count}")
        
        print(f"\n  Results saved to: {self.output_folder}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("ERN Content Curator v2")
    print("Optimized: Single LLM call + Smart Deduplication")
    print("=" * 60)
    
    try:
        curator = ERNCuratorV2(CONFIG)
        curator.process()
    except ValueError as e:
        print(f"\nConfiguration Error:\n{e}")
    except ImportError as e:
        print(f"\nMissing Dependency:\n{e}")


if __name__ == "__main__":
    main()