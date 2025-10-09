#!/usr/bin/env python3
"""
Enhanced Abbreviation Logging Module - NO TRUNCATION VERSION
=============================================================
location: corpus_metadata/abbreviation_debugger.py
This version shows FULL lists and FULL expansions without any truncation
"""

import logging
from typing import Dict, List, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class AbbreviationDebugger:
    """Debug tracker for abbreviations throughout the pipeline"""
    
    def __init__(self, verbose: bool = True, max_display: int = None):
        """
        Initialize debugger
        
        Args:
            verbose: Whether to show verbose output
            max_display: Maximum items to display (None = show all)
        """
        self.verbose = verbose
        self.max_display = max_display  # None means show everything
        self.step_counter = 0
        self.abbreviation_history = []
    
    def log_abbreviations(self, abbreviations: List[Dict], step_name: str, 
                          details: str = "", show_expansions: bool = True):
        """
        Log abbreviations at a specific pipeline step
        
        Args:
            abbreviations: List of abbreviation dictionaries
            step_name: Name of the current step
            details: Additional details about the step
            show_expansions: Whether to show expansions
        """
        self.step_counter += 1
        
        # Store in history
        self.abbreviation_history.append({
            'step': self.step_counter,
            'name': step_name,
            'count': len(abbreviations),
            'abbreviations': abbreviations.copy()
        })
        
        # Group by abbreviation text for analysis
        abbrev_groups = defaultdict(list)
        for abbrev in abbreviations:
            if abbrev:  # Skip None entries
                key = abbrev.get('abbreviation', '')
                abbrev_groups[key].append(abbrev)
        
        # Sort by occurrence count
        sorted_abbrevs = sorted(abbrev_groups.items(), 
                               key=lambda x: len(x[1]), reverse=True)
        
        # Create formatted output
        print("\n" + "="*80)
        print(f"STEP {self.step_counter}: {step_name}")
        if details:
            print(f"Details: {details}")
        print("-"*80)
        print(f"Total instances: {len(abbreviations)}, Unique: {len(abbrev_groups)}")
        print("-"*80)
        
        if show_expansions:
            # Dynamic column width calculation
            max_abbrev_len = max(len(abbrev) for abbrev, _ in sorted_abbrevs) if sorted_abbrevs else 12
            max_abbrev_len = max(max_abbrev_len, 12)  # Minimum 12 characters
            
            # Print header with dynamic widths
            print(f"{'Rank':<6} {'Abbrev':<{max_abbrev_len}} {'Count':<7} {'Expansion'}")
            print("-"*80)
            
            # Determine how many to show
            items_to_show = len(sorted_abbrevs) if self.max_display is None else min(self.max_display, len(sorted_abbrevs))
            
            # Show ALL abbreviations (or up to max_display if set)
            for rank, (abbrev_text, instances) in enumerate(sorted_abbrevs[:items_to_show], 1):
                # Get the full expansion - NO TRUNCATION
                expansion = instances[0].get('expansion', 'N/A') if instances else 'N/A'
                
                # Print with full expansion - will wrap naturally in terminal
                print(f"{rank:<6} {abbrev_text:<{max_abbrev_len}} {len(instances):<7} {expansion}")
            
            # Only show "more" message if we actually limited the display
            if self.max_display and len(sorted_abbrevs) > self.max_display:
                print(f"\n... and {len(sorted_abbrevs) - self.max_display} more unique abbreviations")
                print("(Set max_display=None in AbbreviationDebugger to see all)")
        else:
            # Just show abbreviation list without expansions
            items_to_show = len(sorted_abbrevs) if self.max_display is None else min(self.max_display, len(sorted_abbrevs))
            abbrev_list = [key for key, _ in sorted_abbrevs[:items_to_show]]
            print(f"Abbreviations found ({len(abbrev_list)} shown):")
            
            # Print in columns for better readability
            cols = 4
            for i in range(0, len(abbrev_list), cols):
                row = abbrev_list[i:i+cols]
                print("  " + "".join(f"{abbr:<20}" for abbr in row))
            
            if self.max_display and len(sorted_abbrevs) > self.max_display:
                print(f"\n... and {len(sorted_abbrevs) - self.max_display} more")
        
        print("="*80 + "\n")
        
        # Also log to file with full details
        logger.info(f"[STEP {self.step_counter}] {step_name}: {len(abbreviations)} total, {len(abbrev_groups)} unique")
        
        # Log all abbreviations to file for complete record
        if logger.isEnabledFor(logging.DEBUG):
            for abbrev_text, instances in sorted_abbrevs:
                expansion = instances[0].get('expansion', 'N/A') if instances else 'N/A'
                logger.debug(f"  {abbrev_text}: {expansion} (count: {len(instances)})")
    
    def compare_steps(self, step1: int, step2: int):
        """Compare abbreviations between two steps"""
        if step1 > len(self.abbreviation_history) or step2 > len(self.abbreviation_history):
            print("Invalid step numbers")
            return
        
        hist1 = self.abbreviation_history[step1 - 1]
        hist2 = self.abbreviation_history[step2 - 1]
        
        # Get unique abbreviations for each step
        abbrevs1 = set(a.get('abbreviation', '') for a in hist1['abbreviations'] if a)
        abbrevs2 = set(a.get('abbreviation', '') for a in hist2['abbreviations'] if a)
        
        added = abbrevs2 - abbrevs1
        removed = abbrevs1 - abbrevs2
        unchanged = abbrevs1 & abbrevs2
        
        print("\n" + "="*80)
        print(f"COMPARISON: Step {step1} ({hist1['name']}) → Step {step2} ({hist2['name']})")
        print("-"*80)
        print(f"Step {step1}: {len(abbrevs1)} unique abbreviations")
        print(f"Step {step2}: {len(abbrevs2)} unique abbreviations")
        print(f"Added: {len(added)}")
        print(f"Removed: {len(removed)}")
        print(f"Unchanged: {len(unchanged)}")
        
        if added:
            print(f"\nAdded abbreviations:")
            for abbrev in sorted(added):
                print(f"  + {abbrev}")
        
        if removed:
            print(f"\nRemoved abbreviations:")
            for abbrev in sorted(removed):
                print(f"  - {abbrev}")
        
        print("="*80 + "\n")
    
    def print_summary(self):
        """Print summary of all steps"""
        print("\n" + "="*80)
        print("ABBREVIATION PIPELINE SUMMARY")
        print("="*80)
        
        for hist in self.abbreviation_history:
            unique_count = len(set(a.get('abbreviation', '') for a in hist['abbreviations'] if a))
            print(f"Step {hist['step']:2d}: {hist['name']:<30} - {hist['count']:4d} total, {unique_count:4d} unique")
        
        print("="*80 + "\n")
    
    def get_final_abbreviations(self) -> List[Dict]:
        """Get the final set of abbreviations from the last step"""
        if self.abbreviation_history:
            return self.abbreviation_history[-1]['abbreviations']
        return []


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def integrate_debugging(entity_extraction_module, max_display: int = None):
    """
    Integrate debugging into the entity extraction module
    
    Args:
        entity_extraction_module: The module to patch
        max_display: Maximum items to display (None = show all)
    
    Returns:
        AbbreviationDebugger instance
    """
    # Create debugger instance with no display limit by default
    debugger = AbbreviationDebugger(verbose=True, max_display=max_display)
    
    # Check for process_entities_stage_with_promotion
    if hasattr(entity_extraction_module, 'process_entities_stage_with_promotion'):
        original_process = entity_extraction_module.process_entities_stage_with_promotion
        
        def debug_process_entities(text_content, file_path, components, 
                                  stage_config, stage_results, 
                                  abbreviation_context, console, features, 
                                  use_claude):
            """Wrapped version with debugging"""
            # Intercept at key points
            result = original_process(
                text_content, file_path, components, 
                stage_config, stage_results, 
                abbreviation_context, console, features, 
                use_claude
            )
            
            # Log final state
            if 'abbreviations' in result:
                debugger.log_abbreviations(
                    result['abbreviations'],
                    "ENTITIES_STAGE_END",
                    "After entity extraction"
                )
            
            return result
        
        # Apply patch
        entity_extraction_module.process_entities_stage_with_promotion = debug_process_entities
    
    # Check for deduplicate_by_key
    if hasattr(entity_extraction_module, 'deduplicate_by_key'):
        original_deduplicate = entity_extraction_module.deduplicate_by_key
        
        def debug_deduplicate(items: List[Dict], key_fields: tuple) -> List[Dict]:
            """Wrapped deduplication with debugging"""
            # Log before deduplication (only if it's abbreviations)
            if key_fields == ('abbreviation', 'expansion'):
                debugger.log_abbreviations(
                    items,
                    "BEFORE_DEDUPLICATION",
                    f"Input: {len(items)} items"
                )
            
            # Call original function
            result = original_deduplicate(items, key_fields)
            
            # Log after deduplication (only if it's abbreviations)
            if key_fields == ('abbreviation', 'expansion'):
                debugger.log_abbreviations(
                    result,
                    "AFTER_DEDUPLICATION",
                    f"Output: {len(result)} unique items"
                )
            
            return result
        
        # Apply patch
        entity_extraction_module.deduplicate_by_key = debug_deduplicate
    
    # Store debugger reference for external access
    entity_extraction_module.abbreviation_debugger = debugger
    
    print(f"✓ Abbreviation debugging integrated into pipeline")
    print(f"  - process_entities_stage_with_promotion: {'✓' if hasattr(entity_extraction_module, 'process_entities_stage_with_promotion') else '✗'}")
    print(f"  - deduplicate_by_key: {'✓' if hasattr(entity_extraction_module, 'deduplicate_by_key') else '✗'}")
    print(f"  - Display mode: {'ALL items' if max_display is None else f'Up to {max_display} items'}")
    
    return debugger


# ============================================================================
# STANDALONE DEBUGGING FUNCTIONS (unchanged)
# ============================================================================

def deduplicate_abbreviations(abbreviations: List[Dict]) -> List[Dict]:
    """
    Standalone deduplication function for abbreviations
    Used when the main module doesn't have this function
    """
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for abbrev in abbreviations:
        if abbrev:
            key = (abbrev.get('abbreviation', ''), abbrev.get('expansion', ''))
            grouped[key].append(abbrev)
    
    deduplicated = []
    for instances in grouped.values():
        if not instances:
            continue
        
        # Merge instances
        merged = dict(instances[0])
        merged['occurrences'] = sum(inst.get('occurrences', 1) for inst in instances)
        merged['confidence'] = max(inst.get('confidence', 0) for inst in instances)
        
        # Merge lists
        for field in ['positions', 'dictionary_sources']:
            values = []
            for inst in instances:
                if field in inst:
                    if isinstance(inst[field], list):
                        values.extend(inst[field])
                    else:
                        values.append(inst[field])
            if values:
                merged[field] = list(set(values)) if field == 'dictionary_sources' else values
        
        deduplicated.append(merged)
    
    return deduplicated


def prepare_abbreviation_context(abbrev_results: Dict) -> Dict:
    """
    Standalone function to prepare abbreviation context
    Used when the main module doesn't have this function
    """
    all_abbreviations = abbrev_results.get('abbreviations', [])
    
    # Categorize abbreviations
    drug_candidates = []
    disease_candidates = []
    
    for abbrev in all_abbreviations:
        category = abbrev.get('category', '').lower()
        if category == 'drug':
            drug_candidates.append(abbrev)
        elif category == 'disease':
            disease_candidates.append(abbrev)
        elif category in ['ambiguous', 'unknown']:
            # Add to both for ambiguous cases
            drug_candidates.append(abbrev)
            disease_candidates.append(abbrev)
    
    return {
        'all_abbreviations': all_abbreviations,
        'drug_candidates': drug_candidates,
        'disease_candidates': disease_candidates
    }