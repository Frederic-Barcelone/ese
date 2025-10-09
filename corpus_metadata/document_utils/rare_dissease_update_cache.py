#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/rare_dissease_update_cache.py
#

"""
document_utils/rare_disease_update_cache.py - Cache management for rare disease document processing
===================================================================================================

Run this script when you improve extraction logic and need to reprocess files.
Now uses cache_settings.yaml for all configuration.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from corpus_metadata.document_utils.reader_extraction_cache import VersionedExtractionCache
from corpus_metadata.document_utils.metadata_config_loader import CorpusConfig


# Initialize cache and config
cache = VersionedExtractionCache()
config = CorpusConfig()


def after_drug_detection_improvement():
    """Run this after improving drug detection logic"""
    print("\n🔧 UPDATING DRUG DETECTION")
    print("=" * 60)
    
    # Get current version from config
    current_version = config.get_cache_version('drugs')
    print(f"Current drug detector version: {current_version}")
    
    # Calculate new version
    major, minor = current_version.split('.')
    new_version = f"{major}.{int(minor) + 1}"
    
    # Update version in YAML file
    config.update_cache_version('drugs', new_version)
    print(f"✓ Drug detector version updated to {new_version}")
    
    # Mark files for reprocessing
    # Option 1: Reprocess ALL files
    # cache.mark_for_reprocessing('drugs', pattern='*.pdf')
    
    # Option 2: Reprocess specific patterns
    cache.mark_for_reprocessing('drugs', pattern='00001_ALXN1210-CSA-AKI-318_protocol.pdf')
    cache.mark_for_reprocessing('drugs', pattern='ALXN*.pdf')
    cache.mark_for_reprocessing('drugs', pattern='*protocol*.pdf')
    cache.mark_for_reprocessing('drugs', pattern='*ravulizumab*.pdf')
    
    print("\n✓ Marked for reprocessing:")
    print("  - ALXN*.pdf (Alexion documents)")
    print("  - *protocol*.pdf (All protocols)")
    print("  - *ravulizumab*.pdf (Ravulizumab studies)")
    
    show_status()


def after_classification_improvement():
    """Run this after improving document classification"""
    print("\n🔧 UPDATING CLASSIFICATION")
    print("=" * 60)
    
    # Get current version from config
    current_version = config.get_cache_version('classification')
    print(f"Current classification version: {current_version}")
    
    # Increment version
    major, minor = current_version.split('.')
    new_version = f"{major}.{int(minor) + 1}"
    
    # Update in YAML
    config.update_cache_version('classification', new_version)
    print(f"✓ Classification version updated to {new_version}")
    
    # Mark all files for classification reprocessing
    cache.mark_for_reprocessing('classification', pattern='*.pdf')
    print("✓ All files marked for classification reprocessing")
    
    show_status()


def after_description_improvement():
    """Run this after improving description generation"""
    print("\n🔧 UPDATING DESCRIPTIONS")
    print("=" * 60)
    
    # Get current version from config
    current_version = config.get_cache_version('descriptions')
    print(f"Current description version: {current_version}")
    
    # Increment version
    major, minor = current_version.split('.')
    new_version = f"{major}.{int(minor) + 1}"
    
    # Update in YAML
    config.update_cache_version('descriptions', new_version)
    print(f"✓ Description version updated to {new_version}")
    
    # Mark files for reprocessing
    cache.mark_for_reprocessing('descriptions', pattern='*.pdf')
    print("✓ All files marked for description reprocessing")
    
    show_status()


def after_disease_detection_improvement():
    """Run this after improving disease detection"""
    print("\n🔧 UPDATING DISEASE DETECTION")
    print("=" * 60)
    
    # Get current version
    current_version = config.get_cache_version('diseases')
    print(f"Current disease detector version: {current_version}")
    
    # Increment version
    major, minor = current_version.split('.')
    new_version = f"{major}.{int(minor) + 1}"
    
    # Update in YAML
    config.update_cache_version('diseases', new_version)
    print(f"✓ Disease detector version updated to {new_version}")
    
    # Mark for reprocessing
    cache.mark_for_reprocessing('diseases', pattern='*.pdf')
    print("✓ All files marked for disease reprocessing")
    
    show_status()


def major_version_update(component: str):
    """Increment major version for significant changes"""
    print(f"\n🔄 MAJOR UPDATE FOR {component.upper()}")
    print("=" * 60)
    
    # Get current version
    current_version = config.get_cache_version(component)
    print(f"Current {component} version: {current_version}")
    
    # Increment major version
    major, minor = current_version.split('.')
    new_version = f"{int(major) + 1}.0"
    
    # Update in YAML
    config.update_cache_version(component, new_version)
    print(f"✓ {component} version updated to {new_version} (MAJOR)")
    
    # Clear entire cache for this component
    cache.clear_cache(component)
    print(f"✓ Cleared all {component} cache")
    
    show_status()


def clear_marks():
    """Clear all reprocessing marks after batch is complete"""
    print("\n🧹 CLEARING REPROCESSING MARKS")
    print("=" * 60)
    
    cache.clear_reprocess_marks()
    print("✓ All reprocessing marks cleared")
    print("  Files will now use cache normally")


def show_status():
    """Show current cache status"""
    stats = cache.get_statistics_detailed()
    
    print("\n📊 CACHE STATUS")
    print("=" * 60)
    
    print("\nCurrent component versions (from cache_settings.yaml):")
    for comp, version in stats['current_versions'].items():
        queue_count = stats['reprocess_queue'].get(comp, 0)
        status = f"({queue_count} patterns marked)" if queue_count > 0 else "(up to date)"
        print(f"  - {comp}: v{version} {status}")
    
    print(f"\nCache performance:")
    print(f"  - Hit rate: {stats['hit_rate']}")
    print(f"  - API calls saved: {stats['api_calls_saved']:,}")
    print(f"  - Time saved: {stats['time_saved_minutes']} minutes")
    print(f"  - Cache size: {stats['cache_size_mb']} MB")
    
    print(f"\nCache configuration:")
    config_info = stats.get('configuration', {})
    print(f"  - Storage format: {config_info.get('storage_format', 'unknown')}")
    print(f"  - Compression: {config_info.get('compression', False)}")
    print(f"  - Hash algorithm: {config_info.get('hash_algorithm', 'unknown')}")
    print(f"  - Memory cache: {config_info.get('memory_cache', False)}")
    print(f"  - Batch writes: {config_info.get('batch_writes', False)}")
    
    if any(stats['reprocess_queue'].values()):
        print("\n⚠️  Files are marked for reprocessing!")
        print("   Run your extraction script to process them.")


def reset_specific_component(component='drugs'):
    """Force complete reprocessing of a specific component"""
    print(f"\n🔄 RESETTING {component.upper()}")
    print("=" * 60)
    
    # Clear the entire component cache
    cache.clear_cache(component)
    print(f"✓ Cleared all {component} cache")
    
    # Mark all for reprocessing
    cache.mark_for_reprocessing(component, pattern='*.pdf')
    print(f"✓ All files marked for {component} reprocessing")


def apply_reprocess_patterns():
    """Apply reprocessing patterns from configuration"""
    print("\n📋 APPLYING REPROCESS PATTERNS FROM CONFIG")
    print("=" * 60)
    
    reprocess_config = config.get_reprocess_config()
    
    if not reprocess_config.get('enabled', False):
        print("❌ Reprocessing patterns are disabled in configuration")
        return
    
    patterns = reprocess_config.get('patterns', None)
    print(f"Found {len(patterns)} patterns to apply:")
    
    for pattern in patterns:
        print(f"  - {pattern}")
        # Apply to all cache types
        for cache_type in ['drugs', 'classification', 'descriptions', 'diseases']:
            cache.mark_for_reprocessing(cache_type, pattern)
    
    print("\n✓ Patterns applied to all cache types")
    show_status()


def check_cache_health():
    """Check cache health and configuration"""
    print("\n🏥 CACHE HEALTH CHECK")
    print("=" * 60)
    
    # Check storage settings
    storage_config = config.get_cache_storage_config()
    print("\nStorage Configuration:")
    print(f"  - Base directory: {storage_config.get('base_directory', None)}")
    print(f"  - Max size: {storage_config.get('max_size_gb', 0)} GB")
    print(f"  - Cleanup policy: {storage_config.get('cleanup_policy', None)}")
    print(f"  - Compression: {storage_config.get('compression', None)} (level {storage_config.get('compression_level', None)})")
    print(f"  - File format: {storage_config.get('file_format', None)}")
    
    # Check behavior settings
    behavior_config = config.get_cache_behavior_config()
    print("\nBehavior Configuration:")
    print(f"  - Monitor hit rate: {behavior_config.get('monitor_hit_rate', None)}")
    print(f"  - Min hit rate threshold: {behavior_config.get('min_hit_rate_threshold', None) * 100}%")
    print(f"  - Cache negative results: {behavior_config.get('cache_negative_results', None)}")
    print(f"  - Respect file changes: {behavior_config.get('respect_file_changes', None)}")
    
    # Check expiration settings
    print("\nExpiration Settings (days):")
    expiration = behavior_config.get('expiration_days', None)
    for cache_type, days in expiration.items():
        print(f"  - {cache_type}: {days} days")
    
    # Check current statistics
    stats = cache.get_statistics()
    current_hit_rate = float(stats['hit_rate'].rstrip('%')) / 100
    
    print(f"\nCurrent Performance:")
    print(f"  - Hit rate: {stats['hit_rate']}")
    
    if current_hit_rate < behavior_config.get('min_hit_rate_threshold', None):
        print(f"  ⚠️  WARNING: Hit rate below threshold!")
    else:
        print(f"  ✓ Hit rate is healthy")
    
    # Check for reset triggers
    print("\nReset Triggers:")
    any_resets = False
    if config.should_reset_cache('all'):
        print("  ⚠️  RESET ALL is enabled!")
        any_resets = True
    
    for cache_type in ['drugs', 'classification', 'descriptions', 'diseases', 'clinical_trials', 'abbreviations']:
        if config.should_reset_cache(cache_type):
            print(f"  ⚠️  Reset {cache_type} is enabled!")
            any_resets = True
    
    if not any_resets:
        print("  ✓ No reset triggers active")


# ==============================================================================
# MAIN MENU
# ==============================================================================

if __name__ == "__main__":
    print("\n🚀 RARE DISEASE DOCUMENT CACHE MANAGER")
    print("     Now using cache_settings.yaml configuration")
    print("=" * 60)
    
    # Uncomment the action you want to perform:
    
    # After improving drug detection (e.g., fixed RAvulizumab → Ravulizumab)
    # after_drug_detection_improvement()
    
    # After improving classification prompts
    # after_classification_improvement()
    
    # After improving description generation
    # after_description_improvement()
    
    # After improving disease detection
    # after_disease_detection_improvement()
    
    # Major version update (significant changes)
    # major_version_update('drugs')
    
    # Clear all marks after reprocessing is done
    # clear_marks()
    
    # Force complete reprocessing of drugs
    # reset_specific_component('drugs')
    
    # Apply patterns from configuration file
    # apply_reprocess_patterns()
    
    # Check cache health and configuration
    check_cache_health()
    
    # Just show current status
    show_status()