#!/usr/bin/env python3
"""
Copy FDA Guidance Library Files to Target Directory
====================================================
Copies all files from FDA_guidance_library and its subfolders
to the target directory with "FDA_" prefix.

Features:
- Flattens folder structure (all files go to one folder)
- Adds "FDA_" prefix to filenames
- Skips files that already exist in target
- Handles filename conflicts by adding subfolder name
- Progress tracking and summary

Usage:
    python copy_fda_guidance.py
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


# =============================================================================
# CONFIGURATION - EDIT THESE IF NEEDED
# =============================================================================

# Source folder containing FDA guidance documents
SOURCE_DIR = "FDA_guidance_library"

# Target folder where files will be copied
TARGET_DIR = "FDA_TH_in"

# Prefix to add to all filenames
PREFIX = "FDA_"

# Set to True to see what would be copied without actually copying
DRY_RUN = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sanitize_filename(filename):
    """Remove or replace problematic characters in filenames"""
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove parentheses and brackets
    filename = filename.replace('(', '').replace(')', '')
    filename = filename.replace('[', '').replace(']', '')
    # Remove double underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
    return filename


def get_unique_filename(target_dir, filename, subfolder_name=None):
    """
    Get a unique filename, adding subfolder name if there's a conflict.
    
    Args:
        target_dir: Target directory path
        filename: Proposed filename
        subfolder_name: Name of the source subfolder (for disambiguation)
    
    Returns:
        Unique filename
    """
    target_path = target_dir / filename
    
    if not target_path.exists():
        return filename
    
    # File exists - try adding subfolder name
    if subfolder_name:
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{subfolder_name}{ext}"
        new_filename = sanitize_filename(new_filename)
        target_path = target_dir / new_filename
        
        if not target_path.exists():
            return new_filename
    
    # Still exists - add counter
    counter = 1
    name, ext = os.path.splitext(filename)
    while True:
        new_filename = f"{name}_{counter}{ext}"
        target_path = target_dir / new_filename
        if not target_path.exists():
            return new_filename
        counter += 1


def copy_files(source_dir, target_dir, prefix, dry_run=False):
    """
    Copy all files from source directory (including subfolders) to target.
    
    Args:
        source_dir: Source directory path
        target_dir: Target directory path
        prefix: Prefix to add to filenames
        dry_run: If True, only print what would be done
    
    Returns:
        Dictionary with copy statistics
    """
    stats = {
        'copied': 0,
        'skipped_exists': 0,
        'errors': 0,
        'total_size_mb': 0,
        'files': []
    }
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Get all files recursively
    all_files = list(source_path.rglob('*'))
    all_files = [f for f in all_files if f.is_file()]
    
    # Filter out non-PDF files and special files
    files_to_copy = []
    for f in all_files:
        # Skip hidden files and special files
        if f.name.startswith('.') or f.name.startswith('_'):
            continue
        # Skip JSON files (progress tracking)
        if f.suffix.lower() == '.json':
            continue
        files_to_copy.append(f)
    
    print(f"\n📁 Found {len(files_to_copy)} files to process")
    print("-" * 60)
    
    for file_path in sorted(files_to_copy):
        try:
            # Get the subfolder name (category folder)
            relative_path = file_path.relative_to(source_path)
            subfolder_name = relative_path.parts[0] if len(relative_path.parts) > 1 else None
            
            # Create new filename with prefix
            original_name = file_path.name
            
            # Check if file already has FDA_ prefix (avoid double prefix)
            if original_name.upper().startswith('FDA_'):
                new_name = original_name
            else:
                new_name = f"{prefix}{original_name}"
            
            new_name = sanitize_filename(new_name)
            
            # Check if file already exists in target
            target_file = target_path / new_name
            
            if target_file.exists():
                print(f"  ⏭️  Skipped (exists): {new_name}")
                stats['skipped_exists'] += 1
                continue
            
            # Get unique filename if needed
            final_name = get_unique_filename(target_path, new_name, subfolder_name)
            final_target = target_path / final_name
            
            # Get file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if dry_run:
                print(f"  🔍 Would copy: {original_name} -> {final_name} ({size_mb:.2f} MB)")
            else:
                # Copy the file
                shutil.copy2(file_path, final_target)
                print(f"  ✅ Copied: {final_name} ({size_mb:.2f} MB)")
            
            stats['copied'] += 1
            stats['total_size_mb'] += size_mb
            stats['files'].append({
                'original': str(file_path),
                'new_name': final_name,
                'subfolder': subfolder_name,
                'size_mb': size_mb
            })
            
        except Exception as e:
            print(f"  ❌ Error: {file_path.name} - {e}")
            stats['errors'] += 1
    
    return stats


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    
    start_time = datetime.now()
    
    print("=" * 70)
    print("FDA GUIDANCE LIBRARY FILE COPIER")
    print("=" * 70)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print(f"Prefix: {PREFIX}")
    print(f"Mode: {'DRY RUN (no files will be copied)' if DRY_RUN else 'LIVE'}")
    print("=" * 70)
    
    # Validate source directory
    source_path = Path(SOURCE_DIR)
    if not source_path.exists():
        print(f"\n❌ ERROR: Source directory not found: {SOURCE_DIR}")
        print("\nPlease check that:")
        print("  1. You're running this script from the correct directory")
        print("  2. The FDA_guidance_library folder exists")
        print(f"\nCurrent directory: {Path.cwd()}")
        return
    
    # Create target directory if it doesn't exist
    target_path = Path(TARGET_DIR)
    if not DRY_RUN:
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"\n✅ Target directory ready: {TARGET_DIR}")
    else:
        print(f"\n🔍 DRY RUN - Target directory would be: {TARGET_DIR}")
    
    # Show source structure
    print("\n📂 Source structure:")
    subfolders = [d for d in source_path.iterdir() if d.is_dir()]
    for subfolder in sorted(subfolders):
        file_count = len(list(subfolder.rglob('*')))
        print(f"  - {subfolder.name}/ ({file_count} items)")
    
    # Copy files
    stats = copy_files(source_path, target_path, PREFIX, DRY_RUN)
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("COPY SUMMARY")
    print("=" * 70)
    print(f"\nDuration: {duration}")
    print(f"Files {'would be ' if DRY_RUN else ''}copied: {stats['copied']}")
    print(f"Files skipped (already exist): {stats['skipped_exists']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    
    if not DRY_RUN and stats['copied'] > 0:
        print("\n✅ Copy complete!")
        print(f"Files available in: {TARGET_DIR}")
        
        # Create a manifest file in target
        manifest_path = target_path / "FDA_GUIDANCE_MANIFEST.txt"
        with open(manifest_path, 'w') as f:
            f.write("FDA Guidance Library - Copy Manifest\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Source: {SOURCE_DIR}\n")
            f.write(f"Target: {TARGET_DIR}\n")
            f.write(f"Files copied: {stats['copied']}\n")
            f.write(f"Total size: {stats['total_size_mb']:.2f} MB\n\n")
            
            f.write("Files:\n")
            f.write("-" * 70 + "\n")
            
            # Group by subfolder
            by_subfolder = {}
            for file_info in stats['files']:
                sf = file_info['subfolder'] or 'root'
                if sf not in by_subfolder:
                    by_subfolder[sf] = []
                by_subfolder[sf].append(file_info)
            
            for subfolder, files in sorted(by_subfolder.items()):
                f.write(f"\n{subfolder}/\n")
                for file_info in files:
                    f.write(f"  - {file_info['new_name']} ({file_info['size_mb']:.2f} MB)\n")
        
        print(f"📄 Manifest created: {manifest_path}")
    
    elif DRY_RUN:
        print("\n🔍 DRY RUN complete - no files were copied")
        print("Set DRY_RUN = False to actually copy files")
    
    print("=" * 70 + "\n")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()