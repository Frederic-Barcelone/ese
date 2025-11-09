"""
Copy Important FDA Package Files for RAG System - DUAL CHECK VERSION
=====================================================================
Copies files from FDA_DATA packages to FDA_TH with flattened naming structure.

MODIFIED: Skips files if they exist in EITHER:
  - FDA_TH/ (target directory)
  - FDA_TH_in/ (check directory)

From each package, copies files from:
- approval_letter/
- label/
- other_review/

Output naming: FDA_{package_name}_{original_filename}
"""

import os
import shutil
from pathlib import Path


# Configuration
SOURCE_DIR = "FDA_DATA/approval_packages"  # Source directory
TARGET_DIR = "FDA_TH"                       # Target copy directory
CHECK_DIR = "FDA_TH_in"                     # Also check if file exists here (SKIP if found)

# Folders to copy from each package
FOLDERS_TO_COPY = [
    "approval_letter",
    "label", 
    "other_review"
]

# Files to skip (exact filenames)
FILES_TO_SKIP = [
    "Other_Review(s).pdf",
    "Proprietary_Name_Review(s).pdf"
]


def sanitize_filename(filename):
    """Remove or replace problematic characters in filenames"""
    filename = filename.replace(' ', '_')
    filename = filename.replace('(', '').replace(')', '')
    filename = filename.replace('[', '').replace(']', '')
    return filename


def copy_package_files(package_path, target_dir, check_dir):
    """
    Copy files from specified folders in a package to target directory.
    Skips if file exists in either target_dir OR check_dir.
    
    Args:
        package_path: Path to the package directory
        target_dir: Destination directory
        check_dir: Additional directory to check for existing files
        
    Returns:
        Dictionary with copy statistics
    """
    package_name = package_path.name
    stats = {
        'package': package_name,
        'copied': 0,
        'skipped': 0,
        'skipped_in_target': 0,
        'skipped_in_check': 0,
        'errors': 0,
        'files': []
    }
    
    print(f"\n📦 Processing: {package_name}")
    
    for folder_name in FOLDERS_TO_COPY:
        folder_path = package_path / folder_name
        
        if not folder_path.exists():
            print(f"  ⚠️  Folder not found: {folder_name}")
            continue
        
        # Get all files in this folder (including nested)
        files = list(folder_path.rglob('*'))
        files = [f for f in files if f.is_file()]
        
        if not files:
            print(f"  📁 {folder_name}/: No files")
            continue
        
        print(f"  📁 {folder_name}/: {len(files)} file(s)")
        
        for file_path in files:
            try:
                # Skip files in the exclusion list
                original_name = file_path.name
                if original_name in FILES_TO_SKIP:
                    print(f"    🚫 Skipped (excluded): {original_name}")
                    stats['skipped'] += 1
                    continue
                
                # Create new filename: FDA_packagename_originalfilename
                new_name = f"FDA_{package_name}_{original_name}"
                new_name = sanitize_filename(new_name)
                
                target_path = target_dir / new_name
                check_path = check_dir / new_name
                
                # ⭐ CHECK 1: Does file exist in TARGET directory (FDA_TH)?
                if target_path.exists():
                    print(f"    ⏭️  Skipped (exists in {TARGET_DIR}): {new_name}")
                    stats['skipped'] += 1
                    stats['skipped_in_target'] += 1
                    continue
                
                # ⭐ CHECK 2: Does file exist in CHECK directory (FDA_TH_in)?
                if check_path.exists():
                    print(f"    ⏭️  Skipped (exists in {CHECK_DIR}): {new_name}")
                    stats['skipped'] += 1
                    stats['skipped_in_check'] += 1
                    continue
                
                # File doesn't exist in either location - COPY IT
                shutil.copy2(file_path, target_path)
                
                # Get file size in MB
                size_mb = file_path.stat().st_size / (1024 * 1024)
                
                print(f"    ✅ Copied: {new_name} ({size_mb:.2f} MB)")
                stats['copied'] += 1
                stats['files'].append({
                    'original': str(file_path),
                    'new_name': new_name,
                    'size_mb': size_mb
                })
                
            except Exception as e:
                print(f"    ❌ Error copying {file_path.name}: {e}")
                stats['errors'] += 1
    
    return stats


def main():
    """Main execution function"""
    
    print("="*70)
    print("FDA PACKAGE FILE COPIER - DUAL CHECK VERSION")
    print("="*70)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print(f"Check: {CHECK_DIR} (skip if exists here)")
    print(f"Folders: {', '.join(FOLDERS_TO_COPY)}")
    print(f"Excluded Files: {', '.join(FILES_TO_SKIP)}")
    print("="*70)
    
    # Validate source directory
    source_path = Path(SOURCE_DIR)
    if not source_path.exists():
        print(f"\n❌ ERROR: Source directory not found: {SOURCE_DIR}")
        print("Please update SOURCE_DIR in the script to match your directory structure.")
        return
    
    # Create target directory
    target_path = Path(TARGET_DIR)
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"\n✅ Target directory ready: {TARGET_DIR}")
    
    # Validate check directory (should exist for checking)
    check_path = Path(CHECK_DIR)
    if not check_path.exists():
        print(f"\n⚠️  WARNING: Check directory not found: {CHECK_DIR}")
        print(f"   Creating it now (files will only be checked in {TARGET_DIR})")
        check_path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"✅ Check directory exists: {CHECK_DIR}")
    
    # Find all therapeutic area subdirectories
    therapeutic_areas = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not therapeutic_areas:
        print(f"\n⚠️  No therapeutic area folders found in {SOURCE_DIR}")
        return
    
    print(f"\n🔍 Found {len(therapeutic_areas)} therapeutic area(s):")
    for area in therapeutic_areas:
        print(f"  - {area.name}")
    
    # Process all packages
    all_stats = []
    
    for area_path in therapeutic_areas:
        print(f"\n{'='*70}")
        print(f"THERAPEUTIC AREA: {area_path.name.upper()}")
        print(f"{'='*70}")
        
        # Get all package directories in this therapeutic area
        packages = [d for d in area_path.iterdir() if d.is_dir()]
        
        if not packages:
            print(f"  No packages found in {area_path.name}")
            continue
        
        print(f"Found {len(packages)} package(s)")
        
        for package_path in packages:
            stats = copy_package_files(package_path, target_path, check_path)
            all_stats.append(stats)
    
    # Print summary
    print("\n" + "="*70)
    print("COPY SUMMARY - DUAL CHECK")
    print("="*70)
    
    total_copied = sum(s['copied'] for s in all_stats)
    total_skipped = sum(s['skipped'] for s in all_stats)
    total_skipped_target = sum(s['skipped_in_target'] for s in all_stats)
    total_skipped_check = sum(s['skipped_in_check'] for s in all_stats)
    total_errors = sum(s['errors'] for s in all_stats)
    
    print(f"\nPackages Processed: {len(all_stats)}")
    print(f"Files Copied: {total_copied}")
    print(f"Files Skipped: {total_skipped}")
    print(f"  - Found in {TARGET_DIR}: {total_skipped_target}")
    print(f"  - Found in {CHECK_DIR}: {total_skipped_check}")
    print(f"Errors: {total_errors}")
    
    if all_stats:
        total_size = sum(
            f['size_mb'] 
            for s in all_stats 
            for f in s['files']
        )
        print(f"Total Size Copied: {total_size:.2f} MB")
    
    print(f"\n✅ Copy complete!")
    print(f"Files available in: {TARGET_DIR}")
    print("="*70 + "\n")
    
    # Create a manifest file
    manifest_path = target_path / "MANIFEST.txt"
    with open(manifest_path, 'w') as f:
        f.write("FDA Package Files - Copy Manifest (DUAL CHECK)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Created: {Path.cwd()}\n")
        f.write(f"Source: {SOURCE_DIR}\n")
        f.write(f"Target: {TARGET_DIR}\n")
        f.write(f"Check: {CHECK_DIR}\n")
        f.write(f"Copied: {total_copied} files\n")
        f.write(f"Skipped (in {TARGET_DIR}): {total_skipped_target} files\n")
        f.write(f"Skipped (in {CHECK_DIR}): {total_skipped_check} files\n")
        f.write(f"Total Size: {total_size:.2f} MB\n\n")
        
        f.write("Files by Package:\n")
        f.write("-"*70 + "\n\n")
        
        for stats in all_stats:
            if stats['copied'] > 0:
                f.write(f"Package: {stats['package']}\n")
                f.write(f"  Files copied: {stats['copied']}\n")
                for file_info in stats['files']:
                    f.write(f"  - {file_info['new_name']} ({file_info['size_mb']:.2f} MB)\n")
                f.write("\n")
    
    print(f"📄 Manifest created: {manifest_path}")


if __name__ == "__main__":
    main()