#!/usr/bin/env python3
"""
Incremental Saving Feature Test
================================
Quickly verify that incremental saving is installed and configured correctly
"""

import os
import sys

def test_incremental_saving():
    """Test if incremental saving features are present"""
    
    print("="*70)
    print("INCREMENTAL SAVING FEATURE TEST")
    print("="*70)
    
    base_path = "FDA/fda_syncher/downloaders"
    results = {}
    
    # Test 1: Check adverse_events.py
    print("\n[Test 1] Checking adverse_events.py for incremental saving...")
    ae_file = os.path.join(base_path, "adverse_events.py")
    
    if not os.path.exists(ae_file):
        print(f"❌ File not found: {ae_file}")
        results['adverse_events'] = False
    else:
        with open(ae_file, 'r') as f:
            content = f.read()
        
        # Check for key features
        has_progress = 'progress_file' in content or '.progress' in content
        has_load_progress = '_load_progress' in content or 'load_progress' in content
        has_save_progress = '_save_progress' in content or 'SAVED' in content
        has_batch_saving = 'batch_size' in content or 'batch' in content
        
        if has_progress and has_load_progress and has_save_progress:
            print("✅ Incremental saving DETECTED")
            print(f"   - Progress tracking: {'✓' if has_progress else '✗'}")
            print(f"   - Load progress: {'✓' if has_load_progress else '✗'}")
            print(f"   - Save checkpoints: {'✓' if has_save_progress else '✗'}")
            print(f"   - Batch saving: {'✓' if has_batch_saving else '✗'}")
            results['adverse_events'] = True
        else:
            print("❌ Incremental saving NOT FOUND")
            print("   Using original version without resume capability")
            results['adverse_events'] = False
    
    # Test 2: Check labels.py
    print("\n[Test 2] Checking labels.py for incremental saving...")
    labels_file = os.path.join(base_path, "labels.py")
    
    if not os.path.exists(labels_file):
        print(f"❌ File not found: {labels_file}")
        results['labels'] = False
    else:
        with open(labels_file, 'r') as f:
            content = f.read()
        
        has_progress = 'progress_file' in content or '.progress' in content
        has_load_progress = '_load_progress' in content or 'load_progress' in content
        has_save_progress = '_save_progress' in content or 'SAVED' in content
        has_search_state = 'search_state' in content or 'completed_fields' in content
        
        if has_progress and has_load_progress and has_save_progress:
            print("✅ Incremental saving DETECTED")
            print(f"   - Progress tracking: {'✓' if has_progress else '✗'}")
            print(f"   - Load progress: {'✓' if has_load_progress else '✗'}")
            print(f"   - Save checkpoints: {'✓' if has_save_progress else '✗'}")
            print(f"   - Search state: {'✓' if has_search_state else '✗'}")
            results['labels'] = True
        else:
            print("❌ Incremental saving NOT FOUND")
            print("   Using original version without resume capability")
            results['labels'] = False
    
    # Test 3: Check approval_packages.py (should already be good)
    print("\n[Test 3] Checking approval_packages.py (already incremental)...")
    ap_file = os.path.join(base_path, "approval_packages.py")
    
    if not os.path.exists(ap_file):
        print(f"❌ File not found: {ap_file}")
        results['approval_packages'] = False
    else:
        with open(ap_file, 'r') as f:
            content = f.read()
        
        has_skip = 'skipped' in content and 'exists' in content
        has_per_drug = 'drug_dir' in content
        
        if has_skip:
            print("✅ Already incremental (per-drug saving)")
            results['approval_packages'] = True
        else:
            print("⚠️  May not have skip logic")
            results['approval_packages'] = False
    
    # Test 4: Check for progress directories
    print("\n[Test 4] Checking for progress directories...")
    progress_dirs = [
        "FDA_DATA/adverse_events/.progress",
        "FDA_DATA/labels/.progress"
    ]
    
    for pdir in progress_dirs:
        if os.path.exists(pdir):
            print(f"✅ {pdir} exists")
        else:
            print(f"⚠️  {pdir} will be created on first run")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nComponents Tested: {total}")
    print(f"Incremental Saving: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        print("   Incremental saving is installed and ready!")
        print("\n   Next: Run 'python test_resume.py' to test interruption recovery")
        return True
    elif passed > 0:
        print(f"\n⚠️  PARTIAL: {passed}/{total} components have incremental saving")
        print("\n   Installed:")
        for name, status in results.items():
            if status:
                print(f"     ✓ {name}")
        print("\n   Missing:")
        for name, status in results.items():
            if not status:
                print(f"     ✗ {name}")
        print("\n   Install improved versions for missing components")
        return False
    else:
        print("\n❌ NO INCREMENTAL SAVING FOUND")
        print("   You're using the original version")
        print("\n   Install improved versions:")
        print("     1. Copy adverse_events_IMPROVED.py -> adverse_events.py")
        print("     2. Copy labels_IMPROVED.py -> labels.py")
        return False

if __name__ == "__main__":
    success = test_incremental_saving()
    sys.exit(0 if success else 1)