#!/usr/bin/env python3
"""Fix INSPIREHEP pattern to require 'INSPIRE' prefix"""

import re

file_path = "corpus_metadata/document_utils/entity_reference_patterns.py"

with open(file_path, 'r') as f:
    content = f.read()

# Fix: Remove the optional ? before INSPIRE
content = content.replace(
    r"'pattern': r'\b(?:INSPIRE(?:-HEP)?:?\s*)?\d{6,8}\b',",
    r"'pattern': r'\bINSPIRE(?:-HEP)?:?\s*\d{6,8}\b',"
)

with open(file_path, 'w') as f:
    f.write(content)

print("✅ Fixed INSPIREHEP pattern - now requires 'INSPIRE' prefix")
print("✅ This prevents matching standalone 6-8 digit numbers as INSPIREHEP")