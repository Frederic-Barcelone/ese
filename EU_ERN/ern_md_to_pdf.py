#!/usr/bin/env python3
"""
ERN Markdown to PDF Converter
=============================

Converts curated ERN markdown files to PDFs with rich embedded metadata.
Reads entity data from ern_content_curator.py output.

PIPELINE:
  1. ern_scraper.py          → Raw .md files
  2. ern_content_curator.py  → Curated .md + extracted entities
  3. THIS SCRIPT             → PDFs with embedded metadata

Requirements:
    pip install reportlab pypdf pyyaml

Usage:
    python ern_md_to_pdf.py

Author: ERN RAG Pipeline
Date: 2025-11
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import yaml
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, 
    Table, TableStyle
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from pypdf import PdfReader, PdfWriter


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Input folders
    "curated_folder": "EU_ERN_DATA/rag_content_curated",
    "curation_output": "EU_ERN_DATA/curation_output",
    
    # Output folder  
    "pdf_output": "EU_ERN_DATA/rag_content_pdf",
    
    # Network config
    "network_config": "ern_config.json",
    
    # Options
    "add_metadata_page": True,
    "verbose": True,
    "overwrite_existing": True,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json(filepath: Path) -> Optional[Dict]:
    """Load JSON file."""
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None


def load_network_config(config_path: str) -> Dict:
    """Load network configuration."""
    path = Path(config_path)
    return load_json(path) or {}


def get_network_info(network_config: Dict, network_id: str) -> Dict:
    """Get network info."""
    return network_config.get("networks", {}).get(network_id, {})


def extract_network_id(filename: str) -> str:
    """Extract network ID from filename."""
    parts = filename.split('_')
    if parts:
        nid = parts[0]
        if nid.startswith("ERN") or nid in ["ERKNet", "MetabERN", "VASCERN", "ERNICA"]:
            return nid
    return "Unknown"


def read_markdown_file(filepath: Path) -> Dict:
    """Read markdown file with YAML frontmatter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    result = {
        "filename": filepath.name,
        "filepath": str(filepath),
        "url": "",
        "title": filepath.stem,
        "scraped_date": "",
        "content": content,
    }
    
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
                result["url"] = fm.get("url", "")
                result["title"] = fm.get("title", filepath.stem)
                result["scraped_date"] = str(fm.get("scraped_date", ""))
                result["content"] = parts[2].strip()
            except yaml.YAMLError:
                pass
    
    return result


# ============================================================================
# PDF GENERATOR
# ============================================================================

class PDFGenerator:
    """Generates PDFs with embedded metadata."""
    
    def __init__(self):
        self.styles = self._create_styles()
    
    def _create_styles(self):
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='ERNTitle', parent=styles['Title'],
            fontSize=16, spaceAfter=20,
            textColor=colors.HexColor('#1a5276'),
            alignment=TA_CENTER,
        ))
        
        styles.add(ParagraphStyle(
            name='ERNHeading1', parent=styles['Heading1'],
            fontSize=14, spaceBefore=15, spaceAfter=10,
            textColor=colors.HexColor('#2874a6'),
        ))
        
        styles.add(ParagraphStyle(
            name='ERNHeading2', parent=styles['Heading2'],
            fontSize=12, spaceBefore=12, spaceAfter=8,
            textColor=colors.HexColor('#3498db'),
        ))
        
        styles.add(ParagraphStyle(
            name='ERNBody', parent=styles['Normal'],
            fontSize=10, leading=14,
            spaceBefore=4, spaceAfter=4,
            alignment=TA_JUSTIFY,
        ))
        
        styles.add(ParagraphStyle(
            name='ERNMeta', parent=styles['Normal'],
            fontSize=9, textColor=colors.HexColor('#566573'),
            spaceBefore=2, spaceAfter=2,
        ))
        
        styles.add(ParagraphStyle(
            name='ERNMetaHeader', parent=styles['Heading3'],
            fontSize=11, textColor=colors.HexColor('#1a5276'),
            spaceBefore=10, spaceAfter=5,
        ))
        
        return styles
    
    def _escape(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    def _create_metadata_page(self, file_data: Dict, entities: Dict, 
                               network_info: Dict) -> List:
        """Create metadata summary page."""
        elements = []
        
        elements.append(Paragraph("Document Metadata", self.styles['ERNTitle']))
        elements.append(Spacer(1, 15))
        
        # Basic info table
        network_name = network_info.get("name", extract_network_id(file_data["filename"]))
        disease_area = network_info.get("disease_area", network_info.get("short_description", ""))
        
        table_data = [
            ["Field", "Value"],
            ["Title", self._escape(file_data.get("title", ""))[:70]],
            ["Network", self._escape(network_name)[:70]],
            ["Disease Area", self._escape(disease_area)[:70]],
            ["Document Type", self._escape(entities.get("content_type", ""))],
            ["Source URL", self._escape(file_data.get("url", ""))[:70]],
            ["Quality Score", str(entities.get("quality_score", "N/A"))],
        ]
        
        table = Table(table_data, colWidths=[3*cm, 12*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 15))
        
        # Reason
        if entities.get("reason"):
            elements.append(Paragraph(
                f"<b>Classification:</b> {self._escape(entities.get('reason', ''))}",
                self.styles['ERNMeta']
            ))
            elements.append(Spacer(1, 10))
        
        # Experts
        experts = entities.get("experts", [])
        if experts:
            elements.append(Paragraph("Experts/Staff", self.styles['ERNMetaHeader']))
            for exp in experts[:15]:
                if isinstance(exp, dict):
                    name = exp.get("name", "")
                    role = f" - {exp.get('role', '')}" if exp.get('role') else ""
                    title = f" ({exp.get('title', '')})" if exp.get('title') else ""
                    inst = f" @ {exp.get('institution', '')}" if exp.get('institution') else ""
                    elements.append(Paragraph(
                        f"• {self._escape(name)}{self._escape(title)}{self._escape(role)}{self._escape(inst)}",
                        self.styles['ERNMeta']
                    ))
                else:
                    elements.append(Paragraph(f"• {self._escape(str(exp))}", self.styles['ERNMeta']))
        
        # Institutions
        institutions = entities.get("institutions", [])
        if institutions:
            elements.append(Paragraph("Institutions", self.styles['ERNMetaHeader']))
            for inst in institutions[:10]:
                if isinstance(inst, dict):
                    name = inst.get("name", "")
                    city = f", {inst.get('city', '')}" if inst.get('city') else ""
                    country = f", {inst.get('country', '')}" if inst.get('country') else ""
                    itype = f" [{inst.get('type', '')}]" if inst.get('type') else ""
                    elements.append(Paragraph(
                        f"• {self._escape(name)}{self._escape(city)}{self._escape(country)}{self._escape(itype)}",
                        self.styles['ERNMeta']
                    ))
                else:
                    elements.append(Paragraph(f"• {self._escape(str(inst))}", self.styles['ERNMeta']))
        
        # Diseases
        diseases = entities.get("diseases", [])
        if diseases:
            elements.append(Paragraph("Diseases/Conditions", self.styles['ERNMetaHeader']))
            disease_list = []
            for d in diseases[:15]:
                if isinstance(d, dict):
                    name = d.get("name", "")
                    cat = f" [{d.get('category', '')}]" if d.get('category') else ""
                    disease_list.append(f"{name}{cat}")
                else:
                    disease_list.append(str(d))
            elements.append(Paragraph(self._escape(", ".join(disease_list)), self.styles['ERNMeta']))
        
        # Registries
        registries = entities.get("registries", [])
        if registries:
            elements.append(Paragraph("Registries/Databases", self.styles['ERNMetaHeader']))
            for reg in registries[:10]:
                if isinstance(reg, dict):
                    name = reg.get("name", "")
                    acronym = f" ({reg.get('acronym', '')})" if reg.get('acronym') else ""
                    rtype = f" [{reg.get('type', '')}]" if reg.get('type') else ""
                    focus = f" - {reg.get('focus_area', '')}" if reg.get('focus_area') else ""
                    elements.append(Paragraph(
                        f"• {self._escape(name)}{self._escape(acronym)}{self._escape(rtype)}{self._escape(focus)}",
                        self.styles['ERNMeta']
                    ))
                else:
                    elements.append(Paragraph(f"• {self._escape(str(reg))}", self.styles['ERNMeta']))
        
        # Projects
        projects = entities.get("projects", [])
        if projects:
            elements.append(Paragraph("Projects", self.styles['ERNMetaHeader']))
            for proj in projects[:8]:
                if isinstance(proj, dict):
                    name = proj.get("name", "")
                    acronym = f" ({proj.get('acronym', '')})" if proj.get('acronym') else ""
                    ptype = f" [{proj.get('type', '')}]" if proj.get('type') else ""
                    elements.append(Paragraph(
                        f"• {self._escape(name)}{self._escape(acronym)}{self._escape(ptype)}",
                        self.styles['ERNMeta']
                    ))
                else:
                    elements.append(Paragraph(f"• {self._escape(str(proj))}", self.styles['ERNMeta']))
        
        # Guidelines
        guidelines = entities.get("guidelines", [])
        if guidelines:
            elements.append(Paragraph("Guidelines", self.styles['ERNMetaHeader']))
            for gl in guidelines[:8]:
                if isinstance(gl, dict):
                    title = gl.get("title", "")
                    gtype = f" [{gl.get('type', '')}]" if gl.get('type') else ""
                    area = f" - {gl.get('disease_area', '')}" if gl.get('disease_area') else ""
                    elements.append(Paragraph(
                        f"• {self._escape(title)}{self._escape(gtype)}{self._escape(area)}",
                        self.styles['ERNMeta']
                    ))
                else:
                    elements.append(Paragraph(f"• {self._escape(str(gl))}", self.styles['ERNMeta']))
        
        # Educational resources
        edu = entities.get("educational_resources", [])
        if edu:
            elements.append(Paragraph("Educational Resources", self.styles['ERNMetaHeader']))
            for e in edu[:8]:
                if isinstance(e, dict):
                    title = e.get("title", "")
                    etype = f" [{e.get('type', '')}]" if e.get('type') else ""
                    elements.append(Paragraph(
                        f"• {self._escape(title)}{self._escape(etype)}",
                        self.styles['ERNMeta']
                    ))
                else:
                    elements.append(Paragraph(f"• {self._escape(str(e))}", self.styles['ERNMeta']))
        
        # Valuable content
        valuable = entities.get("valuable_content", [])
        if valuable:
            elements.append(Paragraph("Key Content", self.styles['ERNMetaHeader']))
            elements.append(Paragraph(
                self._escape(", ".join(str(v) for v in valuable[:15])),
                self.styles['ERNMeta']
            ))
        
        elements.append(PageBreak())
        return elements
    
    def _parse_content(self, content: str) -> List:
        """Convert markdown to PDF elements."""
        elements = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line:
                elements.append(Spacer(1, 6))
                continue
            
            # Headers
            if line.startswith('####'):
                text = self._escape(line.lstrip('#').strip())
                elements.append(Paragraph(text, self.styles['ERNHeading2']))
            elif line.startswith('###'):
                text = self._escape(line.lstrip('#').strip())
                elements.append(Paragraph(text, self.styles['ERNHeading2']))
            elif line.startswith('##'):
                text = self._escape(line.lstrip('#').strip())
                elements.append(Paragraph(text, self.styles['ERNHeading1']))
            elif line.startswith('#'):
                text = self._escape(line.lstrip('#').strip())
                elements.append(Paragraph(text, self.styles['ERNTitle']))
            # Blockquote
            elif line.startswith('>'):
                text = self._escape(line.lstrip('>').strip())
                elements.append(Paragraph(f"<i>{text}</i>", self.styles['ERNBody']))
            # List
            elif line.startswith('- '):
                text = self._process_inline(line[2:])
                elements.append(Paragraph(f"• {text}", self.styles['ERNBody']))
            # Regular
            else:
                text = self._process_inline(line)
                elements.append(Paragraph(text, self.styles['ERNBody']))
        
        return elements
    
    def _process_inline(self, text: str) -> str:
        """Process inline markdown."""
        text = self._escape(text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'<u>\1</u>', text)
        return text
    
    def generate_pdf(self, file_data: Dict, entities: Dict, 
                     output_path: Path, network_info: Dict,
                     add_metadata_page: bool = True):
        """Generate PDF with metadata."""
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=2.5*cm, rightMargin=2.5*cm,
            topMargin=2.5*cm, bottomMargin=2.5*cm,
            title=file_data.get("title", ""),
            author=network_info.get("name", "ERN"),
            subject=network_info.get("disease_area", ""),
        )
        
        elements = []
        
        if add_metadata_page:
            elements.extend(self._create_metadata_page(file_data, entities, network_info))
        
        elements.extend(self._parse_content(file_data["content"]))
        
        doc.build(elements)
        
        # Add extended metadata
        self._add_extended_metadata(output_path, file_data, entities, network_info)
    
    def _add_extended_metadata(self, pdf_path: Path, file_data: Dict, 
                                entities: Dict, network_info: Dict):
        """Add custom metadata to PDF."""
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        
        for page in reader.pages:
            writer.add_page(page)
        
        def get_names(items: List, key: str = "name", limit: int = 10) -> str:
            names = []
            for item in items[:limit]:
                if isinstance(item, dict):
                    names.append(str(item.get(key, "")))
                else:
                    names.append(str(item))
            return ", ".join(n for n in names if n)
        
        experts = get_names(entities.get("experts", []))
        institutions = get_names(entities.get("institutions", []))
        diseases = get_names(entities.get("diseases", []))
        registries = get_names(entities.get("registries", []))
        projects = get_names(entities.get("projects", []), key="acronym")
        guidelines = get_names(entities.get("guidelines", []), key="title")
        
        network_id = extract_network_id(file_data["filename"])
        
        metadata = {
            # Standard
            "/Title": str(file_data.get("title", ""))[:200],
            "/Author": str(network_info.get("name", network_id))[:100],
            "/Subject": str(network_info.get("disease_area", ""))[:200],
            "/Keywords": get_names(entities.get("valuable_content", []), limit=20),
            "/Creator": "ERN RAG Pipeline",
            "/Producer": "ERN MD to PDF Converter v2.0",
            
            # Custom ERN metadata
            "/ERNNetworkID": network_id,
            "/ERNNetworkName": str(network_info.get("name", ""))[:100],
            "/ERNDiseaseArea": str(network_info.get("disease_area", ""))[:200],
            "/ERNDocumentType": str(entities.get("content_type", ""))[:50],
            "/ERNQualityScore": str(entities.get("quality_score", "")),
            "/ERNSourceURL": str(file_data.get("url", ""))[:200],
            "/ERNScrapedDate": str(file_data.get("scraped_date", ""))[:30],
            "/ERNConversionDate": datetime.now().isoformat()[:30],
            
            # Extracted entities
            "/ERNExperts": experts[:500],
            "/ERNInstitutions": institutions[:500],
            "/ERNDiseases": diseases[:500],
            "/ERNRegistries": registries[:300],
            "/ERNProjects": projects[:300],
            "/ERNGuidelines": guidelines[:300],
        }
        
        writer.add_metadata(metadata)
        
        with open(pdf_path, 'wb') as f:
            writer.write(f)


# ============================================================================
# MAIN CONVERTER
# ============================================================================

class ERNPDFConverter:
    """Main converter class."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.curated_folder = Path(config["curated_folder"])
        self.curation_output = Path(config["curation_output"])
        self.pdf_output = Path(config["pdf_output"])
        
        self.network_config = load_network_config(config.get("network_config", ""))
        self.curator_results = self._load_curator_results()
        self.pdf_gen = PDFGenerator()
        
        self.converted = 0
        self.skipped = 0
        self.errors = []
    
    def _load_curator_results(self) -> Dict:
        """Load results from curator."""
        results = {}
        
        # Classification results
        class_file = self.curation_output / "classification_results.json"
        if class_file.exists():
            data = load_json(class_file)
            if data and "files" in data:
                for item in data["files"]:
                    filename = item.get("filename", "")
                    results[filename] = item
                print(f"  Loaded classification for {len(results)} files")
        
        # Individual entity files
        for entity_type in ["experts", "institutions", "diseases", "projects", 
                           "registries", "guidelines", "educational_resources"]:
            entity_file = self.curation_output / f"{entity_type}.json"
            if entity_file.exists():
                data = load_json(entity_file)
                if data:
                    for item in data:
                        source = item.get("_source_file", "")
                        if source and source in results:
                            if entity_type not in results[source]:
                                results[source][entity_type] = []
                            results[source][entity_type].append(item)
        
        return results
    
    def _get_entities_for_file(self, filename: str) -> Dict:
        """Get entities for a file."""
        return self.curator_results.get(filename, {})
    
    def convert_file(self, filepath: Path) -> bool:
        """Convert single file."""
        filename = filepath.name
        output_path = self.pdf_output / f"{filepath.stem}.pdf"
        
        if not self.config["overwrite_existing"] and output_path.exists():
            self.skipped += 1
            return True
        
        try:
            file_data = read_markdown_file(filepath)
            entities = self._get_entities_for_file(filename)
            
            network_id = extract_network_id(filename)
            network_info = get_network_info(self.network_config, network_id)
            
            self.pdf_gen.generate_pdf(
                file_data=file_data,
                entities=entities,
                output_path=output_path,
                network_info=network_info,
                add_metadata_page=self.config["add_metadata_page"]
            )
            
            if self.config["verbose"]:
                expert_count = len(entities.get("experts", []))
                disease_count = len(entities.get("diseases", []))
                registry_count = len(entities.get("registries", []))
                print(f"    ✓ PDF ({expert_count} experts, {disease_count} diseases, {registry_count} registries)")
            
            self.converted += 1
            return True
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            self.errors.append({"file": filename, "error": str(e)})
            return False
    
    def run(self):
        """Run conversion."""
        self.pdf_output.mkdir(parents=True, exist_ok=True)
        
        files = sorted(self.curated_folder.glob("*.md"))
        
        if not files:
            print(f"No files in {self.curated_folder}")
            return
        
        print("=" * 60)
        print("ERN Markdown to PDF Converter")
        print("=" * 60)
        print(f"Source:  {self.curated_folder}")
        print(f"Output:  {self.pdf_output}")
        print(f"Files:   {len(files)}")
        print(f"Curator: {self.curation_output}")
        print("=" * 60)
        
        for i, filepath in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {filepath.name}")
            self.convert_file(filepath)
        
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETE")
        print("=" * 60)
        print(f"  Converted: {self.converted}")
        print(f"  Skipped:   {self.skipped}")
        print(f"  Errors:    {len(self.errors)}")
        print(f"\n  Output:    {self.pdf_output}/")
        print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    converter = ERNPDFConverter(CONFIG)
    converter.run()


if __name__ == "__main__":
    main()