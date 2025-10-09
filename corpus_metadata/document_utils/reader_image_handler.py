#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_image_handler.py
#
"""
Image Handler Module
===================

Handler for image files with OCR capabilities.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

from corpus_metadata.document_utils.reader_base import FileHandler
from corpus_metadata.document_utils.reader_ocr_processor import OCRProcessor
from corpus_metadata.document_utils.reader_exceptions import ExtractionError

logger = logging.getLogger(__name__)


class ImageHandler(FileHandler):
    """Handler for image files with OCR support."""
    
    def __init__(self, config: Any):
        """
        Initialize image handler with configuration.
        
        Args:
            config: ReaderConfig instance
        """
        super().__init__(config)
        self.ocr_processor = OCRProcessor(config.ocr_config) if config.enable_ocr else None
    
    def can_handle(self, file_extension: str) -> bool:
        """Check if this handler can process the file type."""
        return file_extension.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']
    
    def process(self, file_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image file and optionally extract text using OCR.
        
        Args:
            file_path: Path to image file
            metadata: Pre-extracted metadata
            
        Returns:
            Dictionary with image information and extracted text
        """
        result = metadata.copy()
        result['is_binary'] = True
        
        try:
            from PIL import Image
            
            # Open and analyze image
            with Image.open(file_path) as img:
                # Extract image properties
                result.update({
                    'image_width': img.width,
                    'image_height': img.height,
                    'image_format': img.format,
                    'image_mode': img.mode,
                    'image_megapixels': round((img.width * img.height) / 1_000_000, 2)
                })
                
                # Check if image might contain text
                likely_has_text = self._assess_text_likelihood(img, file_path)
                result['likely_contains_text'] = likely_has_text
                
                # Extract EXIF data if available
                exif_data = self._extract_exif_data(img)
                if exif_data:
                    result['exif_data'] = exif_data
            
            # Determine content type
            content_type = self._determine_image_type(file_path, result)
            result['image_content_type'] = content_type
            
            # Default content description
            extension = file_path.suffix.lower()
            if extension == '.png':
                result['content'] = '[PNG image - potential diagram, chart, or screenshot]'
            else:
                result['content'] = f'[{extension.upper()} image - {content_type}]'
            
            # Perform OCR if appropriate
            if self._should_perform_ocr(file_path, result, likely_has_text):
                logger.info(f"Performing OCR on image: {file_path.name}")
                ocr_result = self.ocr_processor.process_image(file_path)
                
                if ocr_result and 'content' in ocr_result:
                    extracted_text = ocr_result['content']
                    
                    # Only use OCR text if it's meaningful
                    if len(extracted_text.strip()) > 20:
                        result['content'] = extracted_text
                        result['extraction_method'] = 'OCR'
                        result['ocr_confidence'] = ocr_result.get('ocr_confidence', 0)
                        result['ocr_quality'] = ocr_result.get('ocr_quality', 'Unknown')
                        
                        # Count words
                        words = len(extracted_text.split())
                        result['word_count'] = words
                        
                        # Detect if it's a specific type of diagram
                        if self._is_diagram(extracted_text):
                            result['image_content_type'] = 'diagram'
                            result['appears_to_be_diagram'] = True
                
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            result['content'] = f'[Image file - Could not process: {str(e)}]'
            result['error'] = str(e)
        
        return result
    
    def _assess_text_likelihood(self, img: Any, file_path: Path) -> bool:
        """
        Assess likelihood that image contains text.
        
        Args:
            img: PIL Image object
            file_path: Path to image file
            
        Returns:
            True if image likely contains text
        """
        # PNG files often contain diagrams or screenshots with text
        if file_path.suffix.lower() == '.png':
            return True
        
        # Large images might be documents
        if img.width > 600 and img.height > 600:
            # Check aspect ratio - documents tend to be portrait
            aspect_ratio = img.height / img.width
            if 1.2 < aspect_ratio < 1.8:  # Roughly A4 ratio
                return True
        
        # High resolution suggests possible document scan
        if img.width > 1500 or img.height > 1500:
            return True
        
        # Grayscale or black and white images often contain text
        if img.mode in ['L', '1']:
            return True
        
        return False
    
    def _should_perform_ocr(self, file_path: Path, metadata: Dict[str, Any], likely_has_text: bool) -> bool:
        """
        Determine if OCR should be performed on the image.
        
        Args:
            file_path: Path to image file
            metadata: Image metadata
            likely_has_text: Whether image likely contains text
            
        Returns:
            True if OCR should be performed
        """
        if not self.ocr_processor or not self.ocr_processor.available:
            return False
        
        # Always try OCR on PNG files (often diagrams/charts)
        if file_path.suffix.lower() == '.png':
            return True
        
        # Try OCR if image likely contains text
        if likely_has_text:
            return True
        
        # Skip OCR on very small images
        if metadata.get('image_width', 0) < 200 or metadata.get('image_height', 0) < 200:
            return False
        
        # Skip OCR on very large images (photos)
        if metadata.get('image_megapixels', 0) > 10:
            return False
        
        return False
    
    def _determine_image_type(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Determine the type of image content.
        
        Args:
            file_path: Path to image file
            metadata: Image metadata
            
        Returns:
            Image content type description
        """
        # Check file name for hints
        name_lower = file_path.stem.lower()
        
        if any(term in name_lower for term in ['screenshot', 'screen', 'capture']):
            return 'screenshot'
        elif any(term in name_lower for term in ['diagram', 'chart', 'graph', 'plot']):
            return 'diagram'
        elif any(term in name_lower for term in ['scan', 'document', 'page']):
            return 'scanned document'
        elif any(term in name_lower for term in ['photo', 'img', 'pic', 'image']):
            return 'photograph'
        
        # Check by image properties
        width = metadata.get('image_width', 0)
        height = metadata.get('image_height', 0)
        
        # Very wide images might be panoramas
        if width > height * 2.5:
            return 'panorama or banner'
        
        # Square images might be icons or profile pictures
        if 0.9 < width / height < 1.1 and width < 1000:
            return 'icon or profile image'
        
        # High resolution portrait might be document
        if height > width * 1.2 and height > 1500:
            return 'possible document scan'
        
        # Default based on format
        if file_path.suffix.lower() == '.png':
            return 'diagram or screenshot'
        
        return 'general image'
    
    def _extract_exif_data(self, img: Any) -> Optional[Dict[str, Any]]:
        """
        Extract EXIF data from image.
        
        Args:
            img: PIL Image object
            
        Returns:
            Dictionary with EXIF data or None
        """
        try:
            from PIL.ExifTags import TAGS
            
            if hasattr(img, '_getexif') and img._getexif():
                exif_data = {}
                raw_exif = img._getexif()
                
                # Extract common EXIF fields
                interesting_tags = {
                    'Make': 'camera_make',
                    'Model': 'camera_model',
                    'DateTime': 'date_taken',
                    'ExposureTime': 'exposure_time',
                    'FNumber': 'f_number',
                    'ISO': 'iso',
                    'FocalLength': 'focal_length',
                    'LensModel': 'lens_model',
                    'Software': 'software',
                    'Orientation': 'orientation'
                }
                
                for tag_id, value in raw_exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    
                    if tag_name in interesting_tags:
                        # Convert bytes to string
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except:
                                continue
                        
                        exif_data[interesting_tags[tag_name]] = value
                
                # Check for GPS data
                if 'GPSInfo' in raw_exif:
                    exif_data['has_gps_data'] = True
                
                return exif_data if exif_data else None
                
        except Exception as e:
            logger.warning(f"Failed to extract EXIF data: {e}")
            return None
    
    def _is_diagram(self, text: str) -> bool:
        """
        Check if extracted text suggests this is a diagram.
        
        Args:
            text: Extracted text from OCR
            
        Returns:
            True if text suggests a diagram
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Common diagram indicators
        diagram_terms = [
            'figure', 'fig.', 'chart', 'graph', 'diagram',
            'table', 'plot', 'axis', 'legend', '%',
            'data', 'trend', 'correlation'
        ]
        
        matches = sum(1 for term in diagram_terms if term in text_lower)
        
        # Check for numeric content (common in charts)
        numbers = len(re.findall(r'\d+', text))
        number_ratio = numbers / len(text.split()) if text.split() else 0
        
        return matches >= 2 or number_ratio > 0.2