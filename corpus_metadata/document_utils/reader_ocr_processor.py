#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_ocr_processor.py
#
"""
OCR Processor Module
===================

Handles OCR processing for scanned documents and images.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import re

from corpus_metadata.document_utils.metadata_config_loader import CorpusConfig
from corpus_metadata.document_utils.reader_exceptions import OCRError

# Optional imports - not all may be available
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import pdf2image
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Processes documents and images using OCR to extract text.
    
    Uses Tesseract OCR with various preprocessing techniques
    to improve accuracy.
    """
    
    def __init__(self, config=None):
        """
        Initialize OCR processor with configuration.
        
        Args:
            config: Optional OCRConfig dict for backward compatibility
        """
        # Load configuration from YAML
        self.corpus_config = CorpusConfig()
        ocr_settings = self.corpus_config.config.get('extraction', {}).get('ocr_settings', {})
        
        # Use provided config or load from YAML
        if config and isinstance(config, dict):
            # Merge provided config with YAML settings
            self.languages = config.get('languages', ocr_settings.get('languages', ['eng']))
            self.dpi = config.get('dpi', ocr_settings.get('dpi', 300))
            self.timeout = config.get('timeout', ocr_settings.get('timeout_seconds', 60))
            self.engine = config.get('engine', ocr_settings.get('engine', 'tesseract'))
            self.max_pages = config.get('max_pages', ocr_settings.get('max_pages', 5))
            self.confidence_threshold = config.get('confidence_threshold', 
                                                 ocr_settings.get('confidence_threshold', 60))
        else:
            # Load all settings from YAML
            self.languages = ocr_settings.get('languages', ['eng'])
            self.dpi = ocr_settings.get('dpi', 300)
            self.timeout = ocr_settings.get('timeout_seconds', 60)
            self.engine = ocr_settings.get('engine', 'tesseract')
            self.max_pages = ocr_settings.get('max_pages', 5)
            self.confidence_threshold = ocr_settings.get('confidence_threshold', 60)
        
        # Advanced OCR settings
        advanced_settings = ocr_settings.get('advanced', {})
        self.enable_preprocessing = advanced_settings.get('enable_preprocessing', True)
        self.psm_mode = advanced_settings.get('psm_mode', 3)  # Tesseract page segmentation mode
        self.oem_mode = advanced_settings.get('oem_mode', 3)  # OCR Engine mode
        self.preserve_interword_spaces = advanced_settings.get('preserve_interword_spaces', True)
        self.enable_deskew = advanced_settings.get('enable_deskew', True)
        self.enable_denoise = advanced_settings.get('enable_denoise', True)
        
        # Preprocessing settings
        preprocessing = advanced_settings.get('preprocessing', {})
        self.contrast_enhancement = preprocessing.get('contrast_enhancement', 1.5)
        self.sharpness_enhancement = preprocessing.get('sharpness_enhancement', 2.0)
        self.median_filter_size = preprocessing.get('median_filter_size', 3)
        self.max_image_dimension = preprocessing.get('max_image_dimension', 4000)
        
        # Quality assessment settings
        quality_settings = ocr_settings.get('quality_assessment', {})
        self.min_text_length = quality_settings.get('min_text_length', 10)
        self.max_word_length = quality_settings.get('max_word_length', 15)
        self.max_long_word_ratio = quality_settings.get('max_long_word_ratio', 0.05)
        self.high_confidence_threshold = quality_settings.get('high_confidence_threshold', 80)
        self.medium_confidence_threshold = quality_settings.get('medium_confidence_threshold', 60)
        
        # Error recovery settings
        self.retry_on_failure = ocr_settings.get('retry_on_failure', True)
        self.max_retries = ocr_settings.get('max_retries', 2)
        self.fallback_to_basic = ocr_settings.get('fallback_to_basic_ocr', True)
        
        # Language string for Tesseract
        self.language_string = '+'.join(self.languages)
        
        # Check dependencies
        self.available = self._check_dependencies()
        
        if not self.available:
            logger.warning("OCR dependencies not available")
    
    def _check_dependencies(self) -> bool:
        """Check if required OCR dependencies are available."""
        try:
            if self.engine == 'tesseract':
                if not TESSERACT_AVAILABLE or not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE:
                    logger.error("Tesseract dependencies not available. Install with: pip install pytesseract pdf2image pillow")
                    return False
                
                # Check if Tesseract is installed
                try:
                    version = pytesseract.get_tesseract_version()
                    logger.info(f"Tesseract OCR available: {version}")
                    return True
                except Exception as e:
                    logger.error(f"Tesseract not installed on system. Please install Tesseract OCR.")
                    return False
            
            elif self.engine == 'easyocr':
                if not EASYOCR_AVAILABLE:
                    logger.warning("EasyOCR not installed. Install with: pip install easyocr")
                    return False
                logger.info("EasyOCR available")
                return True
            
            else:
                logger.error(f"Unknown OCR engine: {self.engine}")
                return False
                
        except Exception as e:
            logger.error(f"OCR dependencies check failed: {e}")
            return False
    
    def process_pdf(self, file_path: Path, max_pages: Optional[int] = None) -> Dict[str, Any]:
        """
        Process PDF file using OCR.
        
        Args:
            file_path: Path to PDF file
            max_pages: Maximum number of pages to process
            
        Returns:
            Dictionary with OCR results
        """
        if not self.available:
            raise OCRError("OCR dependencies not available")
        
        max_pages = max_pages or self.max_pages
        
        # Try OCR with retry logic
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                if self.engine == 'tesseract':
                    return self._process_pdf_tesseract(file_path, max_pages)
                elif self.engine == 'easyocr':
                    return self._process_pdf_easyocr(file_path, max_pages)
                else:
                    raise OCRError(f"Unsupported OCR engine: {self.engine}")
                    
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt <= self.max_retries:
                    logger.warning(f"OCR attempt {attempt} failed, retrying: {e}")
                    # Try with reduced settings on retry
                    if self.fallback_to_basic and attempt == self.max_retries:
                        self.enable_preprocessing = False
                        self.dpi = 200
        
        raise OCRError(f"PDF OCR failed after {attempt} attempts: {last_error}")
    
    def _process_pdf_tesseract(self, file_path: Path, max_pages: int) -> Dict[str, Any]:
        """Process PDF using Tesseract OCR."""
        if not PDF2IMAGE_AVAILABLE or not TESSERACT_AVAILABLE:
            raise OCRError("Required dependencies not available for Tesseract OCR")
            
        ocr_text = []
        page_confidences = []
        processing_info = []
        
        # Convert PDF pages to images
        logger.info(f"Converting PDF to images (DPI: {self.dpi}, max pages: {max_pages})")
        
        images = convert_from_path(
            file_path,
            first_page=1,
            last_page=max_pages,
            dpi=self.dpi,
            thread_count=2,
            use_pdftocairo=True,  # Better quality
            timeout=self.timeout
        )
        
        for i, image in enumerate(images):
            try:
                # Preprocess image if enabled
                if self.enable_preprocessing:
                    image = self._preprocess_image(image)
                
                # Get OCR data with confidence
                data = pytesseract.image_to_data(
                    image,
                    lang=self.language_string,
                    config=self._get_tesseract_config(),
                    output_type=pytesseract.Output.DICT,
                    timeout=self.timeout
                )
                
                # Extract text
                text = pytesseract.image_to_string(
                    image,
                    lang=self.language_string,
                    config=self._get_tesseract_config(),
                    timeout=self.timeout
                )
                
                # Clean OCR text
                text = self._clean_ocr_text(text)
                ocr_text.append(text)
                
                # Calculate page confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                page_confidence = sum(confidences) / len(confidences) if confidences else 0
                page_confidences.append(page_confidence)
                
                processing_info.append({
                    'page': i + 1,
                    'confidence': page_confidence,
                    'text_length': len(text),
                    'word_count': len(text.split())
                })
                
                logger.debug(f"OCR completed for page {i+1} (confidence: {page_confidence:.1f}%)")
                
            except Exception as e:
                logger.error(f"OCR failed for page {i+1}: {e}")
                ocr_text.append(f"[OCR error on page {i+1}]")
                page_confidences.append(0)
                processing_info.append({
                    'page': i + 1,
                    'error': str(e)
                })
        
        # Calculate overall confidence
        avg_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0
        
        # Assess overall quality
        full_text = '\n'.join(ocr_text)
        quality = self._assess_ocr_quality(full_text, avg_confidence)
        
        return {
            'content': full_text,
            'extraction_method': 'OCR',
            'ocr_engine': 'tesseract',
            'ocr_pages_processed': len(images),
            'ocr_confidence': avg_confidence,
            'ocr_quality': quality,
            'ocr_config': {
                'dpi': self.dpi,
                'languages': self.languages,
                'preprocessing': self.enable_preprocessing,
                'psm_mode': self.psm_mode,
                'oem_mode': self.oem_mode
            },
            'page_details': processing_info
        }
    
    def _process_pdf_easyocr(self, file_path: Path, max_pages: int) -> Dict[str, Any]:
        """Process PDF using EasyOCR (alternative OCR engine)."""
        # Implementation for EasyOCR if needed
        raise NotImplementedError("EasyOCR support not yet implemented")
    
    def process_image(self, file_path: Path) -> Dict[str, Any]:
        """
        Process image file using OCR.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with OCR results
        """
        if not self.available:
            raise OCRError("OCR dependencies not available")
        
        try:
            if self.engine == 'tesseract':
                return self._process_image_tesseract(file_path)
            elif self.engine == 'easyocr':
                return self._process_image_easyocr(file_path)
            else:
                raise OCRError(f"Unsupported OCR engine: {self.engine}")
                
        except Exception as e:
            logger.error(f"Image OCR processing failed: {e}")
            raise OCRError(f"Image OCR failed: {e}")
    
    def _process_image_tesseract(self, file_path: Path) -> Dict[str, Any]:
        """Process image using Tesseract."""
        if not PIL_AVAILABLE or not TESSERACT_AVAILABLE:
            raise OCRError("Required dependencies not available for Tesseract OCR")
        
        # Open image
        image = Image.open(file_path)
        original_size = (image.width, image.height)
        
        # Preprocess if enabled
        if self.enable_preprocessing:
            image = self._preprocess_image(image)
        
        # Perform OCR with confidence scores
        data = pytesseract.image_to_data(
            image,
            lang=self.language_string,
            config=self._get_tesseract_config(),
            output_type=pytesseract.Output.DICT,
            timeout=self.timeout
        )
        
        # Extract text
        text = pytesseract.image_to_string(
            image,
            lang=self.language_string,
            config=self._get_tesseract_config(),
            timeout=self.timeout
        )
        
        # Clean text
        text = self._clean_ocr_text(text)
        
        # Calculate average confidence
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Determine quality based on confidence
        quality = self._assess_ocr_quality(text, avg_confidence)
        
        # Extract bounding boxes for high-confidence text
        high_conf_boxes = []
        for i in range(len(data['conf'])):
            if int(data['conf'][i]) > self.confidence_threshold:
                high_conf_boxes.append({
                    'text': data['text'][i],
                    'confidence': data['conf'][i],
                    'bbox': (data['left'][i], data['top'][i], 
                            data['width'][i], data['height'][i])
                })
        
        return {
            'content': text,
            'extraction_method': 'OCR',
            'ocr_engine': 'tesseract',
            'ocr_confidence': avg_confidence,
            'ocr_quality': quality,
            'image_info': {
                'original_size': original_size,
                'processed_size': (image.width, image.height),
                'mode': image.mode
            },
            'high_confidence_regions': len(high_conf_boxes),
            'total_text_regions': len([c for c in data['conf'] if int(c) > 0])
        }
    
    def _process_image_easyocr(self, file_path: Path) -> Dict[str, Any]:
        """Process image using EasyOCR."""
        # Implementation for EasyOCR if needed
        raise NotImplementedError("EasyOCR support not yet implemented")
    
    def _get_tesseract_config(self) -> str:
        """
        Build Tesseract configuration string.
        
        Returns:
            Configuration string for Tesseract
        """
        config_parts = [
            f'--psm {self.psm_mode}',
            f'--oem {self.oem_mode}'
        ]
        
        # Add advanced options
        if self.preserve_interword_spaces:
            config_parts.append('-c preserve_interword_spaces=1')
        
        # Add custom character whitelist if specified
        char_whitelist = self.corpus_config.config.get('extraction', {}).get(
            'ocr_settings', {}
        ).get('advanced', {}).get('char_whitelist')
        
        if char_whitelist:
            config_parts.append(f'-c tessedit_char_whitelist="{char_whitelist}"')
        
        return ' '.join(config_parts)
    
    def _preprocess_image(self, image: Any) -> Any:
        """
        Apply preprocessing to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available for preprocessing")
            return image
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        if image.width > self.max_image_dimension or image.height > self.max_image_dimension:
            image.thumbnail((self.max_image_dimension, self.max_image_dimension), 
                          Image.Resampling.LANCZOS)
        
        # Deskew if enabled
        if self.enable_deskew:
            try:
                image = self._deskew_image(image)
            except Exception as e:
                logger.warning(f"Deskew failed: {e}")
        
        # Enhance image
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.contrast_enhancement)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(self.sharpness_enhancement)
        
        # Apply denoising if enabled
        if self.enable_denoise and self.median_filter_size > 0:
            image = image.filter(ImageFilter.MedianFilter(size=self.median_filter_size))
        
        # Convert to grayscale for better OCR
        image = image.convert('L')
        
        # Apply adaptive thresholding if configured
        if self.corpus_config.config.get('extraction', {}).get(
            'ocr_settings', {}
        ).get('advanced', {}).get('preprocessing', {}).get('adaptive_threshold', False):
            image = self._apply_adaptive_threshold(image)
        
        return image
    
    def _deskew_image(self, image: Any) -> Any:
        """
        Deskew an image using Hough transform.
        
        Args:
            image: PIL Image
            
        Returns:
            Deskewed image
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available for deskewing")
            return image
        
        # Convert to numpy array
        img_array = np.array(image.convert('L'))
        
        # Apply edge detection
        edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
        
        # Apply Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            # Calculate the most common angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 <= angle <= 45:  # Reasonable rotation range
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                # Rotate image
                return image.rotate(median_angle, fillcolor=255, expand=True)
        
        return image
    
    def _apply_adaptive_threshold(self, image: Any) -> Any:
        """Apply adaptive thresholding to image."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available for adaptive thresholding")
            return image
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            img_array, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(thresh)
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean up OCR-generated text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Get cleaning rules from config
        cleaning_rules = self.corpus_config.config.get('extraction', {}).get(
            'ocr_settings', {}
        ).get('text_cleaning', {})
        
        # Apply basic cleaning
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix word breaks at line ends if configured
        if cleaning_rules.get('fix_line_breaks', True):
            text = re.sub(r'([a-z])-\s+([a-z])', r'\1\2', text)
        
        # Remove single character "words" that are likely noise
        if cleaning_rules.get('remove_single_chars', True):
            text = re.sub(r'\s[b-zB-Z]\s', ' ', text)
        
        # Apply custom replacements from config
        custom_replacements = cleaning_rules.get('custom_replacements', {})
        for pattern, replacement in custom_replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix common character substitutions
        if cleaning_rules.get('fix_common_ocr_errors', True):
            replacements = {
                '\u2022': '•',  # Bullet point
                '\u2013': '-',  # En dash
                '\u2014': '-',  # Em dash
                '\u2019': "'",  # Right single quote
                '\u201c': '"',  # Left double quote
                '\u201d': '"',  # Right double quote
                '|': 'I',       # Common OCR error
                '0': 'O',       # When in text context
            }
            
            for old, new in replacements.items():
                text = text.replace(old, new)
        
        # Remove multiple consecutive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove non-printable characters if configured
        if cleaning_rules.get('remove_non_printable', True):
            text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()
    
    def _assess_ocr_quality(self, text: str, confidence: float) -> str:
        """
        Assess the quality of OCR results.
        
        Args:
            text: OCR text
            confidence: Average confidence score
            
        Returns:
            Quality assessment (High/Medium/Low)
        """
        if not text or len(text) < self.min_text_length:
            return "Low"
        
        # Check confidence score
        if confidence < self.medium_confidence_threshold:
            return "Low"
        
        # Analyze text characteristics
        words = text.split()
        if not words:
            return "Low"
        
        # Calculate metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Count suspicious patterns
        long_words = sum(1 for word in words if len(word) > self.max_word_length)
        long_word_ratio = long_words / len(words)
        
        # Check for nonsensical character sequences
        consonant_sequences = re.findall(r'[bcdfghjklmnpqrstvwxz]{5,}', text.lower())
        
        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = len(words) / max(1, len(sentences))
        
        # Determine quality
        quality_score = 0
        
        # Confidence score contribution
        if confidence > self.high_confidence_threshold:
            quality_score += 3
        elif confidence > self.medium_confidence_threshold:
            quality_score += 2
        
        # Text structure contribution
        if avg_word_length < 10:
            quality_score += 1
        if long_word_ratio < self.max_long_word_ratio:
            quality_score += 1
        if not consonant_sequences:
            quality_score += 1
        if 5 <= avg_sentence_length <= 30:
            quality_score += 1
        
        # Determine final quality
        if quality_score >= 6:
            return "High"
        elif quality_score >= 4:
            return "Medium"
        else:
            return "Low"
    
    def get_ocr_info(self) -> Dict[str, Any]:
        """
        Get information about OCR configuration.
        
        Returns:
            Dictionary with OCR information
        """
        return {
            'engine': self.engine,
            'available': self.available,
            'languages': self.languages,
            'dpi': self.dpi,
            'preprocessing_enabled': self.enable_preprocessing,
            'confidence_threshold': self.confidence_threshold,
            'max_pages': self.max_pages,
            'configuration_source': 'YAML'
        }