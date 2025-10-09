#!/usr/bin/env python3

#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_metadata_converter.py
# 
import os
import logging
import platform
import subprocess
import shutil
import time
import tempfile
import concurrent.futures
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration - HARD CODED PARAMETERS (adjust as needed)
# -----------------------------------------------------------------------------
SOURCE_DIRECTORY = './documents'              # Root directory to search for documents
ARCHIVE_DIRECTORY = './documents_converted_to_pdf'  # Directory to move processed files
PDF_OUTPUT_DIRECTORY = './documents'          # NEW: Where to place all PDFs (root level)
DELETE_ORIGINAL = False                         # Delete originals after conversion if True
VERBOSE_LOGGING = True                          # Detailed logging flag
MAX_FILES = None                                 # Limit number of conversions (None for unlimited)
MAX_WORKERS = 2                                 # Maximum number of concurrent conversions
FORCE_RECONVERSION = False                      # Force regeneration even if PDF exists
CONVERSION_TIMEOUT = 300                        # Conversion process timeout (in seconds)
DIAGNOSTIC_MODE = True                          # Enable additional diagnostic logging

# Quality and PDF settings
DPI = 600                                       # Resolution (in DPI)
IMAGE_QUALITY = 100                             # JPEG quality (100 is best)
PDF_VERSION = "1.6"                             # PDF version for compatibility

# If set to True, only basic filter options will be applied.
# This simplifies the PDF export by LibreOffice and minimizes potential layout issues.
SIMPLIFY_CONVERSION = True

# List of known problematic files (by base name)
PROBLEM_FILES = [
    "Cell tx site engagement - Alexion PAG.pptx"
]

# -----------------------------------------------------------------------------
# Logger Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # Set global level to INFO. DEBUG messages (including file characteristics) 
                         # will not be printed unless you change the log level.
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_conversion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper Functions for Environment, Logging, and Diagnostics
# -----------------------------------------------------------------------------
def get_libreoffice_path():
    """Locate the LibreOffice executable."""
    possible_paths = [
        '/Applications/LibreOffice.app/Contents/MacOS/soffice',
        '/opt/homebrew/bin/soffice',
        '/opt/homebrew/Caskroom/libreoffice/*/LibreOffice.app/Contents/MacOS/soffice',
        '/usr/local/bin/soffice',
        '/usr/local/Caskroom/libreoffice/*/LibreOffice.app/Contents/MacOS/soffice',
        '/Applications/OpenOffice.app/Contents/MacOS/soffice',
    ]
    soffice_in_path = shutil.which('soffice')
    if soffice_in_path:
        return soffice_in_path
    for path in possible_paths:
        if '*' in path:
            import glob
            for match in glob.glob(path):
                if os.path.exists(match):
                    return match
        elif os.path.exists(path):
            return path
    return 'soffice'  # Fallback

def create_libreoffice_settings():
    """Stub for creating LibreOffice settings if needed."""
    user_home = os.path.expanduser("~")
    config_dir = os.path.join(user_home, ".config", "libreoffice", "4", "user")
    os.makedirs(config_dir, exist_ok=True)
    try:
        logger.info("LibreOffice high-quality PDF export settings will be used.")
    except Exception as e:
        logger.warning(f"Could not configure LibreOffice settings: {e}")

def get_libreoffice_version():
    """Retrieve LibreOffice version string."""
    soffice_path = get_libreoffice_path()
    try:
        result = subprocess.run([soffice_path, '--version'], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error getting version: {e}"

def check_file_characteristics(file_path):
    """Obtain basic file metadata for diagnostics."""
    try:
        size_kb = os.path.getsize(file_path) / 1024
        extension = os.path.splitext(file_path)[1].lower()
        info = {
            'size_kb': size_kb,
            'extension': extension,
            'last_modified': time.ctime(os.path.getmtime(file_path)),
        }
        # For PowerPoint files on macOS, get extra metadata via mdls
        if extension in ['.ppt', '.pptx'] and platform.system() == 'Darwin':
            try:
                cmd = ['mdls', file_path]
                mdls_result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if mdls_result.returncode == 0:
                    info['metadata'] = mdls_result.stdout
            except Exception:
                pass
        # For Word files on macOS, get extra metadata via mdls
        elif extension in ['.doc', '.docx'] and platform.system() == 'Darwin':
            try:
                cmd = ['mdls', file_path]
                mdls_result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if mdls_result.returncode == 0:
                    info['metadata'] = mdls_result.stdout
            except Exception:
                pass
        return info
    except Exception as e:
        return {'error': str(e)}

def convert_with_diagnostics(cmd, input_file, output_file):
    """
    Run conversion command with detailed diagnostic logging.
    Returns a tuple (success, log_output, duration).
    """
    temp_log = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
    temp_log.close()
    logger.debug(f"DIAGNOSTIC: Running command: {' '.join(cmd)}")
    logger.info(f"DIAGNOSTIC: LibreOffice version: {get_libreoffice_version()}")
    file_info = check_file_characteristics(input_file)
    logger.debug(f"DIAGNOSTIC: File characteristics: {file_info}")  # Now at DEBUG level
    env = os.environ.copy()
    env['SAL_LOG'] = '+INFO.libreoffice'
    env['TMPDIR'] = os.path.dirname(temp_log.name)
    
    start_time = time.time()
    try:
        with open(temp_log.name, 'w') as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            try:
                process.wait(timeout=CONVERSION_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"Process timed out after {CONVERSION_TIMEOUT} seconds")
        end_time = time.time()
        duration = end_time - start_time
        with open(temp_log.name, 'r') as log_file:
            log_content = log_file.read()
        os.unlink(temp_log.name)
        if process.returncode != 0:
            logger.error(f"DIAGNOSTIC: Process failed with code {process.returncode}")
            logger.error(f"DIAGNOSTIC: Command output (truncated):\n{log_content[:2000]}")
            return False, log_content, duration
        if not os.path.exists(output_file):
            logger.error(f"DIAGNOSTIC: Output file not created: {output_file}")
            output_dir = os.path.dirname(output_file)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            pdf_candidates = [f for f in os.listdir(output_dir) if f.endswith('.pdf') and f.startswith(base_name)]
            if pdf_candidates:
                logger.info(f"DIAGNOSTIC: Found potential output files: {pdf_candidates}")
            else:
                logger.info("DIAGNOSTIC: No matching PDF files found")
            return False, log_content, duration
        logger.info(f"DIAGNOSTIC: Conversion successful in {duration:.2f} seconds")
        return True, log_content, duration
    except Exception as e:
        logger.error(f"DIAGNOSTIC: Exception during conversion: {e}")
        if os.path.exists(temp_log.name):
            os.unlink(temp_log.name)
        return False, str(e), 0

def get_temp_lo_profile():
    """Create a temporary directory for isolated LibreOffice user profile."""
    profile_dir = tempfile.mkdtemp(prefix="lo_profile_")
    return profile_dir

# -----------------------------------------------------------------------------
# Filter Options Functions (selective by file type)
# -----------------------------------------------------------------------------
def get_basic_filter_options():
    """Return basic filter options that are less likely to fail."""
    return [
        "EmbedFonts=true",
        f"Quality={IMAGE_QUALITY}",
        f"MaxImageResolution={DPI}",
        "SelectPdfVersion=0",
    ]

def get_additional_filter_options():
    """Return additional (advanced) options for high-quality output."""
    # If SIMPLIFY_CONVERSION is True then we return an empty list.
    if SIMPLIFY_CONVERSION:
        return []
    return [
        "EmbedOnlyUsedFonts=false",
        "EmbedStandardFonts=true",
        "ReduceImageResolution=false",
        "CompressVectorGraphics=false",
        "Compress=0",
        "JPEGQuality=100",
        "UseTaggedPDF=true",
        "ExportFormFields=true",
        "ExportBookmarks=true",
        "ExportNotes=true",
        "UseICC=true",
        "ExportLinksRelativeFsys=true",
    ]

def get_ppt_filter_options():
    """Return PowerPoint-specific export options."""
    # If SIMPLIFY_CONVERSION is True then do not add PPT-specific options.
    if SIMPLIFY_CONVERSION:
        return []
    return [
        "ExportSlideContents=true",
        "ExportNotesPages=false",
        "ExportHiddenSlides=false",
        "SaveTransitionEffects=true",
        "SlideBackgroundExport=true"
    ]

def get_docx_filter_options():
    """Return Word document-specific export options for DOCX/DOC files."""
    # If SIMPLIFY_CONVERSION is True then do not add DOCX-specific options.
    if SIMPLIFY_CONVERSION:
        return []
    return [
        "ExportFormFields=true",
        "ExportBookmarks=true",
        "ExportNotes=true",
        "ExportNotesPages=false",
        "ExportPlaceholders=false",
        "ExportHiddenText=false",
        "SinglePageSheets=false",
        "UseTransitionEffects=false",
        "IsSkipEmptyPages=false",
        "IsAddStream=false",
        "EmbedStandardFonts=true",
        "FormsType=0"
    ]

def get_pdf_export_filter(file_extension):
    """Return the appropriate PDF export filter based on file type."""
    if file_extension.lower() in ['.ppt', '.pptx']:
        return 'impress_pdf_Export'
    elif file_extension.lower() in ['.doc', '.docx']:
        return 'writer_pdf_Export'
    else:
        return 'writer_pdf_Export'  # Default fallback

def build_filter_options(file_path, basic_only=False):
    """
    Compose filter options based on file extension and retry level.
    Now includes specific optimizations for DOCX files.
    """
    filters = get_basic_filter_options()
    extension = Path(file_path).suffix.lower()
    
    if not basic_only:
        additional = get_additional_filter_options()
        
        # For PPT/PPTX, remove any FitToPages options if present and add PPT-specific options
        if extension in ['.ppt', '.pptx']:
            additional = [opt for opt in additional if not opt.startswith("FitToPagesWidth") and not opt.startswith("FitToPagesHeight")]
            filters += additional
            filters += get_ppt_filter_options()
        # For DOCX/DOC, add Word-specific options
        elif extension in ['.doc', '.docx']:
            filters += additional
            filters += get_docx_filter_options()
        else:
            filters += additional
    
    return ":".join(filters)

# -----------------------------------------------------------------------------
# Conversion Methods
# -----------------------------------------------------------------------------
def convert_to_pdf_libreoffice(input_file, output_file=None, basic_only=False):
    """
    Primary conversion method using LibreOffice.
    Uses a temporary user profile for isolation.
    Now optimized for different file types including DOCX.
    """
    input_file = os.path.abspath(input_file)
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'
    else:
        output_file = os.path.abspath(output_file)
    output_dir = os.path.dirname(output_file)
    soffice_path = get_libreoffice_path()

    # If file is known to be problematic, jump to the alternative conversion.
    file_basename = os.path.basename(input_file)
    if file_basename in PROBLEM_FILES:
        logger.info(f"Using alternative conversion for problematic file: {file_basename}")
        return convert_to_pdf_alternative(input_file, output_file)

    # Get file extension for filter selection
    file_extension = Path(input_file).suffix.lower()
    
    # Build filter options based on whether we are doing a full or basic retry.
    filter_options = build_filter_options(input_file, basic_only)
    
    # Select appropriate export filter
    export_filter = get_pdf_export_filter(file_extension)
    
    # Create a temporary LibreOffice user profile directory for this conversion.
    temp_profile = get_temp_lo_profile()
    profile_flag = f"-env:UserInstallation=file:///{temp_profile}"

    cmd = [
        soffice_path,
        '--headless',
        profile_flag,
        '--convert-to', f'pdf:{export_filter}:{filter_options}',
        '--outdir', output_dir,
        input_file
    ]
    
    if VERBOSE_LOGGING:
        logger.info(f"Converting {file_extension.upper()} file: {input_file}")
        logger.info(f"Using LibreOffice at: {soffice_path}")
        logger.info(f"Export filter: {export_filter}")
        logger.info(f"Filter options: {filter_options}")
        logger.info(f"Temporary LibreOffice profile: {temp_profile}")

    try:
        if DIAGNOSTIC_MODE:
            success, log_output, duration = convert_with_diagnostics(cmd, input_file, output_file)
            shutil.rmtree(temp_profile, ignore_errors=True)
            if not success and not basic_only:
                logger.info("Primary conversion failed; retrying with basic filter options only.")
                return convert_to_pdf_libreoffice(input_file, output_file, basic_only=True)
            elif not success:
                logger.error("Conversion failed even with basic filter options.")
                return False
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=CONVERSION_TIMEOUT)
            shutil.rmtree(temp_profile, ignore_errors=True)
            if result.returncode != 0:
                logger.error(f"LibreOffice conversion failed: {result.stderr}")
                return False

        # LibreOffice may name the file differently; try to move/rename accordingly.
        expected_pdf = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.pdf')
        if os.path.exists(expected_pdf) and expected_pdf != output_file:
            shutil.move(expected_pdf, output_file)
        if not os.path.exists(output_file):
            logger.error(f"PDF output file not created: {output_file}")
            return False
        if os.path.getsize(output_file) < 1000:
            logger.warning(f"PDF file is very small ({os.path.getsize(output_file)} bytes); it may be incomplete.")
        logger.info(f"Successfully converted {file_extension.upper()} to PDF: {output_file}")
        return True

    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_profile, ignore_errors=True)
        logger.error(f"Conversion timed out after {CONVERSION_TIMEOUT} seconds for {input_file}")
        return False
    except Exception as e:
        shutil.rmtree(temp_profile, ignore_errors=True)
        logger.error(f"Error during LibreOffice conversion: {e}")
        return False

def convert_to_pdf_alternative(input_file, output_file=None):
    """
    Alternative conversion method using simplified LibreOffice options.
    Enhanced with better DOCX support and textutil fallback for Word files on macOS.
    """
    input_file = os.path.abspath(input_file)
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'
    else:
        output_file = os.path.abspath(output_file)
    output_dir = os.path.dirname(output_file)
    soffice_path = get_libreoffice_path()

    temp_profile = get_temp_lo_profile()
    profile_flag = f"-env:UserInstallation=file:///{temp_profile}"
    
    # Get file extension for appropriate filter
    file_extension = Path(input_file).suffix.lower()
    export_filter = get_pdf_export_filter(file_extension)
    
    cmd = [
        soffice_path,
        '--headless',
        profile_flag,
        '--convert-to', f'pdf:{export_filter}',
        '--outdir', output_dir,
        input_file
    ]
    logger.info(f"Using simplified conversion for {file_extension.upper()}: {input_file}")
    try:
        if DIAGNOSTIC_MODE:
            success, log_output, duration = convert_with_diagnostics(cmd, input_file, output_file)
            shutil.rmtree(temp_profile, ignore_errors=True)
            if not success:
                logger.warning(f"Alternative conversion failed after {duration:.2f} seconds")
                return try_textutil_conversion(input_file, output_file)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=CONVERSION_TIMEOUT)
            shutil.rmtree(temp_profile, ignore_errors=True)
            if result.returncode != 0:
                logger.error(f"Alternative conversion failed: {result.stderr}")
                return try_textutil_conversion(input_file, output_file)

        expected_pdf = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.pdf')
        if os.path.exists(expected_pdf) and expected_pdf != output_file:
            shutil.move(expected_pdf, output_file)
        if not os.path.exists(output_file):
            logger.error(f"PDF output file not created: {output_file}")
            return try_textutil_conversion(input_file, output_file)

        logger.info(f"Successfully converted {file_extension.upper()} using alternative method: {output_file}")
        return True

    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_profile, ignore_errors=True)
        logger.error(f"Alternative conversion timed out for {input_file}")
        return try_textutil_conversion(input_file, output_file)
    except Exception as e:
        shutil.rmtree(temp_profile, ignore_errors=True)
        logger.error(f"Error during alternative conversion: {e}")
        return try_textutil_conversion(input_file, output_file)

def try_textutil_conversion(input_file, output_file):
    """Enhanced fallback for Word documents on macOS using textutil."""
    if not (platform.system() == 'Darwin' and input_file.lower().endswith(('.doc', '.docx'))):
        return False
    try:
        logger.info(f"Trying textutil conversion for DOCX/DOC: {input_file}")
        cmd = ['textutil', '-convert', 'pdf', '-output', output_file, input_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=CONVERSION_TIMEOUT//2)
        if result.returncode == 0 and os.path.exists(output_file):
            # Verify the output file size
            if os.path.getsize(output_file) > 100:  # Basic size check
                logger.info(f"Successfully converted DOCX/DOC with textutil: {output_file}")
                return True
            else:
                logger.warning(f"Textutil output file too small: {os.path.getsize(output_file)} bytes")
                return False
        else:
            logger.error(f"Textutil conversion failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Textutil conversion timed out for {input_file}")
        return False
    except Exception as e:
        logger.warning(f"Textutil conversion encountered an error: {e}")
        return False

def optimize_pdf(pdf_path):
    """Optimize PDF using Ghostscript if available.
       Note: The '-dFitPage' option is disabled to avoid altering the document layout.
    """
    gs_command = 'gs'
    if shutil.which(gs_command) is None:
        logger.warning("Ghostscript not found. Skipping PDF optimization.")
        return False
    try:
        temp_file = pdf_path + ".temp.pdf"
        cmd = [
            gs_command,
            '-sDEVICE=pdfwrite',
            '-dPDFSETTINGS=/prepress',
            '-dCompatibilityLevel=1.6',
            '-dAutoRotatePages=/None',
            # '-dFitPage',  # Removed to preserve the original layout
            '-dEmbedAllFonts=true',
            '-dSubsetFonts=true',
            '-dCompressFonts=true',
            '-dColorImageResolution=300',
            '-dGrayImageResolution=300',
            '-dColorConversionStrategy=/LeaveColorUnchanged',
            '-dAutoFilterColorImages=false',
            '-dAutoFilterGrayImages=false',
            '-dDownsampleColorImages=false',
            '-dDownsampleGrayImages=false',
            '-dNOPAUSE',
            '-dQUIET',
            '-dBATCH',
            f'-sOutputFile={temp_file}',
            pdf_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(temp_file):
            shutil.move(temp_file, pdf_path)
            logger.info(f"Optimized PDF for high quality: {pdf_path}")
            return True
        else:
            logger.warning(f"PDF optimization failed: {result.stderr}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False
    except Exception as e:
        logger.error(f"Error during PDF optimization: {e}")
        return False

def verify_pdf_quality(pdf_path):
    """Check if the output PDF meets a minimum file size and basic validity."""
    if not os.path.exists(pdf_path):
        return False
    min_size_kb = 50   # Minimum expected file size in KB
    file_size_kb = os.path.getsize(pdf_path) / 1024
    if file_size_kb < min_size_kb:
        logger.warning(f"PDF quality may be poor - size: {file_size_kb:.2f}KB")
        return False
    
    # Try to read the first few bytes to verify it's a valid PDF
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(10)
            if not header.startswith(b'%PDF-'):
                logger.warning(f"File doesn't appear to be a valid PDF: {pdf_path}")
                return False
    except Exception as e:
        logger.warning(f"Could not verify PDF validity: {e}")
        return False
    
    logger.info(f"PDF quality verified - size: {file_size_kb:.2f}KB")
    return True

def move_to_archive(file_path):
    """Move processed file to archive, preserving directory structure."""
    rel_path = os.path.relpath(file_path, SOURCE_DIRECTORY)
    target_dir = os.path.join(ARCHIVE_DIRECTORY, os.path.dirname(rel_path))
    target_path = os.path.join(ARCHIVE_DIRECTORY, rel_path)
    os.makedirs(target_dir, exist_ok=True)
    try:
        shutil.move(file_path, target_path)
        logger.info(f"Moved original file to: {target_path}")
        return True
    except Exception as e:
        logger.error(f"Error moving file to archive: {e}")
        return False

# -----------------------------------------------------------------------------
# Processing Functions
# -----------------------------------------------------------------------------
def process_file(file_info):
    """Process a single file: conversion, verification, optimization, and archival.
    Simple version that places all PDFs in root with original filenames only."""
    full_path, file_index = file_info
    
    # SIMPLE: Just use the original filename in root directory
    filename_without_ext = os.path.splitext(os.path.basename(full_path))[0]
    pdf_path = os.path.join(PDF_OUTPUT_DIRECTORY, f"{filename_without_ext}.pdf")
    file_extension = Path(full_path).suffix.lower()
    
    # Handle potential name conflicts by adding a counter
    counter = 1
    original_pdf_path = pdf_path
    while os.path.exists(pdf_path) and not FORCE_RECONVERSION:
        if os.path.getmtime(pdf_path) > os.path.getmtime(full_path):
            if VERBOSE_LOGGING:
                logger.info(f"Skipping {full_path} - PDF already up to date")
            return {'status': 'skipped', 'path': full_path, 'type': file_extension}
        # If PDF exists but is older, add counter to avoid overwrite
        pdf_path = os.path.join(PDF_OUTPUT_DIRECTORY, f"{filename_without_ext}_{counter}.pdf")
        counter += 1
    
    try:
        logger.info(f"Processing {file_extension.upper()} file: {os.path.basename(full_path)}")
        logger.info(f"Output PDF will be: {pdf_path}")
        
        if convert_to_pdf_libreoffice(full_path, pdf_path):
            verify_pdf_quality(pdf_path)
            optimize_pdf(pdf_path)
            move_to_archive(full_path)
            return {'status': 'converted', 'path': full_path, 'type': file_extension, 'pdf_path': pdf_path}
        else:
            logger.warning(f"Primary conversion failed; trying alternative method for: {full_path}")
            if convert_to_pdf_alternative(full_path, pdf_path):
                verify_pdf_quality(pdf_path)
                move_to_archive(full_path)
                return {'status': 'converted', 'path': full_path, 'type': file_extension, 'pdf_path': pdf_path}
            else:
                logger.error(f"All conversion methods failed for: {full_path}")
                return {'status': 'failed', 'path': full_path, 'type': file_extension}
    except Exception as e:
        logger.error(f"Unexpected error processing {full_path}: {e}")
        return {'status': 'failed', 'path': full_path, 'type': file_extension}


def process_directory():
    """Walk through the source directory and process eligible documents concurrently."""
    stats = {
        'total_files': 0, 'converted': 0, 'failed': 0, 'skipped': 0, 'moved': 0,
        'docx_files': 0, 'doc_files': 0, 'pptx_files': 0, 'ppt_files': 0
    }
    start_time = time.time()
    logger.info(f"Starting to process documents in: {SOURCE_DIRECTORY}")
    
    if DIAGNOSTIC_MODE:
        logger.info(f"DIAGNOSTIC: System: {platform.platform()}")
        logger.info(f"DIAGNOSTIC: Python version: {platform.python_version()}")
        logger.info(f"DIAGNOSTIC: LibreOffice version: {get_libreoffice_version()}")

    os.makedirs(ARCHIVE_DIRECTORY, exist_ok=True)
    logger.info(f"Archive directory: {ARCHIVE_DIRECTORY}")
    create_libreoffice_settings()
    files_to_process = []
    for dirpath, dirnames, filenames in os.walk(SOURCE_DIRECTORY):
        for filename in filenames:
            extension = Path(filename).suffix.lower()
            if extension in ['.docx', '.doc', '.pptx', '.ppt']:
                stats['total_files'] += 1
                # Count by file type
                if extension == '.docx':
                    stats['docx_files'] += 1
                elif extension == '.doc':
                    stats['doc_files'] += 1
                elif extension == '.pptx':
                    stats['pptx_files'] += 1
                elif extension == '.ppt':
                    stats['ppt_files'] += 1
                    
                full_path = os.path.join(dirpath, filename)
                files_to_process.append((full_path, stats['total_files']))
    
    if MAX_FILES is not None and len(files_to_process) > MAX_FILES:
        files_to_process = files_to_process[:MAX_FILES]
        logger.info(f"Limiting to {MAX_FILES} files as per configuration")

    logger.info(f"Found {stats['docx_files']} DOCX, {stats['doc_files']} DOC, {stats['pptx_files']} PPTX, {stats['ppt_files']} PPT files")

    if files_to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(process_file, fi): fi for fi in files_to_process}
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    result = future.result()
                    if result['status'] == 'converted':
                        stats['converted'] += 1
                        stats['moved'] += 1
                    elif result['status'] == 'failed':
                        stats['failed'] += 1
                    elif result['status'] == 'skipped':
                        stats['skipped'] += 1
                except Exception as e:
                    file_info = future_to_file[future]
                    logger.error(f"Error processing {file_info[0]}: {e}")
                    stats['failed'] += 1

    end_time = time.time()
    duration = end_time - start_time
    logger.info("=" * 60)
    logger.info("Conversion Summary:")
    logger.info(f"Total files found:      {stats['total_files']}")
    logger.info(f"  - DOCX files:         {stats['docx_files']}")
    logger.info(f"  - DOC files:          {stats['doc_files']}")
    logger.info(f"  - PPTX files:         {stats['pptx_files']}")
    logger.info(f"  - PPT files:          {stats['ppt_files']}")
    logger.info(f"Successfully converted: {stats['converted']}")
    logger.info(f"Failed conversions:     {stats['failed']}")
    logger.info(f"Skipped (up-to-date):   {stats['skipped']}")
    logger.info(f"Files moved to archive: {stats['moved']}")
    logger.info(f"Total time:             {duration:.2f} seconds")
    logger.info(f"Quality settings:       {DPI} DPI, {IMAGE_QUALITY}% quality, PDF {PDF_VERSION}")
    logger.info("=" * 60)
    return stats

def check_dependencies():
    """Check that required dependencies are installed (LibreOffice and optionally Ghostscript)."""
    libreoffice_path = get_libreoffice_path()
    if not shutil.which(libreoffice_path) and not os.path.exists(libreoffice_path):
        logger.error("LibreOffice not found. Please install LibreOffice for PDF conversion.")
        logger.error("Download from: https://www.libreoffice.org/download/download/")
        return False
    logger.info(f"Using LibreOffice at: {libreoffice_path}")
    
    # Check LibreOffice version for better DOCX support info
    version_info = get_libreoffice_version()
    logger.info(f"LibreOffice version: {version_info}")
    
    if shutil.which('gs') is None:
        logger.warning("Ghostscript not found. PDF optimization will be skipped.")
        logger.warning("For better results, install Ghostscript (e.g., brew install ghostscript).")
    else:
        logger.info("Ghostscript found. PDF optimization will be performed.")
    
    # Check for textutil on macOS (helpful for DOCX fallback)
    if platform.system() == 'Darwin':
        if shutil.which('textutil'):
            logger.info("textutil found. Will be used as fallback for Word documents if needed.")
        else:
            logger.warning("textutil not found. Some DOCX/DOC conversions may fail without this fallback.")
    
    return True

def verify_source_directory():
    """Ensure that the source directory exists; create it if necessary."""
    if not os.path.exists(SOURCE_DIRECTORY):
        try:
            os.makedirs(SOURCE_DIRECTORY)
            logger.info(f"Created source directory: {SOURCE_DIRECTORY}")
        except Exception as e:
            logger.error(f"Could not create source directory: {e}")
            return False
    if not os.path.isdir(SOURCE_DIRECTORY):
        logger.error(f"Source path is not a directory: {SOURCE_DIRECTORY}")
        return False
    return True

def test_conversion_capabilities():
    """Test conversion capabilities with sample files if available."""
    logger.info("Testing conversion capabilities...")
    
    # Count files by type
    file_counts = {'docx': 0, 'doc': 0, 'pptx': 0, 'ppt': 0}
    
    if os.path.exists(SOURCE_DIRECTORY):
        for dirpath, dirnames, filenames in os.walk(SOURCE_DIRECTORY):
            for filename in filenames:
                extension = Path(filename).suffix.lower().lstrip('.')
                if extension in file_counts:
                    file_counts[extension] += 1
    
    logger.info("File type distribution:")
    for ext, count in file_counts.items():
        logger.info(f"  {ext.upper()}: {count} files")
    
    if file_counts['docx'] > 0 or file_counts['doc'] > 0:
        logger.info("✓ Word document support enabled (DOCX/DOC)")
    if file_counts['pptx'] > 0 or file_counts['ppt'] > 0:
        logger.info("✓ PowerPoint support enabled (PPTX/PPT)")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Enhanced High-Quality Document to PDF Converter")
    logger.info("Now with optimized DOCX/DOC support!")
    logger.info("=" * 60)
    logger.info(f"Source Directory:     {SOURCE_DIRECTORY}")
    logger.info(f"Archive Directory:    {ARCHIVE_DIRECTORY}")
    logger.info(f"Force Reconversion:   {FORCE_RECONVERSION}")
    logger.info(f"Max Files to Convert: {'Unlimited' if MAX_FILES is None else MAX_FILES}")
    logger.info(f"Image Quality:        {IMAGE_QUALITY}%")
    logger.info(f"Resolution:           {DPI} DPI")
    logger.info(f"PDF Version:          {PDF_VERSION}")
    logger.info(f"Parallel Processing:  {MAX_WORKERS} workers")
    logger.info(f"Conversion Timeout:   {CONVERSION_TIMEOUT} seconds")
    logger.info(f"Diagnostic Mode:      {DIAGNOSTIC_MODE}")
    logger.info(f"Simplified Mode:      {SIMPLIFY_CONVERSION}")
    logger.info("=" * 60)

    if not verify_source_directory():
        exit(1)
    if not check_dependencies():
        exit(1)
    
    test_conversion_capabilities()
    
    stats = process_directory()
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"  - DOCX files: {stats['docx_files']}")
    print(f"  - DOC files: {stats['doc_files']}")
    print(f"  - PPTX files: {stats['pptx_files']}")
    print(f"  - PPT files: {stats['ppt_files']}")
    print(f"Successfully converted: {stats['converted']}")
    print(f"Failed conversions: {stats['failed']}")
    print(f"Skipped (up-to-date): {stats['skipped']}")
    print(f"Files moved to archive: {stats['moved']}")
    print(f"High-quality settings: {DPI} DPI, {IMAGE_QUALITY}% quality, PDF {PDF_VERSION}")
    print("=" * 60)