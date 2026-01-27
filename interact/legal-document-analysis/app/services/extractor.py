from PyPDF2 import PdfReader
import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageEnhance, ImageFilter
import io

# ===== COMMENTED OUT - UNUSED FULL PDF EXTRACTION FUNCTIONS =====
# These functions process entire PDFs but we only need first page extraction
# for the current requirements

# def extract_from_pdf(file: bytes) -> str:
#     """Extract text from normal PDF (not scanned)."""
#     data = ""
#     pdf_reader = PdfReader(io.BytesIO(file))
#     for page in pdf_reader.pages:
#         data += page.extract_text() or ""
#     return data

# def extract_from_scanned_pdf(file: bytes) -> str:
#     """Extract text from scanned PDF using OCR."""
#     text = ""
#     pdf = pdfium.PdfDocument(io.BytesIO(file))
#     for page_number in range(len(pdf)):
#         page = pdf.get_page(page_number)
#         img = page.render(scale=1, rotation=0).to_pil()
#         text += pytesseract.image_to_string(img, config="--oem 3 --psm 6")
#     return text

def extract_from_pdf_first_page_only(file: bytes) -> str:
    """Extract text from first page of normal PDF only."""
    pdf_reader = PdfReader(io.BytesIO(file))
    if len(pdf_reader.pages) > 0:
        return pdf_reader.pages[0].extract_text() or ""
    return ""

def extract_from_scanned_pdf_first_page_only(file: bytes) -> str:
    """Extract text from first page of scanned PDF using OCR with preprocessing."""
    pdf = pdfium.PdfDocument(io.BytesIO(file))
    if len(pdf) > 0:
        page = pdf.get_page(0)  # First page only
        img = page.render(scale=1, rotation=0).to_pil()
        # Preprocess for better OCR
        processed_img = preprocess_image_for_ocr(img)
        # Try multiple PSM modes
        results = []
        for psm in [3, 11, 6]:
            try:
                text = pytesseract.image_to_string(processed_img, config=f"--oem 3 --psm {psm}")
                if text.strip():
                    results.append(text)
            except:
                pass
        if results:
            return max(results, key=len).strip()
        return pytesseract.image_to_string(processed_img, config="--oem 3 --psm 3")
    return ""

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy.
    
    Applies:
    - Grayscale conversion (better for OCR)
    - Contrast enhancement
    - Sharpness enhancement
    - Noise reduction
    - Upscaling for small images
    
    Note: Tesseract OCR works best with printed text. Handwritten text
    and text in circular stamps may have lower accuracy.
    """
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to grayscale for better OCR
    img = img.convert('L')  # Grayscale
    
    # Enhance contrast (helps with faint text and stamps)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast
    
    # Enhance brightness if image is too dark
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)  # Slight brightness increase
    
    # Enhance sharpness (helps with blurry images)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)  # Increase sharpness
    
    # Apply slight denoising (removes small artifacts)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Resize if image is too small (improves OCR accuracy significantly)
    # Tesseract works better with higher resolution images
    width, height = img.size
    if width < 1000 or height < 1000:
        scale_factor = max(1000 / width, 1000 / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img

def extract_from_image(file: bytes) -> str:
    """Extract text from image file (PNG, JPG, JPEG) using OCR with enhanced preprocessing."""
    try:
        img = Image.open(io.BytesIO(file))
        
        # Preprocess image for better OCR
        processed_img = preprocess_image_for_ocr(img)
        
        # Try multiple PSM modes for better accuracy
        # PSM 3 = Fully automatic page segmentation (default)
        # PSM 6 = Single uniform block of text
        # PSM 11 = Sparse text (for documents with stamps, annotations)
        # PSM 12 = Sparse text with OSD (Orientation and Script Detection)
        
        results = []
        
        # Try PSM 3 (automatic) - best for most documents
        try:
            text_psm3 = pytesseract.image_to_string(
                processed_img, 
                config="--oem 3 --psm 3"
            )
            if text_psm3.strip():
                results.append(text_psm3)
        except:
            pass
        
        # Try PSM 11 (sparse text) - better for documents with stamps/annotations
        try:
            text_psm11 = pytesseract.image_to_string(
                processed_img,
                config="--oem 3 --psm 11"
            )
            if text_psm11.strip():
                results.append(text_psm11)
        except:
            pass
        
        # Try PSM 6 (single block) - fallback
        try:
            text_psm6 = pytesseract.image_to_string(
                processed_img,
                config="--oem 3 --psm 6"
            )
            if text_psm6.strip():
                results.append(text_psm6)
        except:
            pass
        
        # Combine results, prioritizing longer/more complete extractions
        if results:
            # Return the longest result (usually most complete)
            best_result = max(results, key=len)
            return best_result.strip()
        
        # Fallback to basic extraction
        return pytesseract.image_to_string(processed_img, config="--oem 3 --psm 3")
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")