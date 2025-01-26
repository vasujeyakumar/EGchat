import os
import pytesseract
from .pdfparser import PDFParser  # Import the PDFParser class

class Config:
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update as per your environment
    PDF_PATHS = [
        "Autodesk Inventor Practice Part Drawings.pdf",
        "Engineering+Working+Drawing+Basics.pdf",
    ]
    BASE_OUTPUT_DIR = "data/extracted"  # Change to your desired output directory

    @staticmethod
    def validate_tesseract_path():
        """Validates the Tesseract OCR installation path."""
        if not os.path.exists(Config.TESSERACT_PATH):
            raise Exception(f"Tesseract not found at {Config.TESSERACT_PATH}")

    @staticmethod
    def execute():
        """Executes the PDF parsing process."""
        Config.validate_tesseract_path()
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH

        parser = PDFParser()
        for pdf_path in Config.PDF_PATHS:
            parser.parse_pdf(pdf_path, Config.BASE_OUTPUT_DIR)

if __name__ == "__main__":
    Config.execute()
