import os

class Config:
    TESSERACT_PATH = r"C:\Users\Admin\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111.exe" # Update as per your environment
    PDF_PATHS = [
        "Autodesk Inventor Practice Part Drawings.pdf",
        "Engineering+Working+Drawing+Basics.pdf",
    ]
    BASE_OUTPUT_DIR = "extra"

    @staticmethod
    def validate_tesseract_path():
        if not os.path.exists(Config.TESSERACT_PATH):
            raise Exception(f"Tesseract not found at {Config.TESSERACT_PATH}")