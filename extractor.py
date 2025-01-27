import fitz  # PyMuPDF
import io  # To handle byte streams
from PIL import Image
import pytesseract
import pdfplumber
import json
import os

# Set Tesseract path (if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Default path for Colab

def extract_text(pdf_path):
    text_data = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text_data[page_num] = text.strip()
    return text_data

def extract_images_with_ocr(pdf_path, output_dir, ocr_lang="eng"):
    image_data = []
    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            ocr_text = pytesseract.image_to_string(image, lang=ocr_lang)

            img_info = {
                "page": page_num + 1,
                "width": image.width,
                "height": image.height,
                "image_index": img_index,
                "ext": base_image["ext"],
                "dpi": image.info.get("dpi", (72, 72)),
                "ocr_text": ocr_text.strip(),
                "image_path": f"{pdf_name}_page_{page_num + 1}_image_{img_index}.{base_image['ext']}"
            }
            image_data.append(img_info)

            img_filename = f"{pdf_name}_page_{page_num + 1}_image_{img_index}.{base_image['ext']}"
            img_path = os.path.join(output_dir, img_filename)
            image.save(img_path)
    return image_data

def save_to_json(output_path, text_data, image_data):
    data = {
        "text_data": text_data,
        "image_data": image_data,
    }
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def parse_pdfs(pdf_paths, base_output_dir, ocr_lang="eng"):
    for pdf_path in pdf_paths:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(base_output_dir, pdf_name)
        output_json_path = os.path.join(output_dir, f"{pdf_name}_data.json")

        print(f"Processing PDF: {pdf_path}")
        os.makedirs(output_dir, exist_ok=True)

        text_data = extract_text(pdf_path)
        image_data = extract_images_with_ocr(pdf_path, output_dir, ocr_lang)

        save_to_json(output_json_path, text_data, image_data)
        print(f"Completed processing for {pdf_name}. Results saved to {output_json_path}.\n")

# Example Usage
if __name__ == "__main__":
    pdf_paths = [
        "Autodesk Inventor Practice Part Drawings.pdf",  # Ensure these names match your uploaded files
        "Engineering+Working+Drawing+Basics.pdf",
    ]
    base_output_dir = "extra_newdata"

    parse_pdfs(pdf_paths, base_output_dir, ocr_lang="eng")