import os
import json
import io
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

class PDFParser:
    def __init__(self, ocr_lang="eng"):
        self.ocr_lang = ocr_lang

    def extract_text(self, pdf_path):
        text_data = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        text_data[page_num] = text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text_data

    def extract_images_with_ocr(self, pdf_path, output_dir):
        image_data = []
        try:
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

                    ocr_text = pytesseract.image_to_string(image, lang=self.ocr_lang)

                    img_info = {
                        "page": page_num + 1,
                        "width": image.width,
                        "height": image.height,
                        "image_index": img_index,
                        "ext": base_image["ext"],
                        "dpi": image.info.get("dpi", (72, 72)),
                        "ocr_text": ocr_text.strip(),
                        "image_path": os.path.join(
                            output_dir, 
                            f"{pdf_name}_page{page_num + 1}_image{img_index}.{base_image['ext']}"
                        ),
                    }
                    image_data.append(img_info)

                    img_path = img_info["image_path"]
                    image.save(img_path)
        except Exception as e:
            print(f"Error extracting images from {pdf_path}: {e}")
        return image_data

    def save_to_json(self, output_path, text_data, image_data):
        try:
            data = {
                "text_data": text_data,
                "image_data": image_data,
            }
            with open(output_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
        except Exception as e:
            print(f"Error saving JSON file to {output_path}: {e}")

    def parse_pdfs(self, pdf_paths, base_output_dir):
        for pdf_path in pdf_paths:
            try:
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_dir = os.path.join(base_output_dir, pdf_name)
                output_json_path = os.path.join(output_dir, f"{pdf_name}_data.json")

                print(f"Processing PDF: {pdf_path}")
                os.makedirs(output_dir, exist_ok=True)

                text_data = self.extract_text(pdf_path)
                image_data = self.extract_images_with_ocr(pdf_path, output_dir)

                self.save_to_json(output_json_path, text_data, image_data)
                print(f"Completed processing for {pdf_name}. Results saved to {output_json_path}.\n")
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {e}")
