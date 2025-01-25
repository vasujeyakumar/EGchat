import os
import pytesseract
from parser import PDFParser
from chatpot import Chatbot
from confic import Config

if __name__ == "_main_":
    try:
        # Validate Tesseract installation
        Config.validate_tesseract_path()
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH

        # Extract and process PDFs
        parser = PDFParser()
        parser.parse_pdfs(Config.PDF_PATHS, Config.BASE_OUTPUT_DIR)

        # Initialize Chatbot
        json_data_path = os.path.join(
            Config.BASE_OUTPUT_DIR, 
            "Engineering+Working+Drawing+Basics", 
            "Engineering+Working+Drawing+Basics_data.json"
        )
        chatbot = Chatbot(json_data_path)

        # User Interaction
        question = input("Ask a question: ").strip()
        text_response, image_response = chatbot.get_response(question)

        print("\nRelevant Text:")
        if text_response:
            for page, content in text_response:
                print(f"Page {page}: {content}\n")
        else:
            print("No relevant text found.")

        print("\nRelevant Images:")
        if image_response:
            for image in image_response:
                print(f"Page {image['page']} - Path: {image['image_path']}\n")
        else:
            print("No relevant images found.")

    except Exception as e:
        print(f"Error:{e}")
