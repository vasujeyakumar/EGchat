import os
from parsers.pdf_config import Config

def main():
    extracted_data_path = 'data/extracted/'

    # Check for existing extracted data
    if not os.path.exists(extracted_data_path) or not os.listdir(extracted_data_path):
        print("No extracted files found. Running extraction process...")
        Config.execute()  # Execute the PDF parsing process
        print("Extraction complete.")
    else:
        print("Extracted files found. No need to run extraction.")

if __name__ == "__main__":
    main()
