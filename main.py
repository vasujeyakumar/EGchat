import os
from parsers.pdfparser import PDFParser
from preprocessing.data_preprocessor import DataPreprocessor
from vectorization.vectorizer import Vectorizers
from chatbot.multimodal_search import multimodal_query, generate_response


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
    extracted_data_path = 'data/extracted/'
    vector_db_path = "data/vector_db/"

    # Step 1: Check for extracted files
    if not os.path.exists(extracted_data_path) or not os.listdir(extracted_data_path):
        print("No extracted files found. Running extraction process...")
        pdf_parser = PDFParser()
        pdf_parser.execute()
        print("Extraction complete.")
    else:
        print("Extracted files found. No need to run extraction.")

    # Step 2: Preprocess the extracted data
    json_paths = [
        r"data/extracted/Engineering+Working+Drawing+Basics",
        r"data/extracted/Autodesk Inventor Practice Part Drawings"
    ]

    preprocessor = DataPreprocessor(json_paths)
    preprocessor.load_data()

    text_dataframes = preprocessor.get_text_dataframes()
    image_dataframes = preprocessor.get_image_dataframes()

    # Step 3: Check if the vector database already exists
    if not os.path.exists(vector_db_path) or not os.path.isfile(f"{vector_db_path}/vector_index.faiss"):
        print("Vector database not found. Generating embeddings...")
        vectorizer = Vectorizers(output_dir=vector_db_path)
        index, combined_data = vectorizer.setup_embeddings(text_dataframes, image_dataframes)

        if index is not None:
            vectorizer.save_vector_db(index)
            print("Embeddings generated and saved.")
        else:
            print("No embeddings generated. Skipping saving step.")

    # Step 4: Interactive chatbot query
    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting. Goodbye!")
            break

        results = multimodal_query(query_text=query)  # Add query_image_path if needed
        response = generate_response(results)
        print(response)

if __name__ == "__main__":
    main()
