import os
from parsers.pdfparser import PDFParser
from preprocessing.data_preprocessor import DataPreprocessor
from vectorization.vectorizer import Vectorizers
from chatbot.multimodal_search import multimodal_query 
import faiss  # Ensure to import faiss for loading the index

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
    if not os.path.exists(vector_db_path):
        # Generate and save embeddings
        vectorizer = Vectorizers(output_dir="data/vector_db")
        index, combined_data = vectorizer.setup_embeddings(text_dataframes, image_dataframes)

        if index is not None:
            vectorizer.save_vector_db(index)
            print("Embeddings generated and saved.")
        else:
            print("No embeddings generated. Skipping saving step.")
    else:
        # Load the existing vector index
        print("Loading existing vector database...")
        index = faiss.read_index(r"data\vector_db\vector_index.faiss")
        print("Vector database loaded successfully.")

    # Step 4: Interactive chatbot query
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Perform multimodal retrieval
        results = multimodal_query(query_text=query)
        
        # Process and display results
        if results:
            for result in results:
                print(f"Distance: {result['distance']}")
                print(f"Image Path: {result['image_path']}")
                print(f"Text Data: {result['text_data']}")
                print("-" * 50)
        else:
            print("No relevant information found.")

if __name__ == "__main__":
    main()
