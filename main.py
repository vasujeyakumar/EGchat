import os
import faiss
import numpy as np
from preprocessor import preprocess_data
from vectorizer import embedding_text, embedding_image, create_vector_database,collect_image_paths
from retriver import retrieve_similar_documents
import json

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def save_vector_database(index, filename):
    faiss.write_index(index, filename)
    print(f"Vector database saved at: {filename}")

def save_data_locally(text_data, image_data, output_dir="files"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save text data as JSON
    text_file_path = os.path.join(output_dir, "text_data.json")
    with open(text_file_path, "w") as text_file:
        json.dump(text_data, text_file, indent=4)

    # Save image paths
    image_file_path = os.path.join(output_dir, "image_paths.txt")
    with open(image_file_path, "w") as image_file:
        for img in image_data:
            image_file.write(f"{img}\n")
    
    print(f"Text data saved to {text_file_path}")
    print(f"Image paths saved to {image_file_path}")

def main(base_output_dir, image_dirs, query):
    # Collect image paths from the provided directories
    image_paths = collect_image_paths(image_dirs)

    print("Preprocessing the data...")
    combined_text_df, combined_image_df = preprocess_data(image_dirs, base_output_dir)

    # Generate embeddings for the text data
    print("Generating text embeddings...")
    document_texts = combined_text_df["Text"].tolist()  # Adjust column name as needed
    document_embeddings = np.array([embedding_text(text) for text in document_texts]).squeeze()

    # Generate embeddings for the image data
    print("Generating image embeddings...")
    image_embeddings = np.array([embedding_image(img_path) for img_path in image_paths]).squeeze()

    # Create vector databases for text and image embeddings
    print("Creating vector databases...")
    vector_db_text = create_vector_database(document_embeddings)
    vector_db_image = create_vector_database(image_embeddings)

    # Save the vector databases to files for future use
    save_vector_database(vector_db_text, "text_vector_db.index")
    save_vector_database(vector_db_image, "image_vector_db.index")
    save_data_locally(document_texts,image_paths,"files")
    # Retrieve similar documents based on the query (text and image)
    print(f"Retrieving similar documents for query: {query}")

    # Retrieve similar texts (top 2 results)
    similar_texts = retrieve_similar_documents(query, vector_db_text, document_texts, k=1)
    print("Similar Texts:", similar_texts)

    # Retrieve similar images (top 2 results)
    similar_images = retrieve_similar_documents(query, vector_db_image, image_paths, k=1)
    print("Similar Images:", similar_images)

if __name__ == "__main__":  # List of directories with the extracted JSON files
    query = "beam diagram"  # Example query
    base_output_dir = r"C:/Users/Admin/Engineering_ai/EGchat/extra_newdata"  # Path to the output directory
    image_dirs = [
        r"C:\Users\Admin\Engineering_ai\EGchat\extra_newdata\Autodesk Inventor Practice Part Drawings",
        r"C:\Users\Admin\Engineering_ai\EGchat\extra_newdata\Engineering+Working+Drawing+Basics"
    ]

    main(base_output_dir, image_dirs, query)
