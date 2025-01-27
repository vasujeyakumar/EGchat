import numpy as np
from vectorizer import embedding_text
import json
import os

def retrieve_similar_documents(query, vector_db, documents, k=5):
    query_embedding = embedding_text(query)
    query_embedding = query_embedding.reshape(1, -1)
    D, I = vector_db.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in I[0]]

def load_data_locally(data_dir="files"):
    # Load text data from the saved JSON file
    text_file_path = os.path.join(data_dir, "text_data.json")
    with open(text_file_path, "r") as text_file:
        text_data = json.load(text_file)

    # Load image paths from the saved text file
    image_file_path = os.path.join(data_dir, "image_paths.txt")
    with open(image_file_path, "r") as image_file:
        image_data = [line.strip() for line in image_file.readlines()]

    return text_data, image_data
