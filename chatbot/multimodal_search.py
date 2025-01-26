
from vectorization.vectorizer import Vectorizers  # Import from your vectorization module
import os
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
vector=Vectorizers()

# Initialize the model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)

# Load your combined data and index from the appropriate directory
VECTOR_DB_PATH = r"data/vector_db"
VECTOR_DB_FILE = "vector_index.faiss"

# Load the index
index = faiss.read_index(os.path.join(VECTOR_DB_PATH, VECTOR_DB_FILE))


def normalize_embedding(embedding):
    """Normalize embeddings to unit vectors."""
    return embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)

def multimodal_query(query_text=None, query_image_path=None, k=5):
    """Perform a similarity search based on a multimodal query."""
    text_embedding = np.zeros((1, model.config.projection_dim))
    image_embedding = np.zeros((1, model.config.projection_dim))

    if query_text:
        text_embedding = vector.embed_text(query_text).reshape(1, -1)
    if query_image_path:
        image_embedding =vector.embed_image(query_image_path).reshape(1, -1)

    query_embedding_combined = np.concatenate((text_embedding, image_embedding), axis=-1)
    query_embedding_combined = normalize_embedding(query_embedding_combined)

    distances, indices = index.search(query_embedding_combined, k)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "distance": distance,
            "image_path":image_dataframes[idx]["image_path"],
            "text_data":text_dataframes[idx]["text_data"]
        })

    return results
