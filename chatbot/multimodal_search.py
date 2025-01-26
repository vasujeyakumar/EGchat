from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import numpy as np
import faiss
from vectorization.vectorizer import Vectorizers
from transformers import CLIPProcessor, CLIPModel
from preprocessing.data_preprocessor import DataPreprocessor
import os
# Paths and Preprocessing
json_paths = [
    r"data/extracted/Engineering+Working+Drawing+Basics",
    r"data/extracted/Autodesk Inventor Practice Part Drawings"
]

preprocessor = DataPreprocessor(json_paths)
preprocessor.load_data()

text_dataframes = preprocessor.get_text_dataframes()
image_dataframes = preprocessor.get_image_dataframes()

# Model Initialization
model_name = "openai/clip-vit-base-patch32"
models = CLIPModel.from_pretrained(model_name)

# Initialize the LLM (ensure your API key is set)
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    api_key="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",  # Replace with your actual API key
    temperature=0.6,
)

# Vectorizer Instance
vector = Vectorizers()

# Define the FAISS index
vector_db_path = "data/vector_db/vector_index.faiss"

# Check if the vector database exists
if os.path.exists(vector_db_path):
    index = faiss.read_index(vector_db_path)
    print("Vector database loaded successfully.")
else:
    print("Vector database not found. Creating a new index...")

    # Generate embeddings and build the index
    combined_data = vector.setup_embeddings(text_dataframes, image_dataframes)
    index = faiss.IndexFlatL2(vector.get_embedding_dim())
    
    if combined_data:
        embeddings = combined_data["embeddings"]
        index.add(embeddings)
        faiss.write_index(index, vector_db_path)
        print("Vector database created and saved successfully.")
    else:
        print("No embeddings available to create an index.")
        index = None

# Define the enhanced prompt template
prompt_template = PromptTemplate(
    template=(
        "You are a knowledgeable assistant that can provide detailed descriptions and "
        "diagrams based on queries. Given the following results:\n"
        "{results}\n\n"
        "Please respond with the following:\n"
        "- A meaningful and context-rich description of the relevant information.\n"
        "- If the user specifically asks for a diagram, provide the image path and a brief explanation of the diagram's relevance.\n"
        "- If no relevant information is found in the documents, inform the user that the specific information is not available, "
        "and provide a general answer based on your knowledge about the topic.\n"
        "- Feel free to include any additional insights or related information that may help the user understand better."
    ),
    input_variables=["results"]
)

def normalize_embedding(embedding):
    """Normalize embeddings to unit vectors."""
    norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
    # Prevent division by zero
    return embedding / norm if norm.any() else embedding

def multimodal_query(query_text=None, query_image_path=None, k=5):
    """Perform a similarity search based on a multimodal query."""
    if index is None:
        raise ValueError("FAISS index is not initialized. Please ensure embeddings are generated.")

    text_embedding = np.zeros((1, models.config.projection_dim))
    image_embedding = np.zeros((1, models.config.projection_dim))

    if query_text:
        text_embedding = vector.embed_text(query_text).reshape(1, -1)
    if query_image_path:
        image_embedding = vector.embed_image(query_image_path).reshape(1, -1)

    query_embedding_combined = np.concatenate((text_embedding, image_embedding), axis=-1)
    query_embedding_combined = normalize_embedding(query_embedding_combined)

    distances, indices = index.search(query_embedding_combined, k)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        # Ensure the index is valid
        if 0 <= idx < len(image_dataframes) and 0 <= idx < len(text_dataframes):
            results.append({
                "distance": distance,
                "image_path": image_dataframes[idx]["image_path"],
                "text_data": text_dataframes[idx]["text_data"]
            })
        else:
            print(f"Index {idx} is out of bounds for image_dataframes or text_dataframes.")

    return results


def generate_response(results):
    if results:
        results_str = "\n".join([f"Diagram: {result['image_path']}, Description: {result['text_data']}" for result in results])
    else:
        results_str = "No relevant information found."

    prompt = prompt_template.format(results=results_str)

    print("Generated Prompt:\n", prompt)

    try:
        response = llm.invoke(prompt)  # Use invoke instead of __call__
        return response
    except ValueError as ve:
        return f"Value error occurred: {ve}"
    except TypeError as te:
        return f"Type error occurred: {te}"
    except Exception as e:
        return f"An error occurred while generating a response: {e}"
