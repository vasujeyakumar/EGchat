from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import numpy as np
import faiss
from vectorization.vectorizer import Vectorizers

# Initialize the LLM (ensure your API key is set)
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    api_key="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",  # Replace with your actual API key
    temperature=0.6,
)

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

def multimodal_query(query_text=None, query_image_path=None, k=5):
    """Perform a similarity search based on a multimodal query."""
    text_embedding = np.zeros((1, model.config.projection_dim))
    image_embedding = np.zeros((1, model.config.projection_dim))

    if query_text:
        text_embedding = vector.embed_text(query_text).reshape(1, -1)
    if query_image_path:
        image_embedding = vector.embed_image(query_image_path).reshape(1, -1)

    query_embedding_combined = np.concatenate((text_embedding, image_embedding), axis=-1)
    query_embedding_combined = normalize_embedding(query_embedding_combined)

    distances, indices = index.search(query_embedding_combined, k)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "distance": distance,
            "image_path": image_dataframes[idx]["image_path"],
            "text_data": text_dataframes[idx]["text_data"]
        })

    return results

def generate_response(results):
    if results:
        results_str = "\n".join([
            f"Diagram: {result['image_path']}, Description: {result['text_data']}" for result in results
        ])
    else:
        results_str = "No relevant information found."

    prompt = prompt_template.format(results=results_str)

    print("Generated Prompt:\n", prompt)

    try:
        response = llm(prompt)
        return response
    except Exception as e:
        return f"An error occurred while generating a response: {e}"
