from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from PIL import Image
import faiss
import numpy as np
import os
from retriver import retrieve_similar_documents, load_data_locally
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the vector databases
text_vector_db = faiss.read_index("text_vector_db.index")
image_vector_db = faiss.read_index("image_vector_db.index")
text_data, image_paths = load_data_locally("files")

# Setup Langchain ChatGroq model
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    api_key="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",
    temperature=0.6,
)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "similar_texts", "similar_images"],
    template=""" 
    You are a knowledgeable assistant that provides detailed and relevant information based on user queries related to engineering diagrams and texts.

    Query: {query}

    Here are some related texts:
    {similar_texts}

    Here are some related images (diagrams):
    {similar_images}

    Your response should include the following:
    1. Summarize the key points from the related texts.
    2. Describe any relevant diagrams, including their purpose and context.
    3. If the query specifically asks for a diagram, prioritize presenting the relevant diagrams first.
    4. If there are no relevant images or texts, clearly state that you couldn't find any information.

    Please provide a concise and informative response that directly addresses the user's query, incorporating both text and diagram insights when applicable.
    """
)

# Create the conversation chain
chain = LLMChain(llm=llm, prompt=prompt_template)

def chatbot(query):
    # Retrieve similar texts from vector DB (top 2 results)
    similar_texts = retrieve_similar_documents(query, text_vector_db, text_data, k=2)

    # Retrieve similar images from vector DB (top 2 results)
    similar_images = retrieve_similar_documents(query, image_vector_db, image_paths, k=3)

    # Format the results
    similar_texts_formatted = "\n".join(similar_texts) if similar_texts else "No relevant texts found."
    similar_images_formatted = "\n".join(similar_images) if similar_images else "No relevant images found."

    # Generate a response using the LLM
    response = chain.run(query=query, similar_texts=similar_texts_formatted, similar_images=similar_images_formatted)

    # Display the response
    print(response)

    # Show the images if any
    if similar_images:
        for img_path in similar_images:
            img = Image.open(img_path)
            img.show()

# Main loop for continuous querying
if __name__ == "__main__":
    while True:
        query = input("Please enter your query (type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break
        else:
            chatbot(query)
