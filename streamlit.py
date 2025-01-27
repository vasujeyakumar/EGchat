import os
import faiss
import numpy as np
from PIL import Image
import streamlit as st
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq  # Import ChatGroq for Groq integration
from retriver import retrieve_similar_documents, load_data_locally

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the vector databases and data
text_vector_db = faiss.read_index("text_vector_db.index")
image_vector_db = faiss.read_index("image_vector_db.index")
text_data, image_paths = load_data_locally("files")

# Setup Langchain ChatGroq model with your API key
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Adjust model name if needed
    api_key="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",  # Replace with your actual ChatGroq API key
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
    similar_images = retrieve_similar_documents(query, image_vector_db, image_paths, k=1)

    # Format the results
    similar_texts_formatted = "\n".join(similar_texts) if similar_texts else "No relevant texts found."
    similar_images_formatted = "\n".join(similar_images) if similar_images else "No relevant images found."

    # Generate a response using the LLM
    response = chain.run(query=query, similar_texts=similar_texts_formatted, similar_images=similar_images_formatted)

    return response, similar_images

# Streamlit UI
def main():
    # Set up the title and header
    st.title("Engineering Query Chatbot")
    st.header("Ask me anything about engineering diagrams and texts!")

    # User input
    query = st.text_input("Enter your query:")

    # Query history
    if "history" not in st.session_state:
        st.session_state.history = []

    if query:
        # Show loading spinner
        with st.spinner("Processing your query..."):
            response, similar_images = chatbot(query)

        # Add current query and response to history
        st.session_state.history.append((query, response))

        # Display the response
        st.subheader("Chatbot Response:")
        st.write(response)

        # Display related images
        if similar_images:
            st.subheader("Related Images:")
            for img_path in similar_images:
                img = Image.open(img_path)
                st.image(img, caption=f"Click to view {os.path.basename(img_path)}", use_column_width=True)
                st.markdown(f'<a href="{img_path}" target="_blank">Open image in new tab</a>', unsafe_allow_html=True)

        # Show query history
        st.subheader("Query History")
        for i, (q, r) in enumerate(st.session_state.history):
            st.write(f"**Query {i+1}:** {q}")
            st.write(f"**Response:** {r}")
            st.write("---")

        # Feedback
        feedback = st.radio("Was the answer helpful?", options=["Yes", "No"])
        if feedback:
            st.write(f"Thank you for your feedback: {feedback}")

        # Download response as a text file
        def download_text(response):
            return response.encode("utf-8")

        st.download_button(
            label="Download Response as Text",
            data=download_text(response),
            file_name="response.txt",
            mime="text/plain"
        )

# Run the Streamlit app
if __name__ == "__main__":
    main()
