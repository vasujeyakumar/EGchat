from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from multimodal_search import multimodal_query  # Import your multimodal query function

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

def main():
    query = input("Enter your query: ")
    results = multimodal_query(query_text=query)  # You can add a query_image_path if needed
    print("Retrieved Results:", results)
    response = generate_response(results)
    print(response)

# Call main function to start interaction
if __name__ == "__main__":
    main()
