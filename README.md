# Chatbot with CLIP and FAISS

This project implements a chatbot capable of answering queries based on both textual and image data. The chatbot uses OpenAI's CLIP model for generating embeddings of images and text and FAISS (Facebook AI Similarity Search) for efficient similarity search.

## Features

- **Textual Information Search**: Retrieves relevant text responses based on user queries.
- **Image Search**: Retrieves relevant image paths based on user queries.
- **Multi-modal Retrieval**: Combines both image and text data to provide comprehensive responses.

## Requirements

- Python 3.8+
- Required libraries:
  - `torch`
  - `transformers`
  - `faiss`
  - `numpy`
  - `pillow`
  - `pdfplumber`
  - `pytesseract`
  - `PyMuPDF`
  
To install the dependencies, run:
```bash
pip install -r requirements.txt
