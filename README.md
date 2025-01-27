# Chatbot with CLIP and FAISS

## Description
This project is a chatbot capable of processing both textual and image-based queries. It utilizes OpenAI's CLIP model for embedding generation and FAISS for efficient similarity-based retrieval.

## Features
- **Textual Information Search**: Provides accurate responses based on textual input queries.
- **Image Search**: Retrieves relevant image paths for queries describing visual content.
- **Multi-modal Retrieval**: Merges text and image search for enhanced query resolution.
- **Efficient Search Mechanism**: Leverages FAISS for fast and precise similarity searches.
- **User-friendly Interface**: Designed for simplicity and accessibility.

## Technologies Used
- **Programming Language**: Python 3.8+
- **Machine Learning Frameworks**:
  - OpenAI's CLIP model for text-image embedding.
  - PyTorch for model integration and computations.
- **Search Library**: FAISS for indexing and similarity search.
- **Libraries for Preprocessing**:
  - Pillow for image processing.
  - pdfplumber and PyMuPDF for handling and extracting text from PDFs.
  - pytesseract for OCR text extraction from images.

## Requirements

### System Requirements
- Python 3.8 or higher.

### Required Libraries
- torch
- transformers
- faiss
- numpy
- pillow
- pdfplumber
- pytesseract
- PyMuPDF

To install these dependencies, run:

```bash
pip install -r requirements.txt
