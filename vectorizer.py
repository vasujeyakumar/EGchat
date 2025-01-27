import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import json

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def collect_image_paths(image_dirs):
    image_paths = []
    for image_dir in image_dirs:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    image_paths.append(os.path.join(root, file))
    return image_paths
def embedding_text(text):
    inputs = clip_processor(text=[text], images=[Image.new("RGB", (224, 224))], return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        return outputs.text_embeds.numpy()

def embedding_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[""], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        return outputs.image_embeds.numpy()

def create_vector_database(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index
def load_data_locally(data_dir="data"):
    # Load text data from the saved JSON file
    text_file_path = os.path.join(data_dir, "text_data.json")
    with open(text_file_path, "r") as text_file:
        text_data = json.load(text_file)

    # Load image paths from the saved text file
    image_file_path = os.path.join(data_dir, "image_paths.txt")
    with open(image_file_path, "r") as image_file:
        image_data = [line.strip() for line in image_file.readlines()]

    return text_data, image_data
