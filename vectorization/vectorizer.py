import os
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class Vectorizers:
    def __init__(self, output_dir="data/vector_db"):
        self.output_dir = output_dir  # Directory to save the FAISS index
        self.model_name = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

    def embed_text(self, text):
        """Embeds text using the CLIP model."""
        if not isinstance(text, str) or not text.strip():
            return np.zeros((1, self.model.config.projection_dim))

        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.detach().numpy().flatten()

    def embed_image(self, image_path):
        """Embeds an image using the CLIP model."""
        if not os.path.exists(image_path):
            print(f"Warning: Image path does not exist - {image_path}")
            return np.zeros((1, self.model.config.projection_dim))

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.detach().numpy().flatten()

    def setup_embeddings(self, text_dataframes, image_dataframes):
        """Sets up embeddings for text and images and creates a FAISS index."""
        all_embeddings = []
        combined_data = []

        for text_df, image_df in zip(text_dataframes, image_dataframes):
            # Process each row in the text DataFrame
            for index, row in text_df.iterrows():
                text_string = row['Text']
                text_embedding = self.embed_text(text_string)
                if text_embedding.sum() == 0:  # Skip empty text embeddings
                    print(f"Skipping empty text embedding for row {index}")
                    continue

                # Process each row in the image DataFrame
                for img_row in image_df.itertuples(index=False):
                    image_path = img_row.image_path  # Adjust based on your JSON structure
                    image_embedding = self.embed_image(image_path)

                    if image_embedding.sum() == 0:  # Skip empty image embeddings
                        print(f"Skipping empty image embedding for path {image_path}")
                        continue

                    # Combine text and image embeddings
                    combined_embedding = np.concatenate((text_embedding, image_embedding), axis=-1)
                    all_embeddings.append(combined_embedding)
                    combined_data.append({"image_path": image_path, "text_data": text_string})

        if not all_embeddings:  # If no embeddings are generated
            print("No embeddings generated. Check your data and preprocessing.")
            return None, None

        # Create FAISS index
        embeddings_array = np.array(all_embeddings).astype('float32')
        embedding_dim = embeddings_array.shape[-1]  # This should be 1024 (512 + 512)
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_array)

        return index, combined_data

    def save_vector_db(self, index):
        """Save the FAISS index to the specified directory."""
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure the directory exists
        vector_db_file_path = os.path.join(self.output_dir, "vector_index.faiss")
        
        try:
            if index.ntotal > 0:  # Check if the index contains any vectors
                faiss.write_index(index, vector_db_file_path)
                print(f"Vector database saved to {vector_db_file_path}")
            else:
                print("Warning: Vector index is empty. Nothing to save.")
        except Exception as e:
            print(f"Failed to save the vector database: {e}")
