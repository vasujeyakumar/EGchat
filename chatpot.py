import json

class Chatbot:
    def _init_(self, json_data_path):
        try:
            with open(json_data_path, "r") as file:
                self.data = json.load(file)
        except FileNotFoundError:
            raise Exception(f"File not found: {json_data_path}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON format in file: {json_data_path}")

    def get_response(self, question):
        if not question.strip():
            return [], []

        relevant_text = []
        relevant_images = []

        for page, content in self.data.get("text_data", {}).items():
            if question.lower() in content.lower():
                relevant_text.append((page, content))

        for image in self.data.get("image_data", []):
            if question.lower() in image.get("ocr_text", "").lower():
                relevant_images.append(image)

        return relevant_text,relevant_images