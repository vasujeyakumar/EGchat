import json

class Chatbot:
    def __init__(self, json_data_path):
        with open(json_data_path, "r") as file:
            self.data = json.load(file)

    def get_response(self, question):
        relevant_text = []
        relevant_images = []

        for page, content in self.data["text_data"].items():
            if question.lower() in content.lower():
                relevant_text.append((page, content))

        for image in self.data["image_data"]:
            if question.lower() in image["ocr_text"].lower():
                relevant_images.append(image)

        return relevant_text, relevant_images