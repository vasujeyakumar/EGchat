import os
import json
import pandas as pd

class DataPreprocessor:
    def __init__(self, json_paths):
        self.json_paths = json_paths
        self.text_dataframes = []
        self.image_dataframes = []

    def load_data(self):
        for path in self.json_paths:
            for filename in os.listdir(path):
                if filename.endswith('.json'):
                    json_file_path = os.path.join(path, filename)
                    with open(json_file_path, 'r') as file:
                        data = json.load(file)
                        self.process_json_data(data)

    def process_json_data(self, data):
        # Process text data
        text_data = data.get("text_data", {})
        df_text = pd.DataFrame.from_dict(text_data, orient="index", columns=["Text"])
        self.text_dataframes.append(df_text)

        # Process image data
        image_data = data.get("image_data", [])
        df_image = pd.DataFrame(image_data)
        self.image_dataframes.append(df_image)

    def get_text_dataframes(self):
        return self.text_dataframes

    def get_image_dataframes(self):
        return self.image_dataframes
    def save_dataframes(self):
        # Save text dataframes to CSV
        for i, df in enumerate(self.text_dataframes):
            text_file_path = os.path.join(self.output_dir, f'text_data_{i}.csv')
            df.to_csv(text_file_path, index=True)
            print(f'Saved text DataFrame {i} to {text_file_path}')

        # Save image dataframes to CSV
        for i, df in enumerate(self.image_dataframes):
            image_file_path = os.path.join(self.output_dir, f'image_data_{i}.csv')
            df.to_csv(image_file_path, index=False)
            print(f'Saved image DataFrame {i} to {image_file_path}')