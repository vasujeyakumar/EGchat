import os
import json
import pandas as pd

def preprocess_data(pdf_paths, base_output_dir):
    text_dfs = []
    image_dfs = []

    for i, path in enumerate(pdf_paths):
        for filename in os.listdir(path):
            if filename.endswith(".json"):
                json_file_path = os.path.join(path, filename)
                with open(json_file_path, "r") as f:
                    data = json.load(f)

                text_data = data.get("text_data", {})
                df_text = pd.DataFrame.from_dict(text_data, orient="index", columns=["Text"])
                df_text["source"] = f"doc{i}"
                text_dfs.append(df_text)

                image_data = data.get("image_data", [])
                df_image = pd.DataFrame(image_data)
                df_image["source"] = f"doc{i}"
                image_dfs.append(df_image)

    combined_text_df = pd.concat(text_dfs, ignore_index=True)
    combined_image_df = pd.concat(image_dfs, ignore_index=True)

    combined_text_df["Text"] = combined_text_df["Text"].str.lower()
    combined_text_df["Text"] = combined_text_df["Text"].str.replace("/n", " ", regex=False)

    return combined_text_df, combined_image_df
