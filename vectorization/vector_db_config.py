import os

class VectorDBConfig:
    VECTOR_DB_PATH = r"data/vector_db"
    VECTOR_DB_FILE = "vector_index.faiss"

    @classmethod
    def check_vector_db(cls):
        # Check if the specific vector index file exists
        return os.path.exists(os.path.join(cls.VECTOR_DB_PATH, cls.VECTOR_DB_FILE))




