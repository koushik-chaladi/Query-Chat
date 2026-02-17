from langchain_chroma import Chroma
from embedding import EmbeddingModel
import os

class VectorStore:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance

    def __init__ (self):
       self.connection = self._connect_to_vector_db()

    def _connect_to_vector_db(self):
        persist_directory = os.getenv("PERSIST_DIRECTORY", "vector_db")
        chroma_client = Chroma(persist_directory=persist_directory, embedding_function=EmbeddingModel())
        return chroma_client

    def get_connection(self):
        return self.connection

