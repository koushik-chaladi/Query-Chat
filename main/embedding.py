from langchain.embeddings import Embeddings
import numpy as np
import os
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import asyncio
from pathlib import Path


class EmbeddingModel(Embeddings):

    def __init__(self):
        self.model = SentenceTransformer(self._get_model_path())
        self.loop = asyncio.get_event_loop()

    def _get_model_path(self) -> str:
        model_path = Path(__file__).resolve().parent.parent
        print(Path(__file__).resolve())
        model_path = Path(model_path) / "models"/ "all-MiniLM-L6-v2.pt"
        return str(model_path)

    def _encode(self, texts) -> list[float]:
        return self.model.encode(texts).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(f"All items in the input list must be strings. Found item of type {type(text)}")
            else:
                encoded_texts = self._encode(texts)
                return encoded_texts

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise ValueError(f"Input must be a string. Found input of type {type(text)}")
        else:
            encoded_query = self._encode([text])[0]
            return encoded_query

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(f"All items in the input list must be strings. Found item of type {type(text)}")
            else:
                encoded_texts = self._encode(texts)
                return encoded_texts


    async def aembed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise ValueError(f"Input must be a string. Found input of type {type(text)}")
        else:
            encoded_query = self._encode([text])[0]
            return encoded_query

async def main():
    embedding_model = EmbeddingModel()
    embedding_model.embed_documents(["This is a test document.", "What are you talking"])
    embedding_model.embed_query("What is the meaning of life?")

if __name__ == "__main__":
    asyncio.run(main())