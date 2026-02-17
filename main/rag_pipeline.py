from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
from vectorstore import VectorStore
from tqdm import tqdm

class RAGPipeline:

    def __init__(self, document_folder: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.document_folder = document_folder
        self.chunk_size = int(os.getenv("CHUNK_SIZE", chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", chunk_overlap))
        self.vectorstore = VectorStore().get_connection()

    async def _split_document_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        docs = text_splitter.split_documents(documents)
        return docs

    async def _split_document_word(self, file_path: str):
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        docs = text_splitter.split_documents(documents)
        return docs

    async def load_and_feed_documents(self):
        for file_name in os.listdir(self.document_folder):
            file_path = str(Path(self.document_folder) / file_name)
            if file_name.endswith(".pdf"):
                docs = await self._split_document_pdf(str(file_path))
            elif file_name.endswith(".docx") or file_name.endswith(".doc"):
                docs = await self._split_document_word(str(file_path))
            else:
                continue
            self.vectorstore.add_documents(docs)
