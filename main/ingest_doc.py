from rag_pipeline import RAGPipeline
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

async def main():
    file_path = Path(__file__).resolve().parent.parent
    document_folder = str(file_path / "documents")
    pipeline = RAGPipeline(document_folder=document_folder)
    await pipeline.load_and_feed_documents()

if __name__ == "__main__":
    asyncio.run(main())