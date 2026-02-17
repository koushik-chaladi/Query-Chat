import asyncio
from rag_model import RAGModel
import os
from dotenv import load_dotenv
from langchain.messages import HumanMessage

load_dotenv()

async def main():
    rag_model = RAGModel()
    graph = await rag_model.get_graph()
    print("Welcome to the RAG Model Demo!, Type exit to quit the demo.")
    print("AI: I am a RAG model that can rephrase, decompose and retrieve documents based on your query. How can I assist you today?")
    while True:
        query = input("Human: ")
        if query.lower() == "exit":
            print("AI: Goodbye!")
            break
        state = {"messages": HumanMessage(content=query), "query": query, "top_k": 5, "versions": 1}
        response = await graph.ainvoke(state)
        print("AI:", response["answer"])

if __name__ == "__main__":
    asyncio.run(main())