from assistant import Assistant
from langchain.messages import SystemMessage, HumanMessage
from prompts import DECOMPOSITION_PROMPT, REPHRASE_QUERY, RAG_ANSWER_PROMPT, SUMMARIZE_CHUNKS_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
import asyncio
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from vectorstore import VectorStore
from pydantic import BaseModel
from pprint import pprint

class ListOfStr(BaseModel):
    value: list[str]

class RAGGraphState(TypedDict):
    query: str
    rephrased_query: list[str]
    decomposed_queries: dict[str]
    retrieved_chunks: list[str]
    top_k: int
    versions: int
class RAGRetriever:

    def __init__(self):
        self.vector_store = VectorStore().get_connection()
        self.llm_list_output = Assistant(with_structured_output=True, response_format = ListOfStr)
        self.graph = self._build_graph()
    async def rephrase_query(self, state: RAGGraphState) -> RAGGraphState:
        messages = ChatPromptTemplate.from_messages([SystemMessage(content=REPHRASE_QUERY), HumanMessage(content=state["query"])]).format_messages(versions=state["versions"])
        response = await self.llm_list_output.run(messages)
        state["rephrased_query"] = response.value
        return state

    async def _decompose_query(self, query:str) -> list[str]:
        messages = ChatPromptTemplate.from_messages([SystemMessage(content=DECOMPOSITION_PROMPT), HumanMessage(content=query)]).format_messages()
        response = await self.llm_list_output.run(messages)
        return response.value

    async def decompose_queries(self, state: RAGGraphState) -> RAGGraphState:
        decompose_query = [self._decompose_query(query) for query in state["rephrased_query"]]
        try:
            decomposed_queries = await asyncio.gather(*decompose_query)
            state["decomposed_queries"] = {f"query_{i}": decomposed_query for i, decomposed_query in enumerate(decomposed_queries)}
            return state
        except Exception as e:
            raise e

    async def _retrieve(self, queries:list, top_k:int=5) -> str:
        llm_summarise = Assistant()
        similarity_tasks = [self.vector_store.asimilarity_search(query, top_k) for query in queries]
        retrieved_chunks = await asyncio.gather(*similarity_tasks)
        retrieved_content = [chunk.page_content for sublist in retrieved_chunks for chunk in sublist]
        messages = ChatPromptTemplate.from_messages([SystemMessage(content=RAG_ANSWER_PROMPT), HumanMessage(content=f"Here are the retrieved chunks {retrieved_content}")]).format_messages()
        output = await llm_summarise.run(messages)
        return output.content

    async def retrieve_documents(self, state: RAGGraphState) -> RAGGraphState:
        retrieve_tasks = [self._retrieve(queries, state["top_k"]) for queries in state["decomposed_queries"].values()]
        try:
            summaries = await asyncio.gather(*retrieve_tasks)
            state["retrieved_chunks"] = summaries
            return state
        except Exception as e:
            raise e

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(RAGGraphState)
        graph.add_node("rephrase_query", self.rephrase_query)
        graph.add_node("decompose_queries", self.decompose_queries)
        graph.add_node("retrieve_documents", self.retrieve_documents)
        graph.add_edge(START, "rephrase_query")
        graph.add_edge("rephrase_query", "decompose_queries")
        graph.add_edge("decompose_queries", "retrieve_documents")
        graph.add_edge("retrieve_documents", END)
        return graph.compile()

    async def run(self, query:str, top_k:int=5) -> list[dict]:
        response = await self.graph.ainvoke({"query": query, "top_k": top_k, "versions": 3})
        return response["retrieved_chunks"]

async def main():
    retriever = RAGRetriever()
    query = "Get me agentic patterns"
    retrieved_chunks = await retriever.run(query)
    print(retrieved_chunks)

if __name__ == "__main__":
    asyncio.run(main())