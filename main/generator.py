from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
from prompts import RAG_ANSWER_PROMPT
from assistant import Assistant

class GenGraphState(TypedDict):
    query: str
    retrieved_chunks: list[str]
    answer: str

class RAGGenerator:

    def __init__(self):
        self.graph = self._build_graph()
        self.llm = Assistant()

    async def generate_answer(self, state: GenGraphState) -> GenGraphState:
        messages = ChatPromptTemplate.from_messages([SystemMessage(content=RAG_ANSWER_PROMPT), HumanMessage(content=f"query: {state['query']}, chunks: {state['retrieved_chunks']}")]).format_messages()
        response = await self.llm.run(messages)
        state["answer"] = response.content
        return state
    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(GenGraphState)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_edge(START, "generate_answer")
        graph.add_edge("generate_answer", END)
        compiled_graph = graph.compile()
        return compiled_graph

    async def run(self, query:str, retrieved_chunks: list[dict]) -> str:
        state = {"query": query, "retrieved_chunks": retrieved_chunks, "answer": ""}
        result = await self.graph.ainvoke(state)
        return result

async def main():
    generator = RAGGenerator()
    from test_gen import get_chunks
    query ="What are the agentic patterns"
    answer = await generator.run(query, get_chunks())
    print(answer["answer"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())