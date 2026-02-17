from retriever import RAGRetriever
from generator import RAGGenerator
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from assistant import Assistant
from langchain_core.prompts import ChatPromptTemplate
from prompts import STANDALONE_QUERY

class RAGModelState(TypedDict):
    query: str
    stand_alone_query: str
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_chunks: list[str]
    answer: str

class RAGModel:

    def __init__(self):
        self.graph = self._build_graph()

    async def rewrite(self, state: RAGModelState) -> RAGModelState:
        llm = Assistant()
        messages = ChatPromptTemplate.from_messages([SystemMessage(content=STANDALONE_QUERY), HumanMessage(content=f"query: {state['query']}, history: {state['messages'][-10:]}")]).format_messages()
        response = await llm.run(messages)
        state["stand_alone_query"] = response.content
        return state
    async def retriever(self, state: RAGModelState) -> RAGModelState:
        retriever = RAGRetriever()
        retrieved_chunks = await retriever.run(state["query"])
        state["retrieved_chunks"] = retrieved_chunks
        return state

    async def generator(self, state: RAGModelState) -> RAGModelState:
        generator = RAGGenerator()
        response = await generator.run(state["query"], state["retrieved_chunks"])
        state["answer"] = response["answer"]
        state["messages"].append(AIMessage(content=response["answer"]))
        return state

    def _build_graph(self) -> CompiledStateGraph :
        graph = StateGraph(RAGModelState)
        graph.add_node("rewrite", self.rewrite)
        graph.add_node("retriever", self.retriever)
        graph.add_node("generator", self.generator)
        #graph.add_edge(START, "rewrite")
        graph.add_edge(START, "retriever")
        graph.add_edge("retriever", "generator")
        graph.add_edge("generator", END)
        return graph.compile()

    async def get_graph(self) -> CompiledStateGraph:
        return self.graph

async def main():
    rag_model = RAGModel()
    graph = await rag_model.get_graph()
    initial_state = {"messages": [HumanMessage(content="What are different types of agentic architectures?")]}
    final_state = await graph.ainvoke(initial_state)
    print(final_state["answer"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())