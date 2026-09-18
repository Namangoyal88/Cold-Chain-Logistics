import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


from src.agent_tools import FDE_TOOLS
from src.config import AGENT_LLM, OPENAI_API_KEY, OPENAI_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, OLLAMA_MODEL, SYSTEM_PROMPT_FILE


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_FILE}")

    return SYSTEM_PROMPT_FILE.read_text(encoding = "utf-8")


SYSTEM_PROMPT = load_system_prompt()

def create_llm():

    if AGENT_LLM == "OPENAI":

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is missing."
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY
        )

    if AGENT_LLM == "DEEPSEEK":

        if not DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY is missing."
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=DEEPSEEK_MODEL,
            temperature=0,
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0
    )


llm = create_llm()

llm_with_tools = llm.bind_tools(
    FDE_TOOLS
)


def reasoning_node(
    state: AgentState
):

    messages = state["messages"]

    response = llm_with_tools.invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT
            )
        ] + messages
    )

    return {
        "messages": [response]
    }


def build_graph():

    graph_builder = StateGraph(
        AgentState
    )

    # Add reasoning node
    graph_builder.add_node(
        "reasoner",
        reasoning_node
    )

    # Add tool node
    graph_builder.add_node(
        "tools",
        ToolNode(FDE_TOOLS)
    )

    # START → Reasoner
    graph_builder.add_edge(
        START,
        "reasoner"
    )

    # Reasoner → Tool OR END
    graph_builder.add_conditional_edges(
        "reasoner",
        tools_condition
    )

    # Tool → Reasoner
    graph_builder.add_edge(
        "tools",
        "reasoner"
    )

    # Memory
    memory = MemorySaver()

    graph = graph_builder.compile(
        checkpointer=memory
    )

    return graph


agent_graph = build_graph()


def ask_agent(
    question: str,
    thread_id: str = "default"
) -> str:

    result = agent_graph.invoke(
        {
            "messages": [
                (
                    "user",
                    question
                )
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    messages = result["messages"]

    return messages[-1].content


def stream_agent(
    question: str,
    thread_id: str = "default"
):

    events = agent_graph.stream(
        {
            "messages": [
                (
                    "user",
                    question
                )
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        },
        stream_mode="updates"
    )

    for event in events:

        yield event


if __name__ == "__main__":

    print("=" * 60)
    print("COLD-CHAIN AI AGENT")
    print("=" * 60)

    question = (
        "Find shipments with temperature above "
        "4 degrees Celsius and explain the risks."
    )

    response = ask_agent(
        question
    )

    print("\nFINAL RESPONSE:\n")
    print(response)