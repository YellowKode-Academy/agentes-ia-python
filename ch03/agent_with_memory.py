from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# External history by thread_id: {"thread_id": [{"human": ..., "assistant": ...}]}
_history: dict[str, list[dict]] = {}


def build_agent():
    """Creates an agent with short-term memory and tools."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [TavilySearch(max_results=3)]
    memory = MemorySaver()
    return create_agent(llm, tools, checkpointer=memory)


def chat(agent, message: str, thread_id: str = "default") -> str:
    """Sends a message to the agent while keeping the conversation context."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    response = result["messages"][-1].content

    # Record in the external history for inspection/reporting
    if thread_id not in _history:
        _history[thread_id] = []
    _history[thread_id].append({"human": message, "assistant": response})

    return response


if __name__ == "__main__":
    agent = build_agent()

    print(chat(agent, "My name is Kelvin.", thread_id="test"))
    print(chat(agent, "What is my name?", thread_id="test"))
    print(chat(agent, "I'm analyzing the fintech sector.", thread_id="analysis-01"))
    print(chat(agent, "Who are the main players in this sector?", thread_id="analysis-01"))
