from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

load_dotenv()


def build_agent():
    """Creates and returns the project's base agent."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    return create_agent(llm, tools=[])


def ask(agent, text: str) -> str:
    """Sends a question to the agent and returns the answer as text."""
    result = agent.invoke({
        "messages": [{"role": "user", "content": text}]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = build_agent()
    print(ask(agent, "What is the square root of 144, squared?"))
