import json
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.tools import tool

load_dotenv()


def build_tools():
    """Creates and returns the market agent's list of tools."""
    search = TavilySearch(max_results=3)

    @tool
    def calculate_percentage(value: float, percentage: float) -> str:
        """Calculates a percentage of a value. Use for discounts, taxes or commissions."""
        result = value * (percentage / 100)
        return f"{percentage}% of {value} = {result:.2f}"

    @tool
    def read_competitor_data(file_name: str) -> str:
        """Reads a competitor's data from a JSON file in data/.
        Use when the user asks for an analysis of a specific competitor."""
        path = Path(f"data/{file_name}.json")
        if not path.exists():
            available = [p.stem for p in Path("data").glob("*.json")]
            return f"File not found. Available: {available}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @tool
    def save_analysis(file_name: str, content: str) -> str:
        """Saves an analysis or report to a text file in results/.
        Use when the user asks to save, export or record a result."""
        path = Path(f"results/{file_name}")
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Analysis saved to results/{file_name} ({len(content)} characters)"

    return [search, calculate_percentage, read_competitor_data, save_analysis]


def build_agent():
    """Creates the market agent with search and analysis tools."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = build_tools()
    return create_agent(llm, tools)


def ask(agent, text: str) -> str:
    """Sends a question to the agent and returns the answer as text."""
    result = agent.invoke({
        "messages": [{"role": "user", "content": text}]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = build_agent()
    print(ask(agent, "What is today's dollar exchange rate?"))
