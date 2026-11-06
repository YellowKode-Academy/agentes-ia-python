"""
Chapter 2: Exercise solution
The compare_competitors tool: reads two JSON files and returns a structured comparison.
"""
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def compare_competitors(name_a: str, name_b: str) -> str:
    """Compares two competitors by reading their JSON files in data/.
    Returns a structured analysis of each one's strengths and weaknesses.
    name_a and name_b must be the file names without the .json extension."""
    def read(name: str):
        path = Path(f"data/{name}.json")
        if not path.exists():
            available = [p.stem for p in Path("data").glob("*.json")]
            return None, f"File '{name}.json' not found. Available: {available}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None

    data_a, error_a = read(name_a)
    data_b, error_b = read(name_b)

    if error_a:
        return error_a
    if error_b:
        return error_b

    def extract_metrics(data: dict) -> dict:
        m = data.get("metrics", {})
        return {
            "company": data.get("company", name_a),
            "employees": data.get("employees", "N/A"),
            "arr_brl": m.get("arr_brl", "N/A"),
            "customers": m.get("active_customers", "N/A"),
            "growth": m.get("growth_q1_2025", "N/A"),
            "nps": m.get("nps", "N/A"),
            "opportunities": data.get("opportunities", []),
            "risks": data.get("risks", []),
        }

    ma = extract_metrics(data_a)
    mb = extract_metrics(data_b)

    return json.dumps({
        "comparison": {
            "company_a": ma,
            "company_b": mb,
        },
        "advantage_a_over_b": [
            f"Higher ARR" if ma["arr_brl"] > mb["arr_brl"] else None,
            f"More customers" if ma["customers"] > mb["customers"] else None,
            f"Higher NPS" if ma["nps"] > mb["nps"] else None,
        ],
        "advantage_b_over_a": [
            f"Higher ARR" if mb["arr_brl"] > ma["arr_brl"] else None,
            f"More customers" if mb["customers"] > ma["customers"] else None,
            f"Higher NPS" if mb["nps"] > ma["nps"] else None,
        ],
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [TavilySearch(max_results=3), compare_competitors]
    agent = create_agent(llm, tools)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Compare the competitors 'competitor_a' and 'competitor_b' and say which one has the better market position."}]
    })
    print(result["messages"][-1].content)
