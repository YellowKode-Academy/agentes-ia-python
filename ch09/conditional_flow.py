from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")
search = TavilySearch(max_results=5)


class State(TypedDict):
    topic: str
    data: str
    quality: str   # "sufficient" or "insufficient"
    analysis: str
    report: str
    attempts: int


def research(s: State) -> State:
    """Searches the web for data on the topic."""
    query = f"{s['topic']} Brazil market 2025 size players growth data"
    results = search.invoke(query)
    if isinstance(results, str):
        data = results
    elif isinstance(results, dict):
        items = results.get("results", [])
        data = "\n".join(r.get("content", str(r)) for r in items[:3])
    elif isinstance(results, list):
        data = "\n".join(r.get("content", str(r)) for r in results[:3])
    else:
        data = ""
    return {**s, "data": data, "attempts": s.get("attempts", 0) + 1}


def evaluate(s: State) -> State:
    """Evaluates whether the collected data is enough for an analysis."""
    if not s["data"] or len(s["data"]) < 200:
        return {**s, "quality": "insufficient"}
    messages = [
        SystemMessage(content=(
            "You evaluate the quality of market research data. "
            "Answer ONLY with 'sufficient' or 'insufficient'. "
            "Data is sufficient if it contains at least 2 of the following: "
            "market size, growth rate, main players."
        )),
        HumanMessage(content=f"Data on {s['topic']}:\n{s['data'][:800]}")
    ]
    response = llm.invoke(messages)
    quality = "sufficient" if "sufficient" in response.content.lower() else "insufficient"
    return {**s, "quality": quality}


def analyze(s: State) -> State:
    """Produces a quantitative analysis of the collected data."""
    messages = [
        SystemMessage(content=(
            "You are a market analyst. Based on the data provided, "
            "produce a concise analysis with: estimated CAGR, main players, "
            "opportunities and risks. Be direct and data-based."
        )),
        HumanMessage(content=f"Topic: {s['topic']}\n\nData:\n{s['data'][:2000]}")
    ]
    response = llm.invoke(messages)
    return {**s, "analysis": response.content}


def write_report(s: State) -> State:
    """Writes the final executive report."""
    warning = ""
    if s.get("quality") == "insufficient":
        warning = "\n\nNOTE: The available data was limited. This analysis may be incomplete."
    messages = [
        SystemMessage(content=(
            "You are an executive writer. Based on the research and analysis provided, "
            "write an executive report for an investor. "
            "Structure: Summary (2-3 lines), Market Context, "
            "Quantitative Analysis, Opportunities and Risks, Recommendation."
        )),
        HumanMessage(content=(
            f"Topic: {s['topic']}\n\n"
            f"Research Data:\n{s['data'][:1000]}\n\n"
            f"Analysis:\n{s['analysis']}"
        ))
    ]
    response = llm.invoke(messages)
    return {**s, "report": response.content + warning}


def route(s: State) -> str:
    """Decides the next node based on quality and number of attempts."""
    if s["quality"] == "sufficient" or s.get("attempts", 0) >= 3:
        return "analyze"
    return "research"


def build_app():
    """Creates and compiles the conditional flow graph."""
    g = StateGraph(State)
    g.add_node("research", research)
    g.add_node("evaluate", evaluate)
    g.add_node("analyze", analyze)
    g.add_node("write_report", write_report)
    g.add_edge(START, "research")
    g.add_edge("research", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        route,
        {"research": "research", "analyze": "analyze"}
    )
    g.add_edge("analyze", "write_report")
    g.add_edge("write_report", END)
    return g.compile(checkpointer=MemorySaver())


def run_analysis(topic: str, thread_id: str | None = None) -> dict:
    """Runs the full analysis flow for a topic."""
    app = build_app()
    initial_state: State = {
        "topic": topic,
        "data": "",
        "quality": "insufficient",
        "analysis": "",
        "report": "",
        "attempts": 0
    }
    config = {"configurable": {"thread_id": thread_id or f"analysis-{topic[:15]}"}}
    return app.invoke(initial_state, config=config)


if __name__ == "__main__":
    result = run_analysis("last-mile logtech in Brazil")
    print(f"Research attempts: {result['attempts']}")
    print(f"Data quality: {result['quality']}")
    print("\n=== REPORT ===")
    print(result["report"])
