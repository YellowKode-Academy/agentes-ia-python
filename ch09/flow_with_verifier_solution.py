"""
Chapter 9: Exercise solution
Adds a verify_consistency node between analyze and write_report.
If the analysis contains claims not supported by the collected data
(possible hallucination), the node goes back to the researcher with a more specific query.
"""
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


class VerifiedState(TypedDict):
    topic: str
    data: str
    quality: str
    analysis: str
    consistency: str   # "ok" or "inconsistent"
    unsupported_claims: list[str]
    refined_query: str
    report: str
    attempts: int
    verification_attempts: int


def research(s: VerifiedState) -> VerifiedState:
    query = s.get("refined_query") or f"{s['topic']} Brazil market 2025 data"
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
    return {**s, "data": data, "attempts": s.get("attempts", 0) + 1, "refined_query": ""}


def evaluate(s: VerifiedState) -> VerifiedState:
    if not s["data"] or len(s["data"]) < 200:
        return {**s, "quality": "insufficient"}
    msgs = [
        SystemMessage(content="Answer ONLY 'sufficient' or 'insufficient'. Sufficient data has: market size, growth or players."),
        HumanMessage(content=f"Data on {s['topic']}:\n{s['data'][:800]}")
    ]
    r = llm.invoke(msgs)
    return {**s, "quality": "sufficient" if "sufficient" in r.content.lower() else "insufficient"}


def analyze(s: VerifiedState) -> VerifiedState:
    msgs = [
        SystemMessage(content="Market analyst. Produce an analysis with estimated CAGR, players, opportunities and risks. Base it ONLY on the data provided."),
        HumanMessage(content=f"Topic: {s['topic']}\nData:\n{s['data'][:2000]}")
    ]
    r = llm.invoke(msgs)
    return {**s, "analysis": r.content, "verification_attempts": 0}


def verify_consistency(s: VerifiedState) -> VerifiedState:
    """Checks whether the analysis is consistent with the collected data."""
    msgs = [
        SystemMessage(content=(
            "You are a consistency verifier. Compare the analysis with the research data. "
            "Identify claims in the analysis that have NO basis in the data (made-up numbers, "
            "players not mentioned, percentages without a source). "
            "Answer in JSON: {\"consistent\": true/false, \"unsupported_claims\": [\"list\"], "
            "\"refined_query\": \"empty string if consistent, or a specific query to find the missing data\"}"
        )),
        HumanMessage(content=(
            f"RESEARCH DATA:\n{s['data'][:1500]}\n\n"
            f"ANALYSIS PRODUCED:\n{s['analysis'][:1000]}"
        ))
    ]
    r = llm.invoke(msgs)
    import json, re
    try:
        m = re.search(r'\{.*\}', r.content, re.DOTALL)
        result = json.loads(m.group()) if m else {}
    except Exception:
        result = {}

    consistent = result.get("consistent", True)
    claims = result.get("unsupported_claims", [])
    query = result.get("refined_query", "")
    return {
        **s,
        "consistency": "ok" if consistent else "inconsistent",
        "unsupported_claims": claims,
        "refined_query": query,
        "verification_attempts": s.get("verification_attempts", 0) + 1
    }


def write_report(s: VerifiedState) -> VerifiedState:
    consistency_note = ""
    if s.get("unsupported_claims"):
        consistency_note = (
            f"\n\nVERIFIER NOTE: The following claims were not confirmed by the data: "
            f"{'; '.join(s['unsupported_claims'][:3])}"
        )
    msgs = [
        SystemMessage(content="Executive writer. Write a report for an investor. Structure: Summary, Context, Quantitative Analysis, Opportunities/Risks, Recommendation."),
        HumanMessage(content=f"Topic: {s['topic']}\nData:\n{s['data'][:800]}\nAnalysis:\n{s['analysis']}")
    ]
    r = llm.invoke(msgs)
    return {**s, "report": r.content + consistency_note}


def route_after_evaluation(s: VerifiedState) -> str:
    if s["quality"] == "sufficient" or s.get("attempts", 0) >= 3:
        return "analyze"
    return "research"


def route_after_verification(s: VerifiedState) -> str:
    if s["consistency"] == "ok" or s.get("verification_attempts", 0) >= 2:
        return "write_report"
    return "research"


def build_app_with_verifier():
    """Graph with a consistency check before writing the report."""
    g = StateGraph(VerifiedState)
    g.add_node("research", research)
    g.add_node("evaluate", evaluate)
    g.add_node("analyze", analyze)
    g.add_node("verify_consistency", verify_consistency)
    g.add_node("write_report", write_report)

    g.add_edge(START, "research")
    g.add_edge("research", "evaluate")
    g.add_conditional_edges("evaluate", route_after_evaluation,
                            {"research": "research", "analyze": "analyze"})
    g.add_edge("analyze", "verify_consistency")
    g.add_conditional_edges("verify_consistency", route_after_verification,
                            {"write_report": "write_report", "research": "research"})
    g.add_edge("write_report", END)

    return g.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    app = build_app_with_verifier()
    initial_state = {
        "topic": "precision farming agritechs in Brazil",
        "data": "", "quality": "insufficient",
        "analysis": "", "consistency": "ok",
        "unsupported_claims": [], "refined_query": "",
        "report": "", "attempts": 0, "verification_attempts": 0
    }
    config = {"configurable": {"thread_id": "verified-01"}}
    result = app.invoke(initial_state, config=config)
    print(f"Consistency: {result['consistency']}")
    if result.get("unsupported_claims"):
        print(f"Reviewed claims: {result['unsupported_claims']}")
    print("\n=== REPORT ===")
    print(result["report"])
