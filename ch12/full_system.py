"""
Chapter 12: Full System: Market Intelligence Agent
Integrates: LangGraph (orchestration) + CrewAI (multi-agent) + ChromaDB (long-term memory)
            + Pydantic (validation) + FastAPI (REST API)
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")


class SystemState(TypedDict):
    topic: str
    previous_context: str
    raw_report: str
    validated_report: Optional[dict]
    validation_attempts: int
    status: str


def check_context(s: SystemState) -> SystemState:
    """Retrieves previous analyses from ChromaDB."""
    try:
        from ch04.memory_store import search_with_filter
        docs = search_with_filter(s["topic"], filters={}, k=3)
        context = "\n\n".join(d.page_content for d in docs) if docs else ""
        return {**s, "previous_context": context, "status": "researching"}
    except Exception:
        return {**s, "previous_context": "", "status": "researching"}


def run_crew(s: SystemState) -> SystemState:
    """Runs the 3-agent Crew to produce the report."""
    from ch08.full_crew import build_market_intelligence_crew
    crew = build_market_intelligence_crew()
    extra_prompt = ""
    if s.get("raw_report") and s.get("validation_attempts", 0) > 0:
        extra_prompt = (
            "\n\nIMPORTANT: The previous analysis was rejected by validation. "
            "Answer ONLY in valid JSON with the schema: "
            "{\"topic\": str, \"executive_summary\": str, \"main_players\": [str], "
            "\"market_size\": str, \"annual_growth\": str, "
            "\"opportunities\": [str], \"risks\": [str], \"recommendation\": str}"
        )
    result = crew.kickoff(inputs={"topic": s["topic"] + extra_prompt})
    return {**s, "raw_report": result.raw, "status": "validating"}


def validate_report_node(s: SystemState) -> SystemState:
    """Tries to extract structured JSON from the raw report."""
    text = s["raw_report"]

    # Try to extract JSON directly
    import re
    m = re.search(r'\{[^{}]*"topic"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if all(k in data for k in ["topic", "main_players", "opportunities", "risks", "recommendation"]):
                return {**s, "validated_report": data, "status": "validated",
                        "validation_attempts": s.get("validation_attempts", 0) + 1}
        except json.JSONDecodeError:
            pass

    # Extract structured fields from free text
    structured_report = {
        "topic": s["topic"],
        "executive_summary": text[:300] if text else "",
        "main_players": _extract_list(text, ["player", "company", "competitor"]),
        "market_size": _extract_value(text, ["market", "billion", "million", "R$"]),
        "annual_growth": _extract_value(text, ["growth", "CAGR", "%"]),
        "opportunities": _extract_list(text, ["opportunity", "potential", "growth"]),
        "risks": _extract_list(text, ["risk", "challenge", "threat"]),
        "recommendation": _extract_recommendation(text)
    }

    valid = (len(structured_report["main_players"]) >= 1 and
             len(structured_report["opportunities"]) >= 1 and
             len(structured_report["risks"]) >= 1)

    return {
        **s,
        "validated_report": structured_report if valid else None,
        "status": "validated" if valid else "invalid",
        "validation_attempts": s.get("validation_attempts", 0) + 1
    }


def _extract_list(text: str, keywords: list[str]) -> list[str]:
    """Extracts relevant sentences from the text based on keywords."""
    lines = text.split("\n")
    found = []
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            clean_line = line.strip().lstrip("-•*123456789. ")
            if 10 < len(clean_line) < 200:
                found.append(clean_line)
    return found[:5] if found else ["Information not available"]


def _extract_value(text: str, keywords: list[str]) -> str:
    """Extracts the first relevant numeric value from the text."""
    import re
    lines = text.split("\n")
    for line in lines:
        if any(kw.lower() in line.lower() for kw in keywords):
            m = re.search(r'R?\$?\s*[\d,\.]+\s*(billions?|millions?|%)?', line)
            if m:
                return m.group().strip()
    return "Data not available"


def _extract_recommendation(text: str) -> str:
    """Extracts the final recommendation from the text."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "recommend" in line.lower():
            parts = lines[i:i+3]
            return " ".join(p.strip() for p in parts if p.strip())[:300]
    return text[-300:].strip() if text else "Inconclusive analysis."


def save_result(s: SystemState) -> SystemState:
    """Saves the report to long-term memory (ChromaDB)."""
    try:
        from ch04.memory_store import ingest_text
        content = json.dumps(s.get("validated_report") or {"raw": s["raw_report"][:500]},
                             ensure_ascii=False)
        ingest_text(content, {
            "company": s["topic"],
            "type": "report",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": "full_system"
        })
    except Exception:
        pass
    return {**s, "status": "completed"}


def route_validation(s: SystemState) -> str:
    if s["status"] == "validated":
        return "save"
    elif s.get("validation_attempts", 0) >= 2:
        return "save"
    return "crew"


def build_full_system():
    """Creates and compiles the full system graph."""
    g = StateGraph(SystemState)
    g.add_node("check_context", check_context)
    g.add_node("crew", run_crew)
    g.add_node("validate", validate_report_node)
    g.add_node("save", save_result)
    g.add_edge(START, "check_context")
    g.add_edge("check_context", "crew")
    g.add_edge("crew", "validate")
    g.add_conditional_edges("validate", route_validation,
                            {"save": "save", "crew": "crew"})
    g.add_edge("save", END)
    return g.compile(checkpointer=MemorySaver())


def analyze_topic(topic: str) -> dict:
    """Runs the full analysis and returns the report."""
    app = build_full_system()
    initial_state: SystemState = {
        "topic": topic,
        "previous_context": "",
        "raw_report": "",
        "validated_report": None,
        "validation_attempts": 0,
        "status": "starting"
    }
    config = {"configurable": {"thread_id": f"analysis-{topic[:20].replace(' ', '-')}"}}
    result = app.invoke(initial_state, config=config)
    return result.get("validated_report") or {
        "topic": topic,
        "error": "Validation failed after attempts",
        "raw": result.get("raw_report", "")[:500]
    }


def compare_companies(topics: list[str]) -> dict:
    """Analyzes multiple topics and returns a comparison."""
    results = {}
    for topic in topics:
        print(f"Analyzing: {topic}...")
        results[topic] = analyze_topic(topic)

    comparison = {
        "topics_analyzed": topics,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "players_comparison": {},
        "opportunities_comparison": {},
        "summary": []
    }
    for topic, r in results.items():
        if r and "error" not in r:
            comparison["players_comparison"][topic] = r.get("main_players", [])[:3]
            comparison["opportunities_comparison"][topic] = r.get("opportunities", [])[:2]
            comparison["summary"].append({
                "topic": topic,
                "recommendation": r.get("recommendation", "N/A")[:200],
                "size": r.get("market_size", "N/A")
            })
    return comparison


class MarketMonitor:
    """Continuous monitoring of multiple topics with change detection."""

    def __init__(self, topics: list[str], interval_hours: int = 168):
        self.topics = topics
        self.interval = timedelta(hours=interval_hours)
        self.last_run: dict[str, datetime] = {}
        self.history: dict[str, list[dict]] = {t: [] for t in topics}

    def needs_update(self, topic: str) -> bool:
        last = self.last_run.get(topic)
        return last is None or datetime.now() - last > self.interval

    def run_monitoring(self, topic: str) -> dict:
        report = analyze_topic(topic)
        history = self.history.get(topic, [])
        changes = []

        if history:
            previous_players = set(history[-1].get("main_players", []))
            current_players = set(report.get("main_players", []))
            new = current_players - previous_players
            removed = previous_players - current_players
            if new:
                changes.append(f"New players: {', '.join(list(new)[:3])}")
                print(f"[{topic}] New players: {new}")
            if removed:
                changes.append(f"Players left: {', '.join(list(removed)[:3])}")

        report["_changes"] = changes
        report["_analysis_date"] = datetime.now().isoformat()
        self.history[topic].append(report)
        self.last_run[topic] = datetime.now()
        return report

    async def monitoring_loop(self):
        """Async loop that periodically checks and updates analyses."""
        while True:
            for topic in self.topics:
                if self.needs_update(topic):
                    print(f"Updating analysis: {topic}")
                    await asyncio.to_thread(self.run_monitoring, topic)
            await asyncio.sleep(3600)


if __name__ == "__main__":
    print("=== COMPETITOR ANALYSIS ===")
    r1 = analyze_topic("financial management platforms for SMBs")
    print(f"Topic: {r1.get('topic')}")
    print(f"Players: {r1.get('main_players', [])[:3]}")
    print(f"Recommendation: {str(r1.get('recommendation', ''))[:200]}")

    print("\n=== SECTOR COMPARISON ===")
    comp = compare_companies(["telemedicine healthtechs", "precision agritechs"])
    print(json.dumps(comp, ensure_ascii=False, indent=2))
