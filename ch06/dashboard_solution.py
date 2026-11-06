"""
Chapter 6: Exercise solution
Dashboard that reads logs/agent.log and shows usage statistics:
- Total number of runs
- Average latency per run
- Most used tool
- Estimated total cost for the day
"""
import re
from pathlib import Path
from datetime import datetime, date
from collections import Counter


LOG_FILE = Path("logs/agent.log")


def parse_log_line(line: str) -> dict | None:
    """Extracts structured fields from a structlog log line."""
    if not line.strip():
        return None
    try:
        fields = {}
        parts = line.strip().split(" ")
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k] = v.strip("'\"")
        if "timestamp" not in fields and len(parts) > 1:
            fields["_raw"] = line
        return fields if fields else None
    except Exception:
        return None


def read_log_events(path: Path, date_filter: date | None = None) -> list[dict]:
    """Reads the log file and returns a list of structured events."""
    if not path.exists():
        return []
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            event = parse_log_line(line)
            if event:
                if date_filter:
                    ts = event.get("timestamp", "")
                    if ts and not ts.startswith(str(date_filter)):
                        continue
                events.append({"_raw": line.strip(), **event})
    return events


def calculate_metrics(events: list[dict]) -> dict:
    """Calculates aggregate metrics from the log events."""
    runs_start = [e for e in events if "run_start" in e.get("_raw", "")]
    runs_end = [e for e in events if "run_end" in e.get("_raw", "")]
    runs_error = [e for e in events if "run_error" in e.get("_raw", "")]

    latencies = []
    costs = []
    tools: list[str] = []

    for event in runs_end:
        raw = event.get("_raw", "")
        m_lat = re.search(r"latency_s=([\d.]+)", raw)
        if m_lat:
            latencies.append(float(m_lat.group(1)))
        m_cost = re.search(r"cost_usd=([\d.]+)", raw)
        if m_cost:
            costs.append(float(m_cost.group(1)))
        m_tools = re.search(r"tools_used=\[([^\]]*)\]", raw)
        if m_tools:
            tools_str = m_tools.group(1)
            tools_list = [t.strip().strip("'\"") for t in tools_str.split(",") if t.strip()]
            tools.extend(tools_list)

    for event in events:
        raw = event.get("_raw", "")
        if "tool " in raw or "tool=" in raw:
            m_tool = re.search(r"name='?([^'\s]+)'?", raw)
            if m_tool:
                tools.append(m_tool.group(1))

    most_used_tool = Counter(tools).most_common(1)

    return {
        "total_runs": len(runs_start),
        "completed_runs": len(runs_end),
        "failed_runs": len(runs_error),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "max_latency_s": max(latencies) if latencies else 0,
        "total_cost_usd": round(sum(costs), 4),
        "most_used_tool": most_used_tool[0][0] if most_used_tool else "N/A",
        "total_tool_calls": len(tools),
    }


def show_dashboard(path: Path = LOG_FILE, today_only: bool = True) -> None:
    """Shows the metrics dashboard in the terminal."""
    date_filter = date.today() if today_only else None
    period = f"today ({date.today()})" if today_only else "all history"

    print(f"\n{'='*50}")
    print(f"AGENT DASHBOARD: {period.upper()}")
    print(f"File: {path}")
    print(f"{'='*50}")

    if not path.exists():
        print(f"\nLog file not found: {path}")
        print("Run the agent with ObservableAgent to generate logs.")
        return

    events = read_log_events(path, date_filter)
    if not events:
        print(f"\nNo events recorded for {period}.")
        return

    m = calculate_metrics(events)

    print(f"\n📊 Runs: {m['total_runs']} started, {m['completed_runs']} completed, {m['failed_runs']} failed")
    print(f"⏱  Average latency: {m['avg_latency_s']}s | Max: {m['max_latency_s']}s")
    print(f"🔧 Most used tool: {m['most_used_tool']} ({m['total_tool_calls']} calls total)")
    print(f"💰 Estimated cost {period}: ${m['total_cost_usd']:.4f} USD")

    if m['total_cost_usd'] > 5.0:
        print(f"\n⚠️  ALERT: high accumulated cost (${m['total_cost_usd']:.4f})")

    print()


if __name__ == "__main__":
    show_dashboard()
