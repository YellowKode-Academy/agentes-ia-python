"""
Chapter 12: Final exercise solution
POST /monitor endpoint that:
1. Receives a list of topics
2. Schedules weekly analyses for each topic
3. Compares the current report with the previous one and highlights changes
4. Sends the diff by email via smtplib

Usage:
    uvicorn ch12.monitoring_solution:app --port 8002
    POST /monitor {"topics": ["fintech", "healthtech"], "email": "you@email.com"}
"""
import json
import asyncio
import smtplib
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Market Monitoring: Chapter 12 Solution", version="1.0.0")
API_KEY = os.getenv("API_KEY", "dev-key-only")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

_monitors: dict[str, dict] = {}
_history: dict[str, list[dict]] = {}


async def auth(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return key


class MonitorRequest(BaseModel):
    topics: list[str]
    email: str  # E.g.: "user@domain.com"
    interval_days: int = 7


def generate_diff(previous: dict, current: dict) -> str:
    """Generates comparison text between two reports."""
    diff_parts = []

    players_prev = set(previous.get("main_players", []))
    players_curr = set(current.get("main_players", []))
    new = players_curr - players_prev
    removed = players_prev - players_curr

    if new:
        diff_parts.append(f"NEW PLAYERS: {', '.join(list(new)[:5])}")
    if removed:
        diff_parts.append(f"PLAYERS LEFT: {', '.join(list(removed)[:5])}")

    size_prev = previous.get("market_size", "")
    size_curr = current.get("market_size", "")
    if size_prev and size_curr and size_prev != size_curr:
        diff_parts.append(f"MARKET SIZE: {size_prev} → {size_curr}")

    rec_prev = str(previous.get("recommendation", ""))[:100]
    rec_curr = str(current.get("recommendation", ""))[:100]
    if rec_prev != rec_curr:
        diff_parts.append(f"RECOMMENDATION CHANGED:\n  Before: {rec_prev}\n  Now:    {rec_curr}")

    return "\n".join(diff_parts) if diff_parts else "No significant changes detected."


def send_email_diff(recipient: str, topic: str, diff: str, current_report: dict) -> bool:
    """Sends an email with the report diff via SMTP."""
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        print(f"[EMAIL] SMTP not configured. Diff for '{topic}':\n{diff}")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Monitoring] Update: {topic}"
        msg["From"] = smtp_user
        msg["To"] = recipient

        body_text = (
            f"WEEKLY REPORT: {topic.upper()}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"CHANGES DETECTED:\n{diff}\n\n"
            f"CURRENT PLAYERS: {', '.join(current_report.get('main_players', [])[:5])}\n\n"
            f"CURRENT RECOMMENDATION:\n{current_report.get('recommendation', 'N/A')}"
        )
        body_html = f"""
        <html><body>
        <h2>Market Monitoring: {topic}</h2>
        <p><em>{datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>
        <h3>Detected Changes</h3>
        <pre>{diff}</pre>
        <h3>Current Players</h3>
        <p>{', '.join(current_report.get('main_players', [])[:5])}</p>
        <h3>Current Recommendation</h3>
        <p>{current_report.get('recommendation', 'N/A')}</p>
        </body></html>
        """
        msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
        print(f"[EMAIL] Sent to {recipient}: {topic}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False


async def run_monitoring_cycle(monitor_id: str):
    """Loop that runs weekly analyses and sends diffs by email."""
    from ch12.full_system import analyze_topic

    config = _monitors[monitor_id]
    topics = config["topics"]
    email = config["email"]
    interval = timedelta(days=config["interval_days"])

    while monitor_id in _monitors and _monitors[monitor_id]["active"]:
        for topic in topics:
            last = _monitors[monitor_id].get(f"last_{topic}")
            now = datetime.now()

            if last is None or now - datetime.fromisoformat(last) >= interval:
                print(f"[MONITOR {monitor_id}] Analyzing: {topic}")
                _monitors[monitor_id]["status"] = f"analyzing: {topic}"

                try:
                    current_report = await asyncio.to_thread(analyze_topic, topic)
                    history = _history.setdefault(f"{monitor_id}:{topic}", [])

                    if history:
                        diff = generate_diff(history[-1], current_report)
                        await asyncio.to_thread(send_email_diff, email, topic, diff, current_report)
                    else:
                        await asyncio.to_thread(
                            send_email_diff, email, topic,
                            "First analysis: no previous comparison.", current_report
                        )

                    history.append(current_report)
                    _monitors[monitor_id][f"last_{topic}"] = now.isoformat()

                except Exception as e:
                    print(f"[MONITOR {monitor_id}] Error on '{topic}': {e}")

        _monitors[monitor_id]["status"] = "waiting_next_cycle"
        await asyncio.sleep(3600)


@app.post("/monitor")
async def start_monitoring(req: MonitorRequest, _=Depends(auth)):
    """Starts weekly monitoring of topics with email notification."""
    monitor_id = f"monitor-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    _monitors[monitor_id] = {
        "id": monitor_id,
        "topics": req.topics,
        "email": req.email,
        "interval_days": req.interval_days,
        "created_at": datetime.now().isoformat(),
        "active": True,
        "status": "starting"
    }
    asyncio.create_task(run_monitoring_cycle(monitor_id))
    return {
        "monitor_id": monitor_id,
        "topics": req.topics,
        "email": req.email,
        "interval": f"every {req.interval_days} day(s)",
        "message": "Monitoring started. The first report will be generated shortly."
    }


@app.get("/monitor")
async def list_monitors(_=Depends(auth)):
    """Lists all active monitors."""
    return [
        {
            "id": m["id"],
            "topics": m["topics"],
            "email": m["email"],
            "active": m["active"],
            "status": m.get("status", "unknown"),
            "created_at": m["created_at"]
        }
        for m in _monitors.values()
    ]


@app.delete("/monitor/{monitor_id}")
async def stop_monitoring(monitor_id: str, _=Depends(auth)):
    """Stops an active monitor."""
    if monitor_id not in _monitors:
        raise HTTPException(404, "Monitor not found")
    _monitors[monitor_id]["active"] = False
    return {"message": f"Monitor {monitor_id} stopped"}


@app.get("/monitor/{monitor_id}/history")
async def monitor_history(monitor_id: str, _=Depends(auth)):
    """Returns the report history of a monitor."""
    if monitor_id not in _monitors:
        raise HTTPException(404, "Monitor not found")
    config = _monitors[monitor_id]
    histories = {}
    for topic in config["topics"]:
        key = f"{monitor_id}:{topic}"
        histories[topic] = len(_history.get(key, []))
    return {"monitor_id": monitor_id, "topics": histories}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_monitors": sum(1 for m in _monitors.values() if m["active"])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ch12.monitoring_solution:app", host="0.0.0.0", port=8002, reload=True)
