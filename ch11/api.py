import asyncio
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import json

load_dotenv()

app = FastAPI(
    title="Market Intelligence Agent",
    description="API for automated market analyses with AI agents",
    version="1.0.0"
)

jobs: dict[str, dict] = {}
API_KEY = os.getenv("API_KEY", "dev-key-only")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
_requests: dict = defaultdict(list)


async def auth(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return key


class AnalyzeRequest(BaseModel):
    topic: str
    depth: str = "medium"
    thread_id: Optional[str] = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"]
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    api_key = request.headers.get("X-API-Key", "anonymous")
    now = __import__("time").time()
    _requests[api_key] = [t for t in _requests[api_key] if now - t < 60]
    if len(_requests[api_key]) >= 10:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded: 10 analyses per minute."}
        )
    _requests[api_key].append(now)
    return await call_next(request)


async def run_analysis_bg(job_id: str, topic: str):
    """Runs the analysis in the background using the 3-agent Crew."""
    try:
        jobs[job_id]["status"] = "processing"
        from ch08.full_crew import build_market_intelligence_crew
        crew = build_market_intelligence_crew()
        result = await asyncio.wait_for(
            asyncio.to_thread(crew.kickoff, inputs={"topic": topic}),
            timeout=120
        )
        jobs[job_id].update({
            "status": "completed",
            "result": result.raw,
            "finished_at": datetime.now().isoformat()
        })
    except asyncio.TimeoutError:
        jobs[job_id].update({
            "status": "error",
            "error": "Timeout: analysis exceeded 2 minutes",
            "finished_at": datetime.now().isoformat()
        })
    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "error": str(e),
            "finished_at": datetime.now().isoformat()
        })


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks, _=Depends(auth)):
    """Starts a market analysis. Returns immediately with a job_id."""
    job_id = uuid.uuid4().hex[:8]
    thread_id = req.thread_id or f"thread-{job_id}"
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "topic": req.topic,
        "thread_id": thread_id,
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
        "finished_at": None
    }
    background_tasks.add_task(run_analysis_bg, job_id, req.topic)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Analysis started. Use GET /result/{job_id} to check."
    }


@app.get("/result/{job_id}")
async def get_result(job_id: str, _=Depends(auth)):
    """Returns the status and result of an analysis."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs")
async def list_jobs(_=Depends(auth)):
    """Lists all jobs with their status."""
    return [
        {"job_id": jid, "status": j["status"], "topic": j["topic"], "created_at": j["created_at"]}
        for jid, j in jobs.items()
    ]


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, _=Depends(auth)):
    """Removes a job from the system."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] == "processing":
        return JSONResponse(
            status_code=409,
            content={"detail": f"Job {job_id} is running. Wait for it to finish."}
        )
    del jobs[job_id]
    return {"message": f"Job {job_id} removed"}


@app.get("/metrics")
async def metrics(_=Depends(auth)):
    """Returns aggregate metrics for the jobs."""
    by_status: dict = defaultdict(int)
    for j in jobs.values():
        by_status[j["status"]] += 1
    return {"total_jobs": len(jobs), "by_status": dict(by_status)}


@app.post("/analyze/stream")
async def analyze_stream(req: AnalyzeRequest, _=Depends(auth)):
    """Streams the analysis via Server-Sent Events."""
    async def generate_chunks():
        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic
        from langchain_tavily import TavilySearch

        llm = ChatAnthropic(model="claude-sonnet-4-6")
        agent = create_agent(llm, [TavilySearch(max_results=3)])
        config = {"configurable": {"thread_id": req.thread_id or "stream"}}

        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": f"Analyze: {req.topic}"}]},
            config=config
        ):
            if "agent" in chunk:
                for msg in chunk["agent"]["messages"]:
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        yield f"data: {json.dumps({'text': msg.content}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_chunks(), media_type="text/event-stream")


@app.post("/chat/{thread_id}")
async def chat(thread_id: str, message: str, _=Depends(auth)):
    """Synchronous chat endpoint: response in 2-5 seconds."""
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_tavily import TavilySearch
    from langgraph.checkpoint.memory import MemorySaver

    llm = ChatAnthropic(model="claude-sonnet-4-6")
    agent = create_agent(llm, [TavilySearch(max_results=2)],
                         checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    result = await asyncio.to_thread(
        agent.invoke,
        {"messages": [{"role": "user", "content": message}]},
        config
    )
    return {
        "thread_id": thread_id,
        "response": result["messages"][-1].content,
        "status": "ok"
    }


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "active_jobs": len(jobs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ch11.api:app", host="0.0.0.0", port=8000, reload=True)
