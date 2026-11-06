"""
Chapter 11: Exercise solution
POST /chat/{thread_id} endpoint that accepts a message and returns
the agent's answer immediately (synchronous).
MemorySaver keeps the conversation context in the thread.
"""
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
import os

load_dotenv()

app = FastAPI(title="Chat Agent: Chapter 11 Solution", version="1.0.0")
API_KEY = os.getenv("API_KEY", "dev-key-only")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def auth(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return key


_agent = None


def _get_agent():
    """Returns the shared agent with MemorySaver (thread-safe via thread_id)."""
    global _agent
    if _agent is not None:
        return _agent

    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_tavily import TavilySearch
    from langgraph.checkpoint.memory import MemorySaver

    llm = ChatAnthropic(model="claude-sonnet-4-6", max_retries=2)
    tools = [TavilySearch(max_results=2)]
    _agent = create_agent(llm, tools, checkpointer=MemorySaver())
    return _agent


@app.post("/chat/{thread_id}")
async def chat(thread_id: str, message: str, _=Depends(auth)):
    """
    Synchronous chat endpoint with per-thread memory.
    MemorySaver keeps the conversation context tied to the thread_id.
    """
    agent = _get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await asyncio.to_thread(
            agent.invoke,
            {"messages": [{"role": "user", "content": message}]},
            config
        )
        response = result["messages"][-1].content
        return {
            "thread_id": thread_id,
            "response": response,
            "status": "ok"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ch11.chat_solution:app", host="0.0.0.0", port=8001, reload=True)
