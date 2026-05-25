"""
Capítulo 11 — Solução do exercício
Endpoint POST /conversar/{thread_id} que aceita mensagem e retorna
a resposta do agente imediatamente (síncrono).
O MemorySaver mantém o contexto da conversa na thread.
"""
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
import os

load_dotenv()

app = FastAPI(title="Agente de Conversa — Solução Cap 11", version="1.0.0")
API_KEY = os.getenv("API_KEY", "chave-dev-apenas")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def auth(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Não autorizado")
    return key


_agent = None


def _obter_agente():
    """Retorna o agente compartilhado com MemorySaver (thread-safe via thread_id)."""
    global _agent
    if _agent is not None:
        return _agent

    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langgraph.checkpoint.memory import MemorySaver

    llm = ChatAnthropic(model="claude-sonnet-4-6", max_retries=2)
    tools = [TavilySearchResults(max_results=2)]
    _agent = create_agent(llm, tools, checkpointer=MemorySaver())
    return _agent


@app.post("/conversar/{thread_id}")
async def conversar(thread_id: str, mensagem: str, _=Depends(auth)):
    """
    Endpoint síncrono de conversa com memória por thread.
    O MemorySaver mantém o contexto da conversa associado ao thread_id.
    """
    agent = _obter_agente()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        resultado = await asyncio.to_thread(
            agent.invoke,
            {"messages": [{"role": "user", "content": mensagem}]},
            config
        )
        resposta = resultado["messages"][-1].content
        return {
            "thread_id": thread_id,
            "resposta": resposta,
            "status": "ok"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no agente: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cap11.conversar_solucao:app", host="0.0.0.0", port=8001, reload=True)
