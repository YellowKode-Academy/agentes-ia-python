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
    title="Agente de Inteligência de Mercado",
    description="API para análises automatizadas de mercado com agentes de IA",
    version="1.0.0"
)

jobs: dict[str, dict] = {}
API_KEY = os.getenv("API_KEY", "chave-dev-apenas")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
_requisicoes: dict = defaultdict(list)


async def auth(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Não autorizado")
    return key


class AnalisarRequest(BaseModel):
    tema: str
    profundidade: str = "media"
    thread_id: Optional[str] = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"]
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    api_key = request.headers.get("X-API-Key", "anonimo")
    agora = __import__("time").time()
    _requisicoes[api_key] = [t for t in _requisicoes[api_key] if agora - t < 60]
    if len(_requisicoes[api_key]) >= 10:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit excedido: 10 análises por minuto."}
        )
    _requisicoes[api_key].append(agora)
    return await call_next(request)


async def executar_analise_bg(job_id: str, tema: str):
    """Executa análise em background usando a Crew de 3 agentes."""
    try:
        jobs[job_id]["status"] = "processando"
        from cap08.crew_completa import criar_crew_inteligencia_mercado
        crew = criar_crew_inteligencia_mercado()
        resultado = await asyncio.wait_for(
            asyncio.to_thread(crew.kickoff, inputs={"tema": tema}),
            timeout=120
        )
        jobs[job_id].update({
            "status": "concluido",
            "resultado": resultado.raw,
            "finalizado_em": datetime.now().isoformat()
        })
    except asyncio.TimeoutError:
        jobs[job_id].update({
            "status": "erro",
            "erro": "Timeout: análise excedeu 2 minutos",
            "finalizado_em": datetime.now().isoformat()
        })
    except Exception as e:
        jobs[job_id].update({
            "status": "erro",
            "erro": str(e),
            "finalizado_em": datetime.now().isoformat()
        })


@app.post("/analisar")
async def analisar(req: AnalisarRequest, background_tasks: BackgroundTasks, _=Depends(auth)):
    """Inicia análise de mercado. Retorna imediatamente com job_id."""
    job_id = uuid.uuid4().hex[:8]
    thread_id = req.thread_id or f"thread-{job_id}"
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pendente",
        "tema": req.tema,
        "thread_id": thread_id,
        "criado_em": datetime.now().isoformat(),
        "resultado": None,
        "erro": None,
        "finalizado_em": None
    }
    background_tasks.add_task(executar_analise_bg, job_id, req.tema)
    return {
        "job_id": job_id,
        "status": "pendente",
        "mensagem": f"Análise iniciada. Use GET /resultado/{job_id} para verificar."
    }


@app.get("/resultado/{job_id}")
async def resultado(job_id: str, _=Depends(auth)):
    """Retorna status e resultado de uma análise."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return jobs[job_id]


@app.get("/jobs")
async def listar_jobs(_=Depends(auth)):
    """Lista todos os jobs com seu status."""
    return [
        {"job_id": jid, "status": j["status"], "tema": j["tema"], "criado_em": j["criado_em"]}
        for jid, j in jobs.items()
    ]


@app.delete("/jobs/{job_id}")
async def cancelar_job(job_id: str, _=Depends(auth)):
    """Remove um job do sistema."""
    if job_id not in jobs:
        raise HTTPException(404, "Job não encontrado")
    if jobs[job_id]["status"] == "processando":
        return JSONResponse(
            status_code=409,
            content={"detail": f"Job {job_id} em execução. Aguarde a conclusão."}
        )
    del jobs[job_id]
    return {"mensagem": f"Job {job_id} removido"}


@app.get("/metricas")
async def metricas(_=Depends(auth)):
    """Retorna métricas agregadas dos jobs."""
    por_status: dict = defaultdict(int)
    for j in jobs.values():
        por_status[j["status"]] += 1
    return {"total_jobs": len(jobs), "por_status": dict(por_status)}


@app.post("/analisar/stream")
async def analisar_stream(req: AnalisarRequest, _=Depends(auth)):
    """Streaming de análise via Server-Sent Events."""
    async def gerar_chunks():
        from langchain_anthropic import ChatAnthropic
        from langchain_community.tools.tavily_search import TavilySearchResults
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub

        llm = ChatAnthropic(model="claude-sonnet-4-6")
        tools = [TavilySearchResults(max_results=3)]
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

        async for chunk in executor.astream({"input": f"Analise o mercado de: {req.tema}"}):
            if "output" in chunk:
                yield f"data: {json.dumps({'texto': chunk['output']}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gerar_chunks(), media_type="text/event-stream")


@app.post("/conversar/{thread_id}")
async def conversar(thread_id: str, mensagem: str, _=Depends(auth)):
    """Endpoint síncrono para conversa — resposta em 2-5 segundos."""
    from langchain_anthropic import ChatAnthropic
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub
    from langgraph.checkpoint.memory import MemorySaver

    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [TavilySearchResults(max_results=2)]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools,
        handle_parsing_errors=True, max_iterations=4
    )
    resultado = await asyncio.to_thread(executor.invoke, {"input": mensagem})
    return {
        "thread_id": thread_id,
        "resposta": resultado.get("output", ""),
        "status": "ok"
    }


@app.get("/health")
async def health():
    return {"status": "ok", "versao": "1.0.0", "jobs_ativos": len(jobs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cap11.api:app", host="0.0.0.0", port=8000, reload=True)
