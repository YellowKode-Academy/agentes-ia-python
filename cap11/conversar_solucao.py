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

_agentes: dict = {}


async def auth(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Não autorizado")
    return key


def _obter_ou_criar_agente(thread_id: str):
    """Retorna agente existente para a thread ou cria um novo."""
    if thread_id in _agentes:
        return _agentes[thread_id]

    from langchain_anthropic import ChatAnthropic
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_classic.agents import create_react_agent, AgentExecutor
    from langchain_classic import hub

    llm = ChatAnthropic(model="claude-sonnet-4-6", max_retries=2)
    tools = [TavilySearchResults(max_results=2)]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=5,
        verbose=False
    )
    _agentes[thread_id] = {"executor": executor, "historico": []}
    return _agentes[thread_id]


@app.post("/conversar/{thread_id}")
async def conversar(thread_id: str, mensagem: str, _=Depends(auth)):
    """
    Endpoint síncrono de conversa com memória por thread.
    Mantém histórico das últimas 5 trocas como contexto no prompt.
    """
    estado = _obter_ou_criar_agente(thread_id)
    executor = estado["executor"]
    historico = estado["historico"]

    contexto = ""
    if historico:
        contexto = "Histórico da conversa:\n"
        for troca in historico[-5:]:
            contexto += f"Usuário: {troca['usuario']}\nAssistente: {troca['assistente']}\n\n"
        contexto += "Nova mensagem: "

    input_completo = contexto + mensagem

    try:
        resultado = await asyncio.to_thread(
            executor.invoke, {"input": input_completo}
        )
        resposta = resultado.get("output", "")
        historico.append({"usuario": mensagem, "assistente": resposta})
        return {
            "thread_id": thread_id,
            "resposta": resposta,
            "total_mensagens": len(historico),
            "status": "ok"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no agente: {str(e)}")


@app.delete("/conversar/{thread_id}")
async def limpar_historico(thread_id: str, _=Depends(auth)):
    """Limpa o histórico de uma thread."""
    if thread_id in _agentes:
        del _agentes[thread_id]
    return {"mensagem": f"Histórico da thread '{thread_id}' removido"}


@app.get("/conversar/{thread_id}/historico")
async def ver_historico(thread_id: str, _=Depends(auth)):
    """Retorna o histórico de mensagens de uma thread."""
    if thread_id not in _agentes:
        return {"thread_id": thread_id, "historico": [], "total": 0}
    historico = _agentes[thread_id]["historico"]
    return {
        "thread_id": thread_id,
        "historico": [
            {"i": i + 1, "usuario": t["usuario"][:100], "assistente": t["assistente"][:200]}
            for i, t in enumerate(historico)
        ],
        "total": len(historico)
    }


@app.get("/health")
async def health():
    return {"status": "ok", "threads_ativas": len(_agentes)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cap11.conversar_solucao:app", host="0.0.0.0", port=8001, reload=True)
