"""
Capítulo 9 — Solução do exercício
Adiciona nó verificar_consistencia entre analisar e redigir.
Se a análise contiver afirmações não suportadas pelos dados coletados
(possível alucinação), o nó retorna ao pesquisador com query mais específica.
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
busca = TavilySearch(max_results=5)


class EstadoVerificado(TypedDict):
    tema: str
    dados: str
    qualidade: str
    analise: str
    consistencia: str   # "ok" ou "inconsistente"
    afirmacoes_sem_base: list[str]
    query_refinada: str
    relatorio: str
    tentativas: int
    tentativas_verificacao: int


def pesquisar(e: EstadoVerificado) -> EstadoVerificado:
    query = e.get("query_refinada") or f"{e['tema']} mercado brasil 2025 dados"
    resultados = busca.invoke(query)
    dados = "\n".join(r["content"] for r in resultados[:3]) if resultados else ""
    return {**e, "dados": dados, "tentativas": e.get("tentativas", 0) + 1, "query_refinada": ""}


def avaliar(e: EstadoVerificado) -> EstadoVerificado:
    if not e["dados"] or len(e["dados"]) < 200:
        return {**e, "qualidade": "insuficiente"}
    msgs = [
        SystemMessage(content="Responda APENAS 'suficiente' ou 'insuficiente'. Dados suficientes têm: tamanho de mercado, crescimento ou players."),
        HumanMessage(content=f"Dados sobre {e['tema']}:\n{e['dados'][:800]}")
    ]
    r = llm.invoke(msgs)
    return {**e, "qualidade": "suficiente" if "suficiente" in r.content.lower() else "insuficiente"}


def analisar(e: EstadoVerificado) -> EstadoVerificado:
    msgs = [
        SystemMessage(content="Analista de mercado. Produza análise com CAGR estimado, players, oportunidades e riscos. Baseie-se APENAS nos dados fornecidos."),
        HumanMessage(content=f"Tema: {e['tema']}\nDados:\n{e['dados'][:2000]}")
    ]
    r = llm.invoke(msgs)
    return {**e, "analise": r.content, "tentativas_verificacao": 0}


def verificar_consistencia(e: EstadoVerificado) -> EstadoVerificado:
    """Verifica se a análise é consistente com os dados coletados."""
    msgs = [
        SystemMessage(content=(
            "Você é um verificador de consistência. Compare a análise com os dados de pesquisa. "
            "Identifique afirmações na análise que NÃO têm base nos dados (números inventados, "
            "players não mencionados, percentuais sem fonte). "
            "Responda em JSON: {\"consistente\": true/false, \"afirmacoes_sem_base\": [\"lista\"], "
            "\"query_refinada\": \"string vazia se consistente, ou query específica para buscar dados que faltam\"}"
        )),
        HumanMessage(content=(
            f"DADOS DE PESQUISA:\n{e['dados'][:1500]}\n\n"
            f"ANÁLISE PRODUZIDA:\n{e['analise'][:1000]}"
        ))
    ]
    r = llm.invoke(msgs)
    import json, re
    try:
        m = re.search(r'\{.*\}', r.content, re.DOTALL)
        resultado = json.loads(m.group()) if m else {}
    except Exception:
        resultado = {}

    consistente = resultado.get("consistente", True)
    afirmacoes = resultado.get("afirmacoes_sem_base", [])
    query = resultado.get("query_refinada", "")
    return {
        **e,
        "consistencia": "ok" if consistente else "inconsistente",
        "afirmacoes_sem_base": afirmacoes,
        "query_refinada": query,
        "tentativas_verificacao": e.get("tentativas_verificacao", 0) + 1
    }


def redigir(e: EstadoVerificado) -> EstadoVerificado:
    nota_consistencia = ""
    if e.get("afirmacoes_sem_base"):
        nota_consistencia = (
            f"\n\nNOTA DO VERIFICADOR: As seguintes afirmações não foram confirmadas pelos dados: "
            f"{'; '.join(e['afirmacoes_sem_base'][:3])}"
        )
    msgs = [
        SystemMessage(content="Redator executivo. Escreva relatório para investidor. Estrutura: Sumário, Contexto, Análise Quantitativa, Oportunidades/Riscos, Recomendação."),
        HumanMessage(content=f"Tema: {e['tema']}\nDados:\n{e['dados'][:800]}\nAnálise:\n{e['analise']}")
    ]
    r = llm.invoke(msgs)
    return {**e, "relatorio": r.content + nota_consistencia}


def rotear_apos_avaliacao(e: EstadoVerificado) -> str:
    if e["qualidade"] == "suficiente" or e.get("tentativas", 0) >= 3:
        return "analisar"
    return "pesquisar"


def rotear_apos_verificacao(e: EstadoVerificado) -> str:
    if e["consistencia"] == "ok" or e.get("tentativas_verificacao", 0) >= 2:
        return "redigir"
    return "pesquisar"


def criar_app_com_verificador():
    """Grafo com verificação de consistência antes da redação."""
    g = StateGraph(EstadoVerificado)
    g.add_node("pesquisar", pesquisar)
    g.add_node("avaliar", avaliar)
    g.add_node("analisar", analisar)
    g.add_node("verificar_consistencia", verificar_consistencia)
    g.add_node("redigir", redigir)

    g.add_edge(START, "pesquisar")
    g.add_edge("pesquisar", "avaliar")
    g.add_conditional_edges("avaliar", rotear_apos_avaliacao,
                            {"pesquisar": "pesquisar", "analisar": "analisar"})
    g.add_edge("analisar", "verificar_consistencia")
    g.add_conditional_edges("verificar_consistencia", rotear_apos_verificacao,
                            {"redigir": "redigir", "pesquisar": "pesquisar"})
    g.add_edge("redigir", END)

    return g.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    app = criar_app_com_verificador()
    estado_inicial = {
        "tema": "agritechs de precision farming no Brasil",
        "dados": "", "qualidade": "insuficiente",
        "analise": "", "consistencia": "ok",
        "afirmacoes_sem_base": [], "query_refinada": "",
        "relatorio": "", "tentativas": 0, "tentativas_verificacao": 0
    }
    config = {"configurable": {"thread_id": "verificado-01"}}
    resultado = app.invoke(estado_inicial, config=config)
    print(f"Consistência: {resultado['consistencia']}")
    if resultado.get("afirmacoes_sem_base"):
        print(f"Afirmações revisadas: {resultado['afirmacoes_sem_base']}")
    print("\n=== RELATÓRIO ===")
    print(resultado["relatorio"])
