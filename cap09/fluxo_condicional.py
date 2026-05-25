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


class Estado(TypedDict):
    tema: str
    dados: str
    qualidade: str   # "suficiente" ou "insuficiente"
    analise: str
    relatorio: str
    tentativas: int


def pesquisar(e: Estado) -> Estado:
    """Busca dados sobre o tema na web."""
    query = f"{e['tema']} mercado brasil 2025 tamanho players crescimento dados"
    resultados = busca.invoke(query)
    if isinstance(resultados, str):
        dados = resultados
    elif resultados:
        dados = "\n".join(r.get("content", str(r)) for r in resultados[:3])
    else:
        dados = ""
    return {**e, "dados": dados, "tentativas": e.get("tentativas", 0) + 1}


def avaliar(e: Estado) -> Estado:
    """Avalia se os dados coletados são suficientes para uma análise."""
    if not e["dados"] or len(e["dados"]) < 200:
        return {**e, "qualidade": "insuficiente"}
    mensagens = [
        SystemMessage(content=(
            "Você avalia a qualidade de dados de pesquisa de mercado. "
            "Responda APENAS com 'suficiente' ou 'insuficiente'. "
            "Dados são suficientes se contêm pelo menos 2 dos seguintes: "
            "tamanho do mercado, taxa de crescimento, players principais."
        )),
        HumanMessage(content=f"Dados sobre {e['tema']}:\n{e['dados'][:800]}")
    ]
    resposta = llm.invoke(mensagens)
    qualidade = "suficiente" if "suficiente" in resposta.content.lower() else "insuficiente"
    return {**e, "qualidade": qualidade}


def analisar(e: Estado) -> Estado:
    """Produz análise quantitativa dos dados coletados."""
    mensagens = [
        SystemMessage(content=(
            "Você é um analista de mercado. Com base nos dados fornecidos, "
            "produza uma análise concisa com: CAGR estimado, principais players, "
            "oportunidades e riscos. Seja direto e baseado nos dados."
        )),
        HumanMessage(content=f"Tema: {e['tema']}\n\nDados:\n{e['dados'][:2000]}")
    ]
    resposta = llm.invoke(mensagens)
    return {**e, "analise": resposta.content}


def redigir(e: Estado) -> Estado:
    """Redige o relatório executivo final."""
    aviso = ""
    if e.get("qualidade") == "insuficiente":
        aviso = "\n\nNOTA: Os dados disponíveis foram limitados. Esta análise pode ser incompleta."
    mensagens = [
        SystemMessage(content=(
            "Você é um redator executivo. Com base na pesquisa e análise fornecidas, "
            "escreva um relatório executivo para um investidor. "
            "Estrutura: Sumário (2-3 linhas), Contexto de Mercado, "
            "Análise Quantitativa, Oportunidades e Riscos, Recomendação."
        )),
        HumanMessage(content=(
            f"Tema: {e['tema']}\n\n"
            f"Dados de Pesquisa:\n{e['dados'][:1000]}\n\n"
            f"Análise:\n{e['analise']}"
        ))
    ]
    resposta = llm.invoke(mensagens)
    return {**e, "relatorio": resposta.content + aviso}


def rotear(e: Estado) -> str:
    """Decide próximo nó com base na qualidade e número de tentativas."""
    if e["qualidade"] == "suficiente" or e.get("tentativas", 0) >= 3:
        return "analisar"
    return "pesquisar"


def criar_app():
    """Cria e compila o grafo de fluxo condicional."""
    g = StateGraph(Estado)
    g.add_node("pesquisar", pesquisar)
    g.add_node("avaliar", avaliar)
    g.add_node("analisar", analisar)
    g.add_node("redigir", redigir)
    g.add_edge(START, "pesquisar")
    g.add_edge("pesquisar", "avaliar")
    g.add_conditional_edges(
        "avaliar",
        rotear,
        {"pesquisar": "pesquisar", "analisar": "analisar"}
    )
    g.add_edge("analisar", "redigir")
    g.add_edge("redigir", END)
    return g.compile(checkpointer=MemorySaver())


def executar_analise(tema: str, thread_id: str | None = None) -> dict:
    """Executa o fluxo completo de análise para um tema."""
    app = criar_app()
    estado_inicial: Estado = {
        "tema": tema,
        "dados": "",
        "qualidade": "insuficiente",
        "analise": "",
        "relatorio": "",
        "tentativas": 0
    }
    config = {"configurable": {"thread_id": thread_id or f"analise-{tema[:15]}"}}
    return app.invoke(estado_inicial, config=config)


if __name__ == "__main__":
    resultado = executar_analise("logtech de última milha no Brasil")
    print(f"Tentativas de pesquisa: {resultado['tentativas']}")
    print(f"Qualidade dos dados: {resultado['qualidade']}")
    print("\n=== RELATÓRIO ===")
    print(resultado["relatorio"])
