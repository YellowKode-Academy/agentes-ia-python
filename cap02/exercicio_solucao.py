"""
Capítulo 2 — Solução do exercício
Ferramenta comparar_concorrentes: lê dois JSONs e retorna comparação estruturada.
"""
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_classic.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic import hub

load_dotenv()


@tool
def comparar_concorrentes(nome_a: str, nome_b: str) -> str:
    """Compara dois concorrentes lendo seus arquivos JSON em dados/.
    Retorna análise estruturada de pontos fortes e fracos de cada um.
    nome_a e nome_b devem ser os nomes dos arquivos sem extensão .json."""
    def ler(nome: str):
        caminho = Path(f"dados/{nome}.json")
        if not caminho.exists():
            disponiveis = [p.stem for p in Path("dados").glob("*.json")]
            return None, f"Arquivo '{nome}.json' não encontrado. Disponíveis: {disponiveis}"
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f), None

    dados_a, erro_a = ler(nome_a)
    dados_b, erro_b = ler(nome_b)

    if erro_a:
        return erro_a
    if erro_b:
        return erro_b

    def extrair_metricas(dados: dict) -> dict:
        m = dados.get("metricas", {})
        return {
            "empresa": dados.get("empresa", nome_a),
            "funcionarios": dados.get("funcionarios", "N/D"),
            "arr_brl": m.get("arr_brl", "N/D"),
            "clientes": m.get("clientes_ativos", "N/D"),
            "crescimento": m.get("crescimento_q1_2025", "N/D"),
            "nps": m.get("nps", "N/D"),
            "oportunidades": dados.get("oportunidades", []),
            "riscos": dados.get("riscos", []),
        }

    ma = extrair_metricas(dados_a)
    mb = extrair_metricas(dados_b)

    return json.dumps({
        "comparacao": {
            "empresa_a": ma,
            "empresa_b": mb,
        },
        "vantagem_a_sobre_b": [
            f"ARR maior" if ma["arr_brl"] > mb["arr_brl"] else None,
            f"Mais clientes" if ma["clientes"] > mb["clientes"] else None,
            f"NPS superior" if ma["nps"] > mb["nps"] else None,
        ],
        "vantagem_b_sobre_a": [
            f"ARR maior" if mb["arr_brl"] > ma["arr_brl"] else None,
            f"Mais clientes" if mb["clientes"] > ma["clientes"] else None,
            f"NPS superior" if mb["nps"] > ma["nps"] else None,
        ],
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [TavilySearchResults(max_results=3), comparar_concorrentes]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    resultado = executor.invoke({
        "input": "Compare os concorrentes 'concorrente_a' e 'concorrente_b' e diga qual tem melhor posição de mercado."
    })
    print(resultado["output"])
