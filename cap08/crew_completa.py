from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.tools import tool
import json

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")


def criar_crew_inteligencia_mercado() -> Crew:
    """Cria a Crew completa de inteligência de mercado com 3 agentes especializados."""
    busca = TavilySearch(max_results=5)

    @tool
    def calcular_cagr(valor_inicial: float, valor_final: float, anos: int) -> str:
        """Calcula CAGR (taxa de crescimento anual composta) entre dois valores.
        Exemplo: calcular_cagr(100, 200, 5) → CAGR em 5 anos: 14.87%"""
        if valor_inicial <= 0 or anos <= 0:
            return "Erro: valores devem ser positivos."
        cagr = ((valor_final / valor_inicial) ** (1 / anos) - 1) * 100
        return f"CAGR em {anos} anos: {cagr:.2f}%"

    @tool
    def calcular_concentracao(market_shares_json: str) -> str:
        """Calcula índice de concentração de mercado a partir de dicionário JSON com market shares.
        Entrada: JSON com {'empresa': percentual}. Ex: {"A": 30, "B": 20, "C": 15}"""
        try:
            shares = json.loads(market_shares_json)
            top3 = sorted(shares.values(), reverse=True)[:3]
            cr3 = sum(top3)
            hhi = sum(v**2 for v in shares.values())
            return (
                f"CR3 (concentração top-3): {cr3:.1f}%\n"
                f"HHI: {hhi:.0f} ({'alta' if hhi > 2500 else 'moderada' if hhi > 1500 else 'baixa'} concentração)"
            )
        except (json.JSONDecodeError, TypeError) as e:
            return f"Erro: {e}. Forneça JSON válido com shares percentuais."

    pesquisador = Agent(
        role="Pesquisador de Mercado",
        goal="Coletar dados quantitativos sobre {tema} no Brasil",
        backstory=(
            "Especialista em pesquisa de mercado brasileiro com foco em dados verificáveis. "
            "Você busca dados numéricos concretos com fontes identificadas. "
            "Nunca inventa dados: se não encontrar, informa claramente."
        ),
        tools=[busca],
        llm=llm,
        max_iter=6
    )
    analista = Agent(
        role="Analista Quantitativo",
        goal="Calcular métricas e projeções sobre {tema}",
        backstory=(
            "Analista com experiência em modelagem financeira e análise de mercado. "
            "Seu trabalho é pegar dados brutos e produzir métricas precisas: CAGR, "
            "índices de concentração e projeções baseadas em tendências históricas."
        ),
        tools=[calcular_cagr, calcular_concentracao],
        llm=llm,
        max_iter=4
    )
    redator = Agent(
        role="Redator Executivo",
        goal="Criar relatório executivo claro sobre {tema}",
        backstory=(
            "Especialista em comunicação de análises para tomadores de decisão. "
            "Seu relatório é considerado bom quando o investidor consegue tomar uma decisão "
            "baseada nele sem precisar de pesquisa adicional. "
            "Tom: objetivo, dados primeiro, recomendação clara no final."
        ),
        llm=llm,
        max_iter=3
    )
    t1 = Task(
        description=(
            "Pesquise {tema} no Brasil. "
            "Colete: tamanho do mercado (R$), taxa de crescimento anual, "
            "market share dos 5 principais players, tendências de 2025, principais desafios."
        ),
        expected_output=(
            "Dados quantitativos com fontes. "
            "Top-5 players com market share estimado. "
            "Formato: tópicos estruturados com números e referências."
        ),
        agent=pesquisador
    )
    t2 = Task(
        description=(
            "Com base nos dados coletados sobre {tema}, calcule: "
            "CAGR do setor (use os dados de crescimento), "
            "índice de concentração (CR3 e HHI dos top players), "
            "e identifique os 2 dados mais relevantes para um investidor."
        ),
        expected_output=(
            "Análise quantitativa com: CAGR calculado, CR3 e HHI, "
            "projeção de tamanho do mercado para 2026, "
            "2 insights quantitativos mais relevantes para investimento."
        ),
        agent=analista,
        context=[t1]
    )
    t3 = Task(
        description=(
            "Com base na pesquisa e análise sobre {tema}, escreva um relatório executivo "
            "para um investidor considerando entrada no setor. "
            "Tom: objetivo, dados primeiro, recomendação clara no final."
        ),
        expected_output=(
            "Relatório executivo estruturado em 5 seções, 500-700 palavras: "
            "1) Sumário executivo (3 linhas), "
            "2) Contexto de mercado com dados quantitativos, "
            "3) Análise de concentração e players, "
            "4) Oportunidades e riscos identificados, "
            "5) Recomendação de investimento com justificativa."
        ),
        agent=redator,
        context=[t1, t2]
    )
    return Crew(
        agents=[pesquisador, analista, redator],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )


def auditar_outputs_crew(crew: Crew, inputs: dict) -> dict:
    """Executa a Crew e audita a qualidade de cada output."""
    resultado = crew.kickoff(inputs=inputs)
    auditoria = {}
    for i, task_output in enumerate(resultado.tasks_output):
        output_text = task_output.raw
        tem_numeros = any(c.isdigit() for c in output_text)
        tamanho_ok = len(output_text) > 200
        score = sum([tem_numeros, tamanho_ok, len(output_text) > 500])
        auditoria[f"task_{i+1}"] = {
            "comprimento": len(output_text),
            "tem_dados_numericos": tem_numeros,
            "score": f"{score}/3"
        }
    return {"auditoria": auditoria, "output_final": resultado.raw[:500]}


if __name__ == "__main__":
    crew = criar_crew_inteligencia_mercado()
    resultado = crew.kickoff(inputs={"tema": "agências digitais no Brasil"})
    print(resultado.raw)
