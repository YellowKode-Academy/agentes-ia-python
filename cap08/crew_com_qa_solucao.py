"""
Capítulo 8 — Solução do exercício
Crew com 4 tasks: Pesquisa, Análise quantitativa, Relatório e Verificação de Qualidade.
A 4ª task avalia o relatório final segundo critérios predefinidos
(completude, precisão dos dados, clareza das recomendações) e atribui score 0-10.
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.tools import tool
import json

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")


def criar_crew_com_qa(tema: str) -> dict:
    """Crew de inteligência de mercado com verificação de qualidade (QA)."""
    busca = TavilySearchResults(max_results=5)

    @tool
    def calcular_cagr(valor_inicial: float, valor_final: float, anos: int) -> str:
        """Calcula CAGR (taxa de crescimento anual composta)."""
        if valor_inicial <= 0 or anos <= 0:
            return "Erro: valores devem ser positivos."
        cagr = ((valor_final / valor_inicial) ** (1 / anos) - 1) * 100
        return f"CAGR em {anos} anos: {cagr:.2f}%"

    pesquisador = Agent(
        role="Pesquisador de Mercado",
        goal=f"Coletar dados quantitativos sobre {tema}",
        backstory="Especialista em pesquisa de mercado. Coleta dados com fontes identificadas.",
        tools=[busca], llm=llm, max_iter=6
    )
    analista = Agent(
        role="Analista Quantitativo",
        goal=f"Calcular métricas sobre {tema}",
        backstory="Analista financeiro especialista em modelagem e métricas de mercado.",
        tools=[calcular_cagr], llm=llm, max_iter=4
    )
    redator = Agent(
        role="Redator Executivo",
        goal=f"Criar relatório executivo sobre {tema}",
        backstory=(
            "Especialista em comunicação para tomadores de decisão. "
            "Relatório precisa ter dados quantitativos, análise e recomendação clara."
        ),
        llm=llm, max_iter=3
    )
    qa = Agent(
        role="Analista de Qualidade (QA)",
        goal=f"Avaliar a qualidade do relatório sobre {tema}",
        backstory=(
            "Especialista em controle de qualidade de análises de mercado. "
            "Avalia relatórios por: completude (todos os dados necessários estão presentes?), "
            "precisão (os dados citados são consistentes com a pesquisa?), "
            "clareza das recomendações (são acionáveis e específicas?). "
            "Emite score 0-10 por critério e score geral, com justificativa."
        ),
        llm=llm, max_iter=3
    )

    t1 = Task(
        description=f"Pesquise {tema}: tamanho, players, crescimento, desafios.",
        expected_output="Dados quantitativos com fontes. Top-5 players com market share.",
        agent=pesquisador
    )
    t2 = Task(
        description=f"Calcule CAGR e métricas de concentração do setor de {tema}.",
        expected_output="CAGR calculado, concentração de mercado, projeção 2026.",
        agent=analista, context=[t1]
    )
    t3 = Task(
        description=(
            f"Escreva relatório executivo sobre {tema} para investidor. "
            "Inclua: sumário executivo, contexto de mercado, análise quantitativa, recomendação."
        ),
        expected_output="Relatório de 500-700 palavras estruturado em 5 seções.",
        agent=redator, context=[t1, t2]
    )
    t4 = Task(
        description=(
            f"Avalie a qualidade do relatório sobre {tema} segundo os critérios: "
            "1) COMPLETUDE: todos os dados relevantes estão presentes? (0-10) "
            "2) PRECISÃO: os dados citados são consistentes com a pesquisa? (0-10) "
            "3) CLAREZA DAS RECOMENDAÇÕES: são acionáveis e específicas? (0-10) "
            "Calcule o score geral (média dos 3 critérios) e indique: APROVADO (>= 7) ou REVISAR (< 7)."
        ),
        expected_output=(
            "Relatório de QA com: score por critério (0-10), "
            "score geral (média), veredicto (APROVADO/REVISAR), "
            "e lista de melhorias necessárias se score < 7."
        ),
        agent=qa, context=[t1, t2, t3]
    )

    crew = Crew(
        agents=[pesquisador, analista, redator, qa],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential
    )
    resultado = crew.kickoff(inputs={"tema": tema})
    outputs = resultado.tasks_output
    return {
        "pesquisa": outputs[0].raw if len(outputs) > 0 else "",
        "analise": outputs[1].raw if len(outputs) > 1 else "",
        "relatorio": outputs[2].raw if len(outputs) > 2 else "",
        "qa": outputs[3].raw if len(outputs) > 3 else "",
        "output_final": resultado.raw
    }


if __name__ == "__main__":
    resultado = criar_crew_com_qa("edtechs de ensino superior no Brasil")
    print("=== RELATÓRIO EXECUTIVO ===")
    print(resultado["relatorio"][:600])
    print("\n=== AVALIAÇÃO DE QUALIDADE (QA) ===")
    print(resultado["qa"])
