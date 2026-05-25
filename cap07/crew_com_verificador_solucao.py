"""
Capítulo 7 — Solução do exercício
Crew com 3 agentes: Pesquisador, Analista e Verificador de Fatos.
O Verificador recebe o relatório do analista e verifica se os dados
quantitativos são consistentes com os dados coletados pelo pesquisador,
emitindo um score de confiabilidade (0-10).
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")
busca = TavilySearchResults(max_results=5)


def criar_crew_com_verificador(tema: str) -> dict:
    """Crew com pesquisador, analista e verificador de fatos."""
    pesquisador = Agent(
        role="Pesquisador de Mercado",
        goal=f"Coletar dados precisos sobre {tema} no Brasil",
        backstory=(
            "Especialista em pesquisa de mercado com foco em dados verificáveis. "
            "Você nunca inventa dados: se não encontrar, informa claramente."
        ),
        tools=[busca],
        llm=llm,
        max_iter=6
    )
    analista = Agent(
        role="Analista de Negócios",
        goal=f"Transformar pesquisa sobre {tema} em análise executiva",
        backstory=(
            "Analista sênior que transforma dados em estratégia. "
            "Produz análises claras com conclusões diretas e 3 recomendações práticas."
        ),
        llm=llm,
        max_iter=4
    )
    verificador = Agent(
        role="Verificador de Fatos",
        goal=f"Verificar se o relatório sobre {tema} é consistente com os dados coletados",
        backstory=(
            "Especialista em fact-checking para análises de mercado. "
            "Você compara o relatório com os dados brutos e identifica: "
            "(1) dados citados corretamente, (2) dados distorcidos, (3) afirmações sem base nos dados. "
            "Emite um score de confiabilidade de 0 a 10 com justificativa detalhada."
        ),
        llm=llm,
        max_iter=3
    )

    t_pesquisa = Task(
        description=(
            f"Pesquise o mercado de {tema} no Brasil. "
            "Colete: tamanho do mercado, principais players com market share, "
            "taxa de crescimento, tendências de 2025 e desafios do setor."
        ),
        expected_output=(
            "Dados quantitativos com fontes claras. "
            "Top-5 players com market share estimado. "
            "3 tendências e 2 desafios principais."
        ),
        agent=pesquisador
    )
    t_analise = Task(
        description=(
            f"Com base nos dados coletados sobre {tema}, produza análise executiva "
            "para um investidor. Inclua sumário, oportunidades, riscos e recomendações."
        ),
        expected_output=(
            "Relatório executivo de 400-600 palavras com: "
            "sumário executivo, análise de oportunidades, análise de riscos, "
            "3 recomendações práticas numeradas."
        ),
        agent=analista,
        context=[t_pesquisa]
    )
    t_verificacao = Task(
        description=(
            f"Verifique a consistência do relatório sobre {tema} com os dados de pesquisa. "
            "Para cada dado quantitativo citado no relatório: "
            "1) está presente nos dados de pesquisa? "
            "2) foi representado corretamente? "
            "3) houve exagero ou distorção? "
            "Emita um score de confiabilidade de 0 a 10."
        ),
        expected_output=(
            "Relatório de verificação estruturado com: "
            "lista de dados verificados (corretos/distorcidos/sem base), "
            "score de confiabilidade (0-10) com justificativa, "
            "e recomendação: 'APROVADO' (score >= 7) ou 'REVISAR' (score < 7) com o que corrigir."
        ),
        agent=verificador,
        context=[t_pesquisa, t_analise]
    )

    crew = Crew(
        agents=[pesquisador, analista, verificador],
        tasks=[t_pesquisa, t_analise, t_verificacao],
        process=Process.sequential
    )
    resultado = crew.kickoff(inputs={"tema": tema})
    return {
        "relatorio": resultado.tasks_output[1].raw if len(resultado.tasks_output) > 1 else "",
        "verificacao": resultado.tasks_output[2].raw if len(resultado.tasks_output) > 2 else "",
        "output_final": resultado.raw
    }


if __name__ == "__main__":
    resultado = criar_crew_com_verificador("healthtechs de telemedicina no Brasil")
    print("=== RELATÓRIO ===")
    print(resultado["relatorio"][:600])
    print("\n=== VERIFICAÇÃO DE FATOS ===")
    print(resultado["verificacao"])
