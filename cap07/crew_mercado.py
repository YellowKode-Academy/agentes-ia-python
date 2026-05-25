from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")
busca = TavilySearch(max_results=5)


def criar_crew_mercado(tema: str) -> str:
    """Executa análise de mercado completa para o tema dado."""
    pesquisador = Agent(
        role="Pesquisador de Mercado",
        goal=f"Coletar dados precisos sobre {tema} no Brasil",
        backstory=(
            "Especialista em pesquisa de mercado brasileiro com foco em dados verificáveis. "
            "Você nunca inventa dados: se não encontrar, informa que não encontrou."
        ),
        tools=[busca],
        llm=llm,
        max_iter=6
    )
    analista = Agent(
        role="Analista de Negócios",
        goal=f"Transformar pesquisa sobre {tema} em análise executiva",
        backstory=(
            "Analista sênior com experiência em estratégia de negócios. "
            "Transforma dados em análises claras com conclusões diretas. "
            "Seu relatório sempre termina com 3 recomendações práticas numeradas."
        ),
        llm=llm,
        max_iter=4
    )
    t_pesquisa = Task(
        description=(
            f"Pesquise o mercado de {tema} no Brasil. "
            "Colete: tamanho do mercado (em R$), principais players com estimativa de market share, "
            "taxa de crescimento anual, tendências de 2025 e principais desafios do setor."
        ),
        expected_output=(
            "Dados estruturados com fontes. Formato: tópicos com dados quantitativos. "
            f"Inclua: tamanho do mercado de {tema}, lista de top-5 players, "
            "3 tendências identificadas e 2 desafios principais."
        ),
        agent=pesquisador
    )
    t_analise = Task(
        description=(
            f"Com base na pesquisa sobre {tema}, crie relatório executivo para investidor brasileiro. "
            "Inclua sumário executivo, análise de oportunidades, análise de riscos e recomendações."
        ),
        expected_output=(
            "Relatório executivo de 500-700 palavras com: "
            "sumário executivo (3 linhas), análise de oportunidades, "
            "análise de riscos e 3 recomendações práticas numeradas."
        ),
        agent=analista,
        context=[t_pesquisa]
    )
    crew = Crew(
        agents=[pesquisador, analista],
        tasks=[t_pesquisa, t_analise],
        process=Process.sequential
    )
    return crew.kickoff(inputs={"tema": tema}).raw


if __name__ == "__main__":
    resultado = criar_crew_mercado("fintechs de pagamento")
    print(resultado)
