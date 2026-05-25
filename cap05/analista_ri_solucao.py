"""
Capítulo 5 — Solução do exercício
Agente RAG que responde às 10 perguntas padrão de um analista de equity
a partir de um relatório anual carregado como PDF.

Uso:
    python -m cap05.analista_ri_solucao dados/relatorio_anual.pdf TechVentures
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

from cap05.rag_tools import criar_retriever_de_pdf, criar_ferramenta_rag

load_dotenv()

PERGUNTAS_ANALISTA = [
    "Qual foi a receita líquida reportada no último exercício? Qual foi o crescimento em relação ao ano anterior?",
    "Qual é a margem EBITDA da empresa? Como evoluiu nos últimos anos?",
    "Quais são os principais riscos de mercado que a empresa identifica no relatório?",
    "Qual é a estratégia de expansão geográfica declarada pela empresa?",
    "Qual é a exposição cambial da empresa? Como ela é gerenciada?",
    "Quais são os principais produtos/serviços e qual a participação de cada um na receita?",
    "Como está a estrutura de capital? Qual é o nível de endividamento?",
    "Quais são os principais concorrentes mencionados e como a empresa se posiciona frente a eles?",
    "A empresa pagou dividendos? Qual é a política de distribuição de resultados?",
    "Quais são as perspectivas e guidance para o próximo exercício?",
]


def criar_agente_analista(caminho_pdf: str, nome_empresa: str):
    """Cria agente com ferramenta RAG para o relatório anual."""
    retriever = criar_retriever_de_pdf(caminho_pdf, nome_empresa.lower().replace(" ", "_"))
    ferramenta_doc = criar_ferramenta_rag(
        retriever=retriever,
        nome=f"consultar_{nome_empresa.lower().replace(' ', '_')}",
        descricao=(
            f"Consulta o relatório anual oficial da {nome_empresa}. "
            f"Use para perguntas sobre: receita, margens, riscos, estratégia, "
            f"exposição cambial, dividendos, concorrentes e guidance da {nome_empresa}."
        )
    )
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [ferramenta_doc, TavilySearch(max_results=2)]
    return create_agent(llm, tools)


def executar_analise_equity(caminho_pdf: str, nome_empresa: str) -> None:
    """Executa as 10 perguntas padrão e exibe o relatório."""
    print(f"\n{'='*60}")
    print(f"ANÁLISE EQUITY — {nome_empresa.upper()}")
    print(f"Fonte: {caminho_pdf}")
    print(f"{'='*60}\n")

    if not Path(caminho_pdf).exists():
        print(f"AVISO: arquivo '{caminho_pdf}' não encontrado.")
        print("Crie a pasta 'dados/' e adicione o PDF do relatório anual.")
        print("\nDica: relatórios anuais de empresas brasileiras estão disponíveis")
        print("em: ri.[empresa].com.br ou na seção de RI do site da empresa.\n")
        return

    agent = criar_agente_analista(caminho_pdf, nome_empresa)

    for i, pergunta in enumerate(PERGUNTAS_ANALISTA, 1):
        print(f"[{i:02d}/10] {pergunta}")
        try:
            resultado = agent.invoke({
                "messages": [{"role": "user", "content": pergunta}]
            })
            resposta = resultado["messages"][-1].content
            print(f"Resposta: {resposta[:400]}{'...' if len(resposta) > 400 else ''}")
        except Exception as e:
            print(f"Erro ao processar pergunta: {e}")
        print()


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pdf = sys.argv[1]
        empresa = sys.argv[2]
    else:
        pdf = "dados/relatorio_anual.pdf"
        empresa = "Empresa"
    executar_analise_equity(pdf, empresa)
