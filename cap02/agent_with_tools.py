import json
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain import hub

load_dotenv()


def criar_ferramentas():
    """Cria e retorna a lista de ferramentas do agente de mercado."""
    search = TavilySearchResults(max_results=3)

    @tool
    def calcular_percentual(valor: float, percentual: float) -> str:
        """Calcula percentual de um valor. Use para desconto, imposto ou comissão."""
        resultado = valor * (percentual / 100)
        return f"{percentual}% de {valor} = {resultado:.2f}"

    @tool
    def ler_dados_concorrente(nome_arquivo: str) -> str:
        """Lê dados de um concorrente a partir de um arquivo JSON em dados/.
        Use quando o usuário pedir análise de um concorrente específico."""
        caminho = Path(f"dados/{nome_arquivo}.json")
        if not caminho.exists():
            disponiveis = [p.stem for p in Path("dados").glob("*.json")]
            return f"Arquivo não encontrado. Disponíveis: {disponiveis}"
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        return json.dumps(dados, ensure_ascii=False, indent=2)

    @tool
    def salvar_analise(nome_arquivo: str, conteudo: str) -> str:
        """Salva uma análise ou relatório em arquivo de texto em resultados/.
        Use quando o usuário pedir para salvar, exportar ou registrar um resultado."""
        caminho = Path(f"resultados/{nome_arquivo}")
        caminho.parent.mkdir(exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
        return f"Análise salva em resultados/{nome_arquivo} ({len(conteudo)} caracteres)"

    return [search, calcular_percentual, ler_dados_concorrente, salvar_analise]


def criar_agente():
    """Cria o agente de mercado com ferramentas de busca e análise."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = criar_ferramentas()
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def perguntar(executor, texto: str) -> str:
    """Envia uma pergunta ao agente e retorna a resposta como texto."""
    result = executor.invoke({"input": texto})
    return result.get("output", "")


if __name__ == "__main__":
    executor = criar_agente()
    print(perguntar(executor, "Qual é a cotação do dólar hoje?"))
