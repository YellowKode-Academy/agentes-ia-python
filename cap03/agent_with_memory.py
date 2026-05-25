from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic import hub

load_dotenv()

_agent_executor = None
_historico: dict[str, list] = {}


def criar_agente():
    """Cria agente com memória de curto prazo simulada via histórico de mensagens."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [TavilySearchResults(max_results=3)]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False,
                         handle_parsing_errors=True, max_iterations=6)


def conversar(agent_executor, mensagem: str, thread_id: str = "default") -> str:
    """Envia mensagem ao agente mantendo contexto da conversa via histórico acumulado."""
    if thread_id not in _historico:
        _historico[thread_id] = []

    historico = _historico[thread_id]
    contexto = ""
    if historico:
        contexto = "Histórico da conversa:\n"
        for troca in historico[-5:]:
            contexto += f"Usuário: {troca['humano']}\nAssistente: {troca['assistente']}\n\n"
        contexto += "Nova mensagem: "

    input_completo = contexto + mensagem
    result = agent_executor.invoke({"input": input_completo})
    resposta = result.get("output", "")

    _historico[thread_id].append({"humano": mensagem, "assistente": resposta})
    return resposta


def limpar_historico(thread_id: str) -> None:
    """Remove o histórico de uma thread específica."""
    _historico.pop(thread_id, None)


if __name__ == "__main__":
    executor = criar_agente()

    print(conversar(executor, "Meu nome é Kelvin.", thread_id="teste"))
    print(conversar(executor, "Qual é o meu nome?", thread_id="teste"))
    print(conversar(executor, "Estou analisando o setor de fintechs.", thread_id="analise-01"))
    print(conversar(executor, "Quais são os principais players nesse setor?", thread_id="analise-01"))
