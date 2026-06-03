from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Histórico externo por thread_id: {"thread_id": [{"humano": ..., "assistente": ...}]}
_historico: dict[str, list[dict]] = {}


def criar_agente():
    """Cria agente com memória de curto prazo e ferramentas."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [TavilySearch(max_results=3)]
    memoria = MemorySaver()
    return create_agent(llm, tools, checkpointer=memoria)


def conversar(agent, mensagem: str, thread_id: str = "default") -> str:
    """Envia mensagem para o agente mantendo o contexto da conversa."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": mensagem}]},
        config=config
    )
    resposta = result["messages"][-1].content

    # Registrar no histórico externo para inspeção/relatório
    if thread_id not in _historico:
        _historico[thread_id] = []
    _historico[thread_id].append({"humano": mensagem, "assistente": resposta})

    return resposta


if __name__ == "__main__":
    agent = criar_agente()

    print(conversar(agent, "Meu nome é Kelvin.", thread_id="teste"))
    print(conversar(agent, "Qual é o meu nome?", thread_id="teste"))
    print(conversar(agent, "Estou analisando o setor de fintechs.", thread_id="analise-01"))
    print(conversar(agent, "Quais são os principais players nesse setor?", thread_id="analise-01"))
