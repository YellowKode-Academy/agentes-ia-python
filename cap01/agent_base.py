from dotenv import load_dotenv
from langchain.agents import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain import hub

load_dotenv()


def criar_agente():
    """Cria e retorna o agente base do projeto."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    prompt = hub.pull("hwchase17/react")
    return create_react_agent(llm, tools=[], prompt=prompt)


def perguntar(agent, texto: str) -> str:
    """Envia uma pergunta ao agente e retorna a resposta como texto."""
    from langchain.agents import AgentExecutor
    executor = AgentExecutor(agent=agent, tools=[], verbose=False)
    result = executor.invoke({"input": texto})
    return result.get("output", "")


if __name__ == "__main__":
    agent = criar_agente()
    print(perguntar(agent, "Qual é a raiz quadrada de 144 elevada ao quadrado?"))
