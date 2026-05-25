from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

load_dotenv()


def criar_agente():
    """Cria e retorna o agente base do projeto."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    return create_agent(llm, tools=[])


def perguntar(agent, texto: str) -> str:
    """Envia uma pergunta ao agente e retorna a resposta como texto."""
    result = agent.invoke({
        "messages": [{"role": "user", "content": texto}]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = criar_agente()
    print(perguntar(agent, "Qual é a raiz quadrada de 144 elevada ao quadrado?"))
