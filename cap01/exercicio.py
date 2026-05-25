"""
Capítulo 1 — Exercício
Faz o agente raciocinar sobre o próprio processo de raciocínio.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub

load_dotenv()


def perguntar(agent_executor, texto: str) -> str:
    result = agent_executor.invoke({"input": texto})
    return result.get("output", "")


if __name__ == "__main__":
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools=[], prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=[], verbose=False)

    perguntas = [
        "Quando você não tem certeza de uma resposta, o que você faz?",
        "Como você decide quando uma resposta é boa o suficiente?",
        "Qual a diferença entre uma resposta provável e uma resposta verificada?",
    ]

    for pergunta in perguntas:
        print(f"P: {pergunta}")
        print(f"R: {perguntar(executor, pergunta)}")
        print()
