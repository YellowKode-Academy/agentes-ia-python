"""
Capítulo 1 — Solução do exercício
Agente raciocina sobre incerteza e verificação, com análise do padrão de resposta.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()


def perguntar(executor, texto: str) -> str:
    result = executor.invoke({"input": texto})
    return result.get("output", "")


def analisar_padrao_resposta(resposta: str) -> dict:
    """Analisa padrões de incerteza na resposta do agente."""
    marcadores_incerteza = [
        "não tenho certeza", "pode ser", "provavelmente", "acredito",
        "talvez", "possivelmente", "estimo", "não sei ao certo"
    ]
    marcadores_confianca = [
        "certamente", "definitivamente", "com certeza", "é fato",
        "confirmado", "verificado"
    ]
    incerteza = sum(1 for m in marcadores_incerteza if m in resposta.lower())
    confianca = sum(1 for m in marcadores_confianca if m in resposta.lower())
    return {
        "marcadores_incerteza": incerteza,
        "marcadores_confianca": confianca,
        "tom": "cauteloso" if incerteza > confianca else "assertivo"
    }


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

    print("=== ANÁLISE DO PADRÃO DE RACIOCÍNIO DO AGENTE ===\n")
    for pergunta in perguntas:
        print(f"P: {pergunta}")
        resposta = perguntar(executor, pergunta)
        print(f"R: {resposta}")
        analise = analisar_padrao_resposta(resposta)
        print(f"[Análise] Tom: {analise['tom']} | "
              f"Incerteza: {analise['marcadores_incerteza']} | "
              f"Confiança: {analise['marcadores_confianca']}")
        print()
