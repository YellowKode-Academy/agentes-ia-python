"""
Capítulo 1 — Exercício
Faz o agente raciocinar sobre o próprio processo de raciocínio.
"""
from agent_base import criar_agente, perguntar

agent = criar_agente()

perguntas = [
    "Quando você não tem certeza de uma resposta, o que você faz?",
    "Como você decide quando uma resposta é boa o suficiente?",
    "Qual a diferença entre uma resposta provável e uma resposta verificada?",
]

for pergunta in perguntas:
    print(f"P: {pergunta}")
    print(f"R: {perguntar(agent, pergunta)}")
    print()
