"""
Capítulo 3 — Solução do exercício
Script que executa 5 trocas com o agente, imprime o histórico completo
e calcula custo estimado da conversa.
"""
from dotenv import load_dotenv
from cap03.agent_with_memory import criar_agente, conversar, _historico
from cap03.utils import relatorio_uso, estimar_tokens

load_dotenv()

THREAD_ID = "exercicio-cap03"

PERGUNTAS = [
    "Vou analisar o setor de healthtechs no Brasil. O foco é telemedicina.",
    "Quais são os principais players desse segmento?",
    "Qual o tamanho estimado do mercado que você mencionou?",
    "Quais são os maiores riscos para uma startup entrando nesse mercado agora?",
    "Dada essa análise, você recomendaria entrar nesse mercado?",
]


if __name__ == "__main__":
    executor = criar_agente()

    print("=== INICIANDO CONVERSA ===\n")
    for i, pergunta in enumerate(PERGUNTAS, 1):
        print(f"[{i}/5] Pergunta: {pergunta}")
        resposta = conversar(executor, pergunta, thread_id=THREAD_ID)
        print(f"Resposta: {resposta[:300]}{'...' if len(resposta) > 300 else ''}")
        print()

    historico = _historico.get(THREAD_ID, [])

    print("=== RELATÓRIO DE USO ===")
    relatorio = relatorio_uso(historico)
    print(f"Mensagens trocadas: {relatorio['mensagens']}")
    print(f"Tokens estimados: {relatorio['tokens_estimados']:,}")
    print(f"Custo estimado (USD): ${relatorio['custo_estimado_usd']:.4f}")
    print(f"Custo estimado (BRL): R${relatorio['custo_estimado_brl']:.4f}")

    print("\n=== HISTÓRICO COMPLETO ===")
    for i, troca in enumerate(historico, 1):
        print(f"[{i}] Humano: {troca['humano'][:100]}")
        print(f"    Assistente: {troca['assistente'][:200]}...")
        print()
