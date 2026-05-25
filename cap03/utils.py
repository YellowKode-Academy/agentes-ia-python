"""
Utilitários para gerenciamento de contexto e histórico de conversas.
"""
from typing import Optional


def truncar_historico(historico: list[dict], max_mensagens: int = 20) -> list[dict]:
    """Mantém apenas as N mensagens mais recentes do histórico."""
    return historico[-max_mensagens:] if len(historico) > max_mensagens else historico


def estimar_tokens(texto: str) -> int:
    """Estimativa aproximada de tokens (1 token ~ 4 caracteres em português)."""
    return len(texto) // 4


def estimar_custo_usd(tokens: int, tipo: str = "sonnet") -> float:
    """Estima custo em USD com base no número de tokens.
    Valores de referência: Claude Sonnet (input $0.003/1K, output $0.015/1K).
    Usa média ponderada assumindo 70% input, 30% output.
    """
    tarifas = {
        "sonnet": {"input": 0.003 / 1000, "output": 0.015 / 1000},
        "opus": {"input": 0.015 / 1000, "output": 0.075 / 1000},
        "haiku": {"input": 0.00025 / 1000, "output": 0.00125 / 1000},
    }
    t = tarifas.get(tipo, tarifas["sonnet"])
    tokens_input = int(tokens * 0.7)
    tokens_output = int(tokens * 0.3)
    return tokens_input * t["input"] + tokens_output * t["output"]


def formatar_historico_para_contexto(
    historico: list[dict],
    max_mensagens: int = 10
) -> str:
    """Formata o histórico de conversas para inclusão no prompt."""
    historico_truncado = truncar_historico(historico, max_mensagens)
    if not historico_truncado:
        return ""
    linhas = []
    for troca in historico_truncado:
        linhas.append(f"Usuário: {troca.get('humano', '')}")
        linhas.append(f"Assistente: {troca.get('assistente', '')}")
    return "\n".join(linhas)


def relatorio_uso(historico: list[dict]) -> dict:
    """Gera relatório de uso de tokens e custo estimado de uma conversa."""
    total_chars = sum(
        len(t.get("humano", "")) + len(t.get("assistente", ""))
        for t in historico
    )
    tokens = estimar_tokens(total_chars)
    return {
        "mensagens": len(historico),
        "caracteres_total": total_chars,
        "tokens_estimados": tokens,
        "custo_estimado_usd": round(estimar_custo_usd(tokens), 4),
        "custo_estimado_brl": round(estimar_custo_usd(tokens) * 5.8, 4),
    }
