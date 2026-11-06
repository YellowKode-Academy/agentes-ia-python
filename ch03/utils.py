"""
Utilities for managing context and conversation history.
"""
from typing import Optional


def truncate_history(history: list[dict], max_messages: int = 20) -> list[dict]:
    """Keeps only the N most recent messages of the history."""
    return history[-max_messages:] if len(history) > max_messages else history


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ~ 4 characters in English)."""
    return len(text) // 4


def estimate_cost_usd(tokens: int, model: str = "sonnet") -> float:
    """Estimates the cost in USD based on the number of tokens.
    Reference values: Claude Sonnet (input $0.003/1K, output $0.015/1K).
    Uses a weighted average assuming 70% input, 30% output.
    """
    rates = {
        "sonnet": {"input": 0.003 / 1000, "output": 0.015 / 1000},
        "opus": {"input": 0.015 / 1000, "output": 0.075 / 1000},
        "haiku": {"input": 0.00025 / 1000, "output": 0.00125 / 1000},
    }
    r = rates.get(model, rates["sonnet"])
    input_tokens = int(tokens * 0.7)
    output_tokens = int(tokens * 0.3)
    return input_tokens * r["input"] + output_tokens * r["output"]


def format_history_for_context(
    history: list[dict],
    max_messages: int = 10
) -> str:
    """Formats the conversation history for inclusion in the prompt."""
    truncated_history = truncate_history(history, max_messages)
    if not truncated_history:
        return ""
    lines = []
    for exchange in truncated_history:
        lines.append(f"User: {exchange.get('human', '')}")
        lines.append(f"Assistant: {exchange.get('assistant', '')}")
    return "\n".join(lines)


def usage_report(history: list[dict]) -> dict:
    """Generates a report of token usage and estimated cost for a conversation."""
    total_chars = sum(
        len(t.get("human", "")) + len(t.get("assistant", ""))
        for t in history
    )
    tokens = estimate_tokens(total_chars)
    return {
        "messages": len(history),
        "total_characters": total_chars,
        "estimated_tokens": tokens,
        "estimated_cost_usd": round(estimate_cost_usd(tokens), 4),
        "estimated_cost_brl": round(estimate_cost_usd(tokens) * 5.8, 4),
    }
