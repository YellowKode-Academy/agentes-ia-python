"""
Chapter 3: Exercise solution
Script that runs 5 exchanges with the agent, prints the full history
and calculates the estimated cost of the conversation.
"""
from dotenv import load_dotenv
from ch03.agent_with_memory import build_agent, chat, _history
from ch03.utils import usage_report, estimate_tokens

load_dotenv()

THREAD_ID = "exercise-ch03"

QUESTIONS = [
    "I'm going to analyze the healthtech sector in Brazil. The focus is telemedicine.",
    "Who are the main players in this segment?",
    "What is the estimated size of the market you mentioned?",
    "What are the biggest risks for a startup entering this market right now?",
    "Given this analysis, would you recommend entering this market?",
]


if __name__ == "__main__":
    executor = build_agent()

    print("=== STARTING CONVERSATION ===\n")
    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/5] Question: {question}")
        response = chat(executor, question, thread_id=THREAD_ID)
        print(f"Answer: {response[:300]}{'...' if len(response) > 300 else ''}")
        print()

    history = _history.get(THREAD_ID, [])

    print("=== USAGE REPORT ===")
    report = usage_report(history)
    print(f"Messages exchanged: {report['messages']}")
    print(f"Estimated tokens: {report['estimated_tokens']:,}")
    print(f"Estimated cost (USD): ${report['estimated_cost_usd']:.4f}")
    print(f"Estimated cost (BRL): R${report['estimated_cost_brl']:.4f}")

    print("\n=== FULL HISTORY ===")
    for i, exchange in enumerate(history, 1):
        print(f"[{i}] Human: {exchange['human'][:100]}")
        print(f"    Assistant: {exchange['assistant'][:200]}...")
        print()
