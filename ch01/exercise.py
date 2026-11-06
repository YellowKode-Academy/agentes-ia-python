"""
Chapter 1: Exercise
Makes the agent reason about its own reasoning process.
"""
from agent_base import build_agent, ask

agent = build_agent()

questions = [
    "When you are not sure about an answer, what do you do?",
    "How do you decide when an answer is good enough?",
    "What is the difference between a probable answer and a verified answer?",
]

for question in questions:
    print(f"Q: {question}")
    print(f"A: {ask(agent, question)}")
    print()
