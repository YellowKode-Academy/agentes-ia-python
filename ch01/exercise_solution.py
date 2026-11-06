"""
Chapter 1: Exercise solution
The agent reasons about uncertainty and verification, with an analysis of the answer pattern.
"""
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

load_dotenv()


def ask(agent, text: str) -> str:
    result = agent.invoke({
        "messages": [{"role": "user", "content": text}]
    })
    return result["messages"][-1].content


def analyze_response_pattern(response: str) -> dict:
    """Analyzes uncertainty patterns in the agent's response."""
    uncertainty_markers = [
        "i'm not sure", "might be", "probably", "i believe",
        "maybe", "possibly", "i estimate", "i don't know for sure"
    ]
    confidence_markers = [
        "certainly", "definitely", "for sure", "it is a fact",
        "confirmed", "verified"
    ]
    uncertainty = sum(1 for m in uncertainty_markers if m in response.lower())
    confidence = sum(1 for m in confidence_markers if m in response.lower())
    return {
        "uncertainty_markers": uncertainty,
        "confidence_markers": confidence,
        "tone": "cautious" if uncertainty > confidence else "assertive"
    }


if __name__ == "__main__":
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    agent = create_agent(llm, tools=[])

    questions = [
        "When you are not sure about an answer, what do you do?",
        "How do you decide when an answer is good enough?",
        "What is the difference between a probable answer and a verified answer?",
    ]

    print("=== ANALYSIS OF THE AGENT'S REASONING PATTERN ===\n")
    for question in questions:
        print(f"Q: {question}")
        response = ask(agent, question)
        print(f"A: {response}")
        analysis = analyze_response_pattern(response)
        print(f"[Analysis] Tone: {analysis['tone']} | "
              f"Uncertainty: {analysis['uncertainty_markers']} | "
              f"Confidence: {analysis['confidence_markers']}")
        print()
