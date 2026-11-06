from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.tools import tool
import json

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")


def build_market_intelligence_crew() -> Crew:
    """Creates the full market intelligence Crew with 3 specialized agents."""
    search = TavilySearch(max_results=5)

    @tool
    def calculate_cagr(initial_value: float, final_value: float, years: int) -> str:
        """Calculates CAGR (compound annual growth rate) between two values.
        Example: calculate_cagr(100, 200, 5) → CAGR over 5 years: 14.87%"""
        if initial_value <= 0 or years <= 0:
            return "Error: values must be positive."
        cagr = ((final_value / initial_value) ** (1 / years) - 1) * 100
        return f"CAGR over {years} years: {cagr:.2f}%"

    @tool
    def calculate_concentration(market_shares_json: str) -> str:
        """Calculates a market concentration index from a JSON dictionary of market shares.
        Input: JSON with {'company': percentage}. Ex: {"A": 30, "B": 20, "C": 15}"""
        try:
            shares = json.loads(market_shares_json)
            top3 = sorted(shares.values(), reverse=True)[:3]
            cr3 = sum(top3)
            hhi = sum(v**2 for v in shares.values())
            return (
                f"CR3 (top-3 concentration): {cr3:.1f}%\n"
                f"HHI: {hhi:.0f} ({'high' if hhi > 2500 else 'moderate' if hhi > 1500 else 'low'} concentration)"
            )
        except (json.JSONDecodeError, TypeError) as e:
            return f"Error: {e}. Provide valid JSON with percentage shares."

    researcher = Agent(
        role="Market Researcher",
        goal="Collect quantitative data on {topic} in Brazil",
        backstory=(
            "Brazilian market research specialist focused on verifiable data. "
            "You look for concrete numerical data with identified sources. "
            "You never make up data: if you can't find it, you say so clearly."
        ),
        tools=[search],
        llm=llm,
        max_iter=6
    )
    analyst = Agent(
        role="Quantitative Analyst",
        goal="Calculate metrics and projections on {topic}",
        backstory=(
            "Analyst with experience in financial modeling and market analysis. "
            "Your job is to take raw data and produce precise metrics: CAGR, "
            "concentration indexes and projections based on historical trends."
        ),
        tools=[calculate_cagr, calculate_concentration],
        llm=llm,
        max_iter=4
    )
    writer = Agent(
        role="Executive Writer",
        goal="Create a clear executive report on {topic}",
        backstory=(
            "Specialist in communicating analyses to decision makers. "
            "Your report is considered good when the investor can make a decision "
            "based on it without needing additional research. "
            "Tone: objective, data first, clear recommendation at the end."
        ),
        llm=llm,
        max_iter=3
    )
    t1 = Task(
        description=(
            "Research {topic} in Brazil. "
            "Collect: market size (R$), annual growth rate, "
            "market share of the 5 main players, 2025 trends, main challenges."
        ),
        expected_output=(
            "Quantitative data with sources. "
            "Top-5 players with estimated market share. "
            "Format: structured bullet points with numbers and references."
        ),
        agent=researcher
    )
    t2 = Task(
        description=(
            "Based on the data collected on {topic}, calculate: "
            "the sector's CAGR (use the growth data), "
            "concentration index (CR3 and HHI of the top players), "
            "and identify the 2 most relevant data points for an investor."
        ),
        expected_output=(
            "Quantitative analysis with: calculated CAGR, CR3 and HHI, "
            "market size projection for 2026, "
            "2 most relevant quantitative insights for investment."
        ),
        agent=analyst,
        context=[t1]
    )
    t3 = Task(
        description=(
            "Based on the research and analysis on {topic}, write an executive report "
            "for an investor considering entering the sector. "
            "Tone: objective, data first, clear recommendation at the end."
        ),
        expected_output=(
            "Executive report structured in 5 sections, 500-700 words: "
            "1) Executive summary (3 lines), "
            "2) Market context with quantitative data, "
            "3) Concentration and player analysis, "
            "4) Identified opportunities and risks, "
            "5) Investment recommendation with justification."
        ),
        agent=writer,
        context=[t1, t2]
    )
    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )


def audit_crew_outputs(crew: Crew, inputs: dict) -> dict:
    """Runs the Crew and audits the quality of each output."""
    result = crew.kickoff(inputs=inputs)
    audit = {}
    for i, task_output in enumerate(result.tasks_output):
        output_text = task_output.raw
        has_numbers = any(c.isdigit() for c in output_text)
        size_ok = len(output_text) > 200
        score = sum([has_numbers, size_ok, len(output_text) > 500])
        audit[f"task_{i+1}"] = {
            "length": len(output_text),
            "has_numeric_data": has_numbers,
            "score": f"{score}/3"
        }
    return {"audit": audit, "final_output": result.raw[:500]}


if __name__ == "__main__":
    crew = build_market_intelligence_crew()
    result = crew.kickoff(inputs={"topic": "digital agencies in Brazil"})
    print(result.raw)
