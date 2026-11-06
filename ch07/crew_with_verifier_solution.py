"""
Chapter 7: Exercise solution
Crew with 3 agents: Researcher, Analyst and Fact Checker.
The Fact Checker receives the analyst's report and checks whether the quantitative
data is consistent with the data collected by the researcher,
issuing a reliability score (0-10).
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")
search = TavilySearch(max_results=5)


def build_crew_with_verifier(topic: str) -> dict:
    """Crew with a researcher, an analyst and a fact checker."""
    researcher = Agent(
        role="Market Researcher",
        goal=f"Collect accurate data on {topic} in Brazil",
        backstory=(
            "Market research specialist focused on verifiable data. "
            "You never make up data: if you can't find it, you say so clearly."
        ),
        tools=[search],
        llm=llm,
        max_iter=6
    )
    analyst = Agent(
        role="Business Analyst",
        goal=f"Turn research on {topic} into an executive analysis",
        backstory=(
            "Senior analyst who turns data into strategy. "
            "Produces clear analyses with direct conclusions and 3 practical recommendations."
        ),
        llm=llm,
        max_iter=4
    )
    verifier = Agent(
        role="Fact Checker",
        goal=f"Check whether the report on {topic} is consistent with the collected data",
        backstory=(
            "Fact-checking specialist for market analyses. "
            "You compare the report with the raw data and identify: "
            "(1) data cited correctly, (2) distorted data, (3) claims with no basis in the data. "
            "You issue a reliability score from 0 to 10 with a detailed justification."
        ),
        llm=llm,
        max_iter=3
    )

    t_research = Task(
        description=(
            f"Research the {topic} market in Brazil. "
            "Collect: market size, main players with market share, "
            "growth rate, 2025 trends and the sector's challenges."
        ),
        expected_output=(
            "Quantitative data with clear sources. "
            "Top-5 players with estimated market share. "
            "3 trends and 2 main challenges."
        ),
        agent=researcher
    )
    t_analysis = Task(
        description=(
            f"Based on the data collected on {topic}, produce an executive analysis "
            "for an investor. Include summary, opportunities, risks and recommendations."
        ),
        expected_output=(
            "Executive report of 400-600 words with: "
            "executive summary, opportunity analysis, risk analysis, "
            "3 numbered practical recommendations."
        ),
        agent=analyst,
        context=[t_research]
    )
    t_verification = Task(
        description=(
            f"Check the consistency of the report on {topic} against the research data. "
            "For each quantitative data point cited in the report: "
            "1) is it present in the research data? "
            "2) was it represented correctly? "
            "3) was there exaggeration or distortion? "
            "Issue a reliability score from 0 to 10."
        ),
        expected_output=(
            "Structured verification report with: "
            "list of checked data (correct/distorted/no basis), "
            "reliability score (0-10) with justification, "
            "and recommendation: 'APPROVED' (score >= 7) or 'REVIEW' (score < 7) with what to fix."
        ),
        agent=verifier,
        context=[t_research, t_analysis]
    )

    crew = Crew(
        agents=[researcher, analyst, verifier],
        tasks=[t_research, t_analysis, t_verification],
        process=Process.sequential
    )
    result = crew.kickoff(inputs={"topic": topic})
    return {
        "report": result.tasks_output[1].raw if len(result.tasks_output) > 1 else "",
        "verification": result.tasks_output[2].raw if len(result.tasks_output) > 2 else "",
        "final_output": result.raw
    }


if __name__ == "__main__":
    result = build_crew_with_verifier("telemedicine healthtechs in Brazil")
    print("=== REPORT ===")
    print(result["report"][:600])
    print("\n=== FACT CHECK ===")
    print(result["verification"])
