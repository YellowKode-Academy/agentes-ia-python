"""
Chapter 8: Exercise solution
Crew with 4 tasks: Research, Quantitative analysis, Report and Quality Check.
The 4th task evaluates the final report against predefined criteria
(completeness, data accuracy, clarity of recommendations) and assigns a 0-10 score.
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.tools import tool
import json

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")


def build_crew_with_qa(topic: str) -> dict:
    """Market intelligence Crew with quality assurance (QA)."""
    search = TavilySearch(max_results=5)

    @tool
    def calculate_cagr(initial_value: float, final_value: float, years: int) -> str:
        """Calculates CAGR (compound annual growth rate)."""
        if initial_value <= 0 or years <= 0:
            return "Error: values must be positive."
        cagr = ((final_value / initial_value) ** (1 / years) - 1) * 100
        return f"CAGR over {years} years: {cagr:.2f}%"

    researcher = Agent(
        role="Market Researcher",
        goal=f"Collect quantitative data on {topic}",
        backstory="Market research specialist. Collects data with identified sources.",
        tools=[search], llm=llm, max_iter=6
    )
    analyst = Agent(
        role="Quantitative Analyst",
        goal=f"Calculate metrics on {topic}",
        backstory="Financial analyst specialized in modeling and market metrics.",
        tools=[calculate_cagr], llm=llm, max_iter=4
    )
    writer = Agent(
        role="Executive Writer",
        goal=f"Create an executive report on {topic}",
        backstory=(
            "Specialist in communication for decision makers. "
            "The report needs quantitative data, analysis and a clear recommendation."
        ),
        llm=llm, max_iter=3
    )
    qa = Agent(
        role="Quality Analyst (QA)",
        goal=f"Evaluate the quality of the report on {topic}",
        backstory=(
            "Specialist in quality control of market analyses. "
            "Evaluates reports on: completeness (are all the necessary data present?), "
            "accuracy (are the cited data consistent with the research?), "
            "clarity of recommendations (are they actionable and specific?). "
            "Issues a 0-10 score per criterion and an overall score, with justification."
        ),
        llm=llm, max_iter=3
    )

    t1 = Task(
        description=f"Research {topic}: size, players, growth, challenges.",
        expected_output="Quantitative data with sources. Top-5 players with market share.",
        agent=researcher
    )
    t2 = Task(
        description=f"Calculate CAGR and sector concentration metrics for {topic}.",
        expected_output="Calculated CAGR, market concentration, 2026 projection.",
        agent=analyst, context=[t1]
    )
    t3 = Task(
        description=(
            f"Write an executive report on {topic} for an investor. "
            "Include: executive summary, market context, quantitative analysis, recommendation."
        ),
        expected_output="Report of 500-700 words structured in 5 sections.",
        agent=writer, context=[t1, t2]
    )
    t4 = Task(
        description=(
            f"Evaluate the quality of the report on {topic} against these criteria: "
            "1) COMPLETENESS: are all the relevant data present? (0-10) "
            "2) ACCURACY: are the cited data consistent with the research? (0-10) "
            "3) CLARITY OF RECOMMENDATIONS: are they actionable and specific? (0-10) "
            "Calculate the overall score (average of the 3 criteria) and state: APPROVED (>= 7) or REVIEW (< 7)."
        ),
        expected_output=(
            "QA report with: score per criterion (0-10), "
            "overall score (average), verdict (APPROVED/REVIEW), "
            "and a list of needed improvements if score < 7."
        ),
        agent=qa, context=[t1, t2, t3]
    )

    crew = Crew(
        agents=[researcher, analyst, writer, qa],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential
    )
    result = crew.kickoff(inputs={"topic": topic})
    outputs = result.tasks_output
    return {
        "research": outputs[0].raw if len(outputs) > 0 else "",
        "analysis": outputs[1].raw if len(outputs) > 1 else "",
        "report": outputs[2].raw if len(outputs) > 2 else "",
        "qa": outputs[3].raw if len(outputs) > 3 else "",
        "final_output": result.raw
    }


if __name__ == "__main__":
    result = build_crew_with_qa("higher education edtechs in Brazil")
    print("=== EXECUTIVE REPORT ===")
    print(result["report"][:600])
    print("\n=== QUALITY EVALUATION (QA) ===")
    print(result["qa"])
