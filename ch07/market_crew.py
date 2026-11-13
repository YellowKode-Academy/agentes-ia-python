from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

llm = LLM(model="anthropic/claude-sonnet-4-6", max_tokens=4096)
tavily = TavilySearch(max_results=5)

@tool("Web search")
def search(query: str) -> str:
    """Searches the web and returns the most relevant results."""
    return str(tavily.invoke({"query": query}))


def build_market_crew(topic: str) -> str:
    """Runs a full market analysis for the given topic."""
    researcher = Agent(
        role="Market Researcher",
        goal=f"Collect accurate data on {topic} in Brazil",
        backstory=(
            "Brazilian market research specialist focused on verifiable data. "
            "You never make up data: if you can't find it, you say you didn't find it."
        ),
        tools=[search],
        llm=llm,
        max_iter=6
    )
    analyst = Agent(
        role="Business Analyst",
        goal=f"Turn research on {topic} into an executive analysis",
        backstory=(
            "Senior analyst with experience in business strategy. "
            "Turns data into clear analyses with direct conclusions. "
            "Your report always ends with 3 numbered practical recommendations."
        ),
        llm=llm,
        max_iter=4
    )
    t_research = Task(
        description=(
            f"Research the {topic} market in Brazil. "
            "Collect: market size (in R$), main players with estimated market share, "
            "annual growth rate, 2025 trends and the sector's main challenges."
        ),
        expected_output=(
            "Structured data with sources. Format: bullet points with quantitative data. "
            f"Include: {topic} market size, list of top-5 players, "
            "3 identified trends and 2 main challenges."
        ),
        agent=researcher
    )
    t_analysis = Task(
        description=(
            f"Based on the research on {topic}, create an executive report for a Brazilian investor. "
            "Include executive summary, opportunity analysis, risk analysis and recommendations."
        ),
        expected_output=(
            "Executive report of 500-700 words with: "
            "executive summary (3 lines), opportunity analysis, "
            "risk analysis and 3 numbered practical recommendations."
        ),
        agent=analyst,
        context=[t_research]
    )
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[t_research, t_analysis],
        process=Process.sequential
    )
    return crew.kickoff(inputs={"topic": topic}).raw


if __name__ == "__main__":
    result = build_market_crew("payments fintechs")
    print(result)
