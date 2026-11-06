"""
Chapter 5: Exercise solution
A RAG agent that answers an equity analyst's 10 standard questions
from an annual report loaded as a PDF.

Usage:
    python -m ch05.ir_analyst_solution data/annual_report.pdf TechVentures
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

from ch05.rag_tools import create_retriever_from_pdf, create_rag_tool

load_dotenv()

ANALYST_QUESTIONS = [
    "What was the net revenue reported in the last fiscal year? What was the growth compared to the previous year?",
    "What is the company's EBITDA margin? How has it evolved over the past years?",
    "What are the main market risks the company identifies in the report?",
    "What is the company's stated geographic expansion strategy?",
    "What is the company's currency exposure? How is it managed?",
    "What are the main products/services and what share of revenue does each represent?",
    "What does the capital structure look like? What is the level of indebtedness?",
    "Which main competitors are mentioned and how does the company position itself against them?",
    "Did the company pay dividends? What is its profit distribution policy?",
    "What are the outlook and guidance for the next fiscal year?",
]


def build_analyst_agent(pdf_path: str, company_name: str):
    """Creates an agent with a RAG tool for the annual report."""
    retriever = create_retriever_from_pdf(pdf_path, company_name.lower().replace(" ", "_"))
    document_tool = create_rag_tool(
        retriever=retriever,
        name=f"query_{company_name.lower().replace(' ', '_')}",
        description=(
            f"Queries the official annual report of {company_name}. "
            f"Use for questions about: revenue, margins, risks, strategy, "
            f"currency exposure, dividends, competitors and guidance of {company_name}."
        )
    )
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [document_tool, TavilySearch(max_results=2)]
    return create_agent(llm, tools)


def run_equity_analysis(pdf_path: str, company_name: str) -> None:
    """Runs the 10 standard questions and displays the report."""
    print(f"\n{'='*60}")
    print(f"EQUITY ANALYSIS: {company_name.upper()}")
    print(f"Source: {pdf_path}")
    print(f"{'='*60}\n")

    if not Path(pdf_path).exists():
        print(f"WARNING: file '{pdf_path}' not found.")
        print("Create the 'data/' folder and add the annual report PDF.")
        print("\nTip: annual reports of Brazilian companies are available")
        print("at: ri.[company].com.br or in the IR section of the company's website.\n")
        return

    agent = build_analyst_agent(pdf_path, company_name)

    for i, question in enumerate(ANALYST_QUESTIONS, 1):
        print(f"[{i:02d}/10] {question}")
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            response = result["messages"][-1].content
            print(f"Answer: {response[:400]}{'...' if len(response) > 400 else ''}")
        except Exception as e:
            print(f"Error processing question: {e}")
        print()


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pdf = sys.argv[1]
        company = sys.argv[2]
    else:
        pdf = "data/annual_report.pdf"
        company = "Company"
    run_equity_analysis(pdf, company)
