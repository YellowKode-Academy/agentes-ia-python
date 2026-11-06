"""
Chapter 4: Exercise solution
Reads all .txt and .json files in data/, ingests them into the vector store with metadata,
runs 5 test queries and prints the total number of chunks stored.
"""
import json
from pathlib import Path
from dotenv import load_dotenv
from ch04.memory_store import ingest_file, ingest_text, search_with_score

load_dotenv()

DATA_DIR = Path("data")

TEST_QUERIES = [
    "company growth and expansion",
    "main competitors and market share",
    "market risks and challenges",
    "team size and number of employees",
    "product and competitive differentiators",
]


def ingest_structured_json(path: Path) -> int:
    """Ingests a JSON file, converting it to readable text."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    company = data.get("company", path.stem)
    return ingest_text(text, {
        "company": company,
        "type": "raw_data",
        "source": str(path),
        "file": path.name
    })


if __name__ == "__main__":
    total_chunks = 0

    print("=== DATA INGESTION ===\n")
    if not DATA_DIR.exists():
        print(f"Folder '{DATA_DIR}' not found. Create the folder and add files.")
    else:
        for file in DATA_DIR.glob("*.json"):
            n = ingest_structured_json(file)
            total_chunks += n
            print(f"✓ {file.name} - {n} chunks")

        for file in DATA_DIR.glob("*.txt"):
            n = ingest_file(str(file), {"type": "text", "source": file.name})
            total_chunks += n
            print(f"✓ {file.name} - {n} chunks")

    print(f"\nTotal chunks stored: {total_chunks}")

    print("\n=== SEARCH TESTS (with scores) ===\n")
    for query in TEST_QUERIES:
        print(f"Query: '{query}'")
        results = search_with_score(query, k=2)
        for doc, score in results:
            print(f"  Score: {score:.3f} | {doc.page_content[:120]}...")
        print()
