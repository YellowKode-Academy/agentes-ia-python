from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from datetime import datetime

load_dotenv()

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_vector_store = Chroma(
    embedding_function=_embeddings,
    persist_directory="./long_term_memory",
    collection_name="market_intelligence"
)
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)


def ingest_text(text: str, metadata: dict) -> int:
    """Ingests a text into the vector store with the given metadata. Returns the chunks created."""
    chunks = _splitter.split_text(text)
    if not chunks:
        return 0
    _vector_store.add_texts(chunks, metadatas=[metadata] * len(chunks))
    return len(chunks)


def ingest_file(path: str, metadata: dict) -> int:
    """Reads a text file and ingests it into the vector store."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return ingest_text(text, {**metadata, "file": path})


def ingest_pdf(pdf_path: str, metadata: dict) -> int:
    """Reads a PDF and ingests each page as a separate document."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    total = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        meta = {**metadata, "page": i + 1, "file": pdf_path}
        total += ingest_text(text, meta)
    return total


@tool
def search_memory(query: str) -> str:
    """Searches the history of past analyses and briefings.
    Use when the user mentions past companies or analyses,
    or when you need context from previous sessions."""
    docs = _vector_store.similarity_search(query, k=3)
    if not docs:
        return "Nothing found in memory on this topic."
    return "\n\n".join(
        f"[{d.metadata.get('type', 'note')} - {d.metadata.get('company', 'general')} "
        f"({d.metadata.get('date', 'no date')})]\n{d.page_content}"
        for d in docs
    )


@tool
def save_to_memory(content: str, company: str, kind: str = "analysis") -> str:
    """Saves a result or finding to long-term memory for future use.
    kind can be: analysis, briefing, report, note.
    Use at the end of analyses so the content can be retrieved later."""
    n = ingest_text(content, {
        "company": company,
        "type": kind,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": "agent"
    })
    return f"Saved to memory: {n} chunks about '{company}' (type: {kind})"


def search_with_filter(query: str, filters: dict, k: int = 5) -> list:
    """Semantic search with metadata filters."""
    return _vector_store.similarity_search(query, k=k, filter=filters)


def search_with_score(query: str, k: int = 3) -> list:
    """Search with similarity scores to evaluate relevance."""
    return _vector_store.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    print("Ingesting sample data...")
    n = ingest_text(
        "TechVentures is a B2B SaaS company with 200 employees and ARR of R$18M.",
        {"company": "TechVentures", "type": "briefing", "source": "manual"}
    )
    print(f"Ingested {n} chunks.")

    print("\nSearching memory...")
    result = search_memory.invoke("TechVentures team size")
    print(result)
