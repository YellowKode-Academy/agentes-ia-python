from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from pathlib import Path

load_dotenv()

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def create_retriever_from_pdf(pdf_path: str, collection_name: str):
    """Creates a retriever from a PDF and persists it in ChromaDB."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunks = _splitter.split_documents(docs)
    vs = Chroma.from_documents(
        chunks, embedding=_embeddings,
        persist_directory=f"./indexes/{collection_name}"
    )
    return vs.as_retriever(search_kwargs={"k": 4})


def create_rag_tool(retriever, name: str, description: str):
    """Wraps a retriever as a tool for the agent."""
    @tool(name=name)
    def query_document(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return f"No relevant information found in {name}."
        results = [f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs]
        return "\n\n".join(results)
    query_document.__doc__ = description
    return query_document


def load_and_index(paths: list[str], collection_name: str) -> Chroma:
    """Loads documents from multiple paths and creates a vector store."""
    all_docs = []
    for path_str in paths:
        path = Path(path_str)
        if path.suffix == ".pdf":
            loader = PyPDFLoader(path_str)
        elif path.suffix == ".csv":
            loader = CSVLoader(path_str, encoding="utf-8")
        else:
            loader = TextLoader(path_str, encoding="utf-8")
        all_docs.extend(loader.load())
    chunks = _splitter.split_documents(all_docs)
    return Chroma.from_documents(
        chunks, embedding=_embeddings,
        persist_directory=f"./indexes/{collection_name}"
    )


def rag_tool(name: str, paths: list[str], description: str):
    """Creates a RAG tool from a list of document paths."""
    vs = load_and_index(paths, name)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    @tool(name=f"query_{name}")
    def query_docs(query: str) -> str:
        docs = retriever.invoke(query)
        return "\n\n".join(
            f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs
        ) if docs else "No relevant information found."
    query_docs.__doc__ = description
    return query_docs


def evaluate_rag(retriever, qa_pairs: list[dict]) -> dict:
    """Evaluates the retriever's precision using known question-answer pairs."""
    hits = 0
    for pair in qa_pairs:
        docs = retriever.invoke(pair["question"])
        texts = "\n".join(d.page_content for d in docs)
        if pair["expected_answer"].lower() in texts.lower():
            hits += 1
        else:
            print(f"FAIL: '{pair['question']}'")
    return {
        "total": len(qa_pairs),
        "hits": hits,
        "precision": hits / len(qa_pairs) if qa_pairs else 0
    }
