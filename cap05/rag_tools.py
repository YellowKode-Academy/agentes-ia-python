from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from pathlib import Path

load_dotenv()

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def criar_retriever_de_pdf(caminho_pdf: str, collection_name: str):
    """Cria retriever a partir de um PDF e persiste em ChromaDB."""
    loader = PyPDFLoader(caminho_pdf)
    docs = loader.load()
    chunks = _splitter.split_documents(docs)
    vs = Chroma.from_documents(
        chunks, embedding=_embeddings,
        persist_directory=f"./indices/{collection_name}"
    )
    return vs.as_retriever(search_kwargs={"k": 4})


def criar_ferramenta_rag(retriever, nome: str, descricao: str):
    """Encapsula um retriever como ferramenta para o agente."""
    @tool(name=nome)
    def consultar_documento(consulta: str) -> str:
        docs = retriever.invoke(consulta)
        if not docs:
            return f"Nenhuma informação relevante encontrada em {nome}."
        resultados = [f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs]
        return "\n\n".join(resultados)
    consultar_documento.__doc__ = descricao
    return consultar_documento


def carregar_e_indexar(caminhos: list[str], collection_name: str) -> Chroma:
    """Carrega documentos de múltiplos caminhos e cria vector store."""
    todos_docs = []
    for caminho in caminhos:
        path = Path(caminho)
        if path.suffix == ".pdf":
            loader = PyPDFLoader(caminho)
        elif path.suffix == ".csv":
            loader = CSVLoader(caminho, encoding="utf-8")
        else:
            loader = TextLoader(caminho, encoding="utf-8")
        todos_docs.extend(loader.load())
    chunks = _splitter.split_documents(todos_docs)
    return Chroma.from_documents(
        chunks, embedding=_embeddings,
        persist_directory=f"./indices/{collection_name}"
    )


def ferramenta_rag(nome: str, caminhos: list[str], descricao: str):
    """Cria uma ferramenta RAG a partir de lista de caminhos de documentos."""
    vs = carregar_e_indexar(caminhos, nome)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    @tool(name=f"consultar_{nome}")
    def consultar(consulta: str) -> str:
        docs = retriever.invoke(consulta)
        return "\n\n".join(
            f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs
        ) if docs else "Nenhuma informação relevante encontrada."
    consultar.__doc__ = descricao
    return consultar


def avaliar_rag(retriever, pares_qa: list[dict]) -> dict:
    """Avalia a precisão do retriever com pares pergunta-resposta conhecidos."""
    acertos = 0
    for par in pares_qa:
        docs = retriever.invoke(par["pergunta"])
        textos = "\n".join(d.page_content for d in docs)
        if par["resposta_esperada"].lower() in textos.lower():
            acertos += 1
        else:
            print(f"FALHA: '{par['pergunta']}'")
    return {
        "total": len(pares_qa),
        "acertos": acertos,
        "precisao": acertos / len(pares_qa) if pares_qa else 0
    }
