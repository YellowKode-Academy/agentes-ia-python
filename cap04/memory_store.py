from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from datetime import datetime

load_dotenv()

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_vectorstore = Chroma(
    embedding_function=_embeddings,
    persist_directory="./memoria_longa",
    collection_name="inteligencia_mercado"
)
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)


def ingerir_texto(texto: str, metadados: dict) -> int:
    """Ingere um texto no vector store com metadados fornecidos. Retorna chunks criados."""
    chunks = _splitter.split_text(texto)
    if not chunks:
        return 0
    _vectorstore.add_texts(chunks, metadatas=[metadados] * len(chunks))
    return len(chunks)


def ingerir_arquivo(caminho: str, metadados: dict) -> int:
    """Lê um arquivo de texto e ingere no vector store."""
    with open(caminho, "r", encoding="utf-8") as f:
        texto = f.read()
    return ingerir_texto(texto, {**metadados, "arquivo": caminho})


def ingerir_pdf(caminho_pdf: str, metadados: dict) -> int:
    """Lê um PDF e ingere cada página como documento separado."""
    from pypdf import PdfReader
    reader = PdfReader(caminho_pdf)
    total = 0
    for i, pagina in enumerate(reader.pages):
        texto = pagina.extract_text()
        if not texto or not texto.strip():
            continue
        meta = {**metadados, "pagina": i + 1, "arquivo": caminho_pdf}
        total += ingerir_texto(texto, meta)
    return total


@tool
def buscar_memoria(consulta: str) -> str:
    """Busca no histórico de análises e briefings anteriores.
    Use quando o usuário mencionar empresas ou análises passadas,
    ou quando precisar de contexto de sessões anteriores."""
    docs = _vectorstore.similarity_search(consulta, k=3)
    if not docs:
        return "Nada encontrado na memória sobre esse tema."
    return "\n\n".join(
        f"[{d.metadata.get('tipo', 'nota')} - {d.metadata.get('empresa', 'geral')} "
        f"({d.metadata.get('data', 'sem data')})]\n{d.page_content}"
        for d in docs
    )


@tool
def salvar_na_memoria(conteudo: str, empresa: str, tipo: str = "analise") -> str:
    """Salva resultado ou descoberta na memória de longo prazo para uso futuro.
    tipo pode ser: analise, briefing, relatorio, nota.
    Use ao final de análises para que o conteúdo possa ser recuperado depois."""
    n = ingerir_texto(conteudo, {
        "empresa": empresa,
        "tipo": tipo,
        "data": datetime.now().strftime("%Y-%m-%d"),
        "fonte": "agente"
    })
    return f"Salvo na memória: {n} chunks sobre '{empresa}' (tipo: {tipo})"


def buscar_com_filtro(consulta: str, filtros: dict, k: int = 5) -> list:
    """Busca semântica com filtros de metadados."""
    return _vectorstore.similarity_search(consulta, k=k, filter=filtros)


def buscar_com_score(consulta: str, k: int = 3) -> list:
    """Busca com scores de similaridade para avaliar relevância."""
    return _vectorstore.similarity_search_with_score(consulta, k=k)


if __name__ == "__main__":
    print("Ingerindo dados de exemplo...")
    n = ingerir_texto(
        "TechVentures é uma empresa de SaaS B2B com 200 funcionários e ARR de R$18M.",
        {"empresa": "TechVentures", "tipo": "briefing", "fonte": "manual"}
    )
    print(f"Ingeridos {n} chunks.")

    print("\nBuscando na memória...")
    resultado = buscar_memoria.invoke("TechVentures tamanho equipe")
    print(resultado)
