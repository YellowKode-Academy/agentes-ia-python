"""
Capítulo 10 — Solução do exercício
Cache em disco para resultados da Tavily com TTL de 24 horas.
Reduz chamadas à API e permite operar quando a Tavily está indisponível.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path("cache")
TTL_HORAS = 24


def _chave(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def _caminho(query: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"tavily_{_chave(query)}.json"


def get_cache(query: str) -> Optional[list[dict]]:
    """Retorna resultado em cache se existir e não estiver expirado."""
    arq = _caminho(query)
    if not arq.exists():
        return None
    try:
        dados = json.loads(arq.read_text(encoding="utf-8"))
        criado = datetime.fromisoformat(dados["criado_em"])
        if criado + timedelta(hours=TTL_HORAS) < datetime.now():
            arq.unlink()
            return None
        print(f"[CACHE HIT] '{query[:50]}' (criado: {dados['criado_em'][:16]})")
        return dados["resultado"]
    except (json.JSONDecodeError, KeyError):
        return None


def set_cache(query: str, resultado: list[dict]) -> None:
    """Salva resultado no cache."""
    arq = _caminho(query)
    arq.write_text(
        json.dumps(
            {"criado_em": datetime.now().isoformat(), "query": query, "resultado": resultado},
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )


def buscar_com_cache(query: str, max_results: int = 5) -> list[dict]:
    """Busca na Tavily com cache em disco. Usa cache se disponível."""
    cached = get_cache(query)
    if cached is not None:
        return cached

    from langchain_tavily import TavilySearch
    search = TavilySearch(max_results=max_results)
    try:
        resultado = search.invoke(query)
        set_cache(query, resultado)
        return resultado
    except Exception as e:
        print(f"[TAVILY ERROR] {e}. Tentando cache mesmo expirado...")
        arq = _caminho(query)
        if arq.exists():
            try:
                dados = json.loads(arq.read_text(encoding="utf-8"))
                print(f"[CACHE STALE] Usando resultado antigo de {dados['criado_em'][:16]}")
                return dados["resultado"]
            except Exception:
                pass
        return []


def limpar_cache_expirado() -> int:
    """Remove entradas de cache expiradas. Retorna número removido."""
    CACHE_DIR.mkdir(exist_ok=True)
    removidos = 0
    for arq in CACHE_DIR.glob("tavily_*.json"):
        try:
            dados = json.loads(arq.read_text(encoding="utf-8"))
            criado = datetime.fromisoformat(dados["criado_em"])
            if criado + timedelta(hours=TTL_HORAS) < datetime.now():
                arq.unlink()
                removidos += 1
        except Exception:
            pass
    return removidos


def status_cache() -> dict:
    """Retorna estatísticas do cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    arquivos = list(CACHE_DIR.glob("tavily_*.json"))
    expirados = 0
    validos = 0
    for arq in arquivos:
        try:
            dados = json.loads(arq.read_text(encoding="utf-8"))
            criado = datetime.fromisoformat(dados["criado_em"])
            if criado + timedelta(hours=TTL_HORAS) < datetime.now():
                expirados += 1
            else:
                validos += 1
        except Exception:
            expirados += 1
    return {"total": len(arquivos), "validos": validos, "expirados": expirados}


if __name__ == "__main__":
    print("=== TESTE DO CACHE ===\n")
    queries_teste = [
        "fintech pagamento brasil 2025 mercado",
        "edtech ensino superior brasil crescimento",
        "fintech pagamento brasil 2025 mercado",  # Repetida — deve usar cache
    ]
    for q in queries_teste:
        print(f"Query: '{q}'")
        resultados = buscar_com_cache(q, max_results=3)
        print(f"Resultados: {len(resultados)} itens\n")

    print(f"\nStatus do cache: {status_cache()}")
    removidos = limpar_cache_expirado()
    print(f"Entradas expiradas removidas: {removidos}")
