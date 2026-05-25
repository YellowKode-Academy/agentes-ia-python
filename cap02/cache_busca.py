"""
Cache local para resultados da Tavily — evita buscas duplicadas e
permite desenvolvimento sem consumir o limite mensal da API.
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


class CacheBusca:
    """Cache em disco para resultados de busca com TTL configurável."""

    def __init__(self, cache_dir: str = "cache/busca", ttl_horas: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_horas)

    def _chave(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[list]:
        caminho = self.cache_dir / f"{self._chave(query)}.json"
        if not caminho.exists():
            return None
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        criado = datetime.fromisoformat(dados["criado_em"])
        if criado + self.ttl < datetime.now():
            caminho.unlink()
            return None
        return dados["resultado"]

    def set(self, query: str, resultado: list) -> None:
        caminho = self.cache_dir / f"{self._chave(query)}.json"
        caminho.write_text(
            json.dumps({"criado_em": datetime.now().isoformat(), "resultado": resultado},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def invalidar(self, query: str) -> bool:
        caminho = self.cache_dir / f"{self._chave(query)}.json"
        if caminho.exists():
            caminho.unlink()
            return True
        return False

    def limpar_expirados(self) -> int:
        removidos = 0
        for arq in self.cache_dir.glob("*.json"):
            try:
                dados = json.loads(arq.read_text(encoding="utf-8"))
                criado = datetime.fromisoformat(dados["criado_em"])
                if criado + self.ttl < datetime.now():
                    arq.unlink()
                    removidos += 1
            except (json.JSONDecodeError, KeyError):
                arq.unlink()
                removidos += 1
        return removidos


def criar_busca_com_cache(max_results: int = 3, ttl_horas: int = 24):
    """Cria função de busca Tavily com cache transparente."""
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily = TavilySearchResults(max_results=max_results)
    cache = CacheBusca(ttl_horas=ttl_horas)

    def buscar(query: str) -> list:
        cached = cache.get(query)
        if cached is not None:
            print(f"[cache] '{query[:50]}...' — servido do cache")
            return cached
        resultado = tavily.invoke(query)
        cache.set(query, resultado)
        return resultado

    return buscar


if __name__ == "__main__":
    buscar = criar_busca_com_cache()
    resultados = buscar("fintech pagamento Brasil 2025")
    for r in resultados[:2]:
        print(r.get("title", ""))
        print(r.get("content", "")[:150])
        print()
