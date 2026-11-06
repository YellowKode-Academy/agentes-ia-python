"""
Local cache for Tavily results. It avoids duplicate searches and
lets you develop without using up the API's monthly limit.
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


class SearchCache:
    """Disk cache for search results with a configurable TTL."""

    def __init__(self, cache_dir: str = "cache/search", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[list]:
        path = self.cache_dir / f"{self._key(query)}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        created = datetime.fromisoformat(data["created_at"])
        if created + self.ttl < datetime.now():
            path.unlink()
            return None
        return data["result"]

    def set(self, query: str, result: list) -> None:
        path = self.cache_dir / f"{self._key(query)}.json"
        path.write_text(
            json.dumps({"created_at": datetime.now().isoformat(), "result": result},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def invalidate(self, query: str) -> bool:
        path = self.cache_dir / f"{self._key(query)}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_expired(self) -> int:
        removed = 0
        for file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                created = datetime.fromisoformat(data["created_at"])
                if created + self.ttl < datetime.now():
                    file.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError):
                file.unlink()
                removed += 1
        return removed


def build_cached_search(max_results: int = 3, ttl_hours: int = 24):
    """Creates a Tavily search function with transparent caching."""
    from langchain_tavily import TavilySearch
    tavily = TavilySearch(max_results=max_results)
    cache = SearchCache(ttl_hours=ttl_hours)

    def search(query: str) -> list:
        cached = cache.get(query)
        if cached is not None:
            print(f"[cache] '{query[:50]}...' served from cache")
            return cached
        result = tavily.invoke(query)
        cache.set(query, result)
        return result

    return search


if __name__ == "__main__":
    search = build_cached_search()
    results = search("payments fintech Brazil 2025")
    for r in results[:2]:
        print(r.get("title", ""))
        print(r.get("content", "")[:150])
        print()
