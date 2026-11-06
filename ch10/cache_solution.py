"""
Chapter 10: Exercise solution
Disk cache for Tavily results with a 24-hour TTL.
Reduces API calls and lets you keep operating when Tavily is unavailable.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path("cache")
TTL_HOURS = 24


def _key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def _path(query: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"tavily_{_key(query)}.json"


def get_cache(query: str) -> Optional[list[dict]]:
    """Returns the cached result if it exists and has not expired."""
    file = _path(query)
    if not file.exists():
        return None
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        created = datetime.fromisoformat(data["created_at"])
        if created + timedelta(hours=TTL_HOURS) < datetime.now():
            file.unlink()
            return None
        print(f"[CACHE HIT] '{query[:50]}' (created: {data['created_at'][:16]})")
        return data["result"]
    except (json.JSONDecodeError, KeyError):
        return None


def set_cache(query: str, result: list[dict]) -> None:
    """Saves a result to the cache."""
    file = _path(query)
    file.write_text(
        json.dumps(
            {"created_at": datetime.now().isoformat(), "query": query, "result": result},
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )


def _extract_list(result) -> list[dict]:
    """Normalizes Tavily's return value (dict or list) to list[dict]."""
    if isinstance(result, dict):
        return result.get("results", [])
    elif isinstance(result, list):
        return result
    return []


def search_with_cache(query: str, max_results: int = 5) -> list[dict]:
    """Searches Tavily with a disk cache. Uses the cache if available."""
    cached = get_cache(query)
    if cached is not None:
        return _extract_list(cached) if isinstance(cached, dict) else cached

    from langchain_tavily import TavilySearch
    search = TavilySearch(max_results=max_results)
    try:
        result = search.invoke(query)
        set_cache(query, result)
        return _extract_list(result)
    except Exception as e:
        print(f"[TAVILY ERROR] {e}. Trying the cache even if expired...")
        file = _path(query)
        if file.exists():
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                print(f"[CACHE STALE] Using old result from {data['created_at'][:16]}")
                return data["result"]
            except Exception:
                pass
        return []


def clear_expired_cache() -> int:
    """Removes expired cache entries. Returns the number removed."""
    CACHE_DIR.mkdir(exist_ok=True)
    removed = 0
    for file in CACHE_DIR.glob("tavily_*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(data["created_at"])
            if created + timedelta(hours=TTL_HOURS) < datetime.now():
                file.unlink()
                removed += 1
        except Exception:
            pass
    return removed


def cache_status() -> dict:
    """Returns cache statistics."""
    CACHE_DIR.mkdir(exist_ok=True)
    files = list(CACHE_DIR.glob("tavily_*.json"))
    expired = 0
    valid = 0
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(data["created_at"])
            if created + timedelta(hours=TTL_HOURS) < datetime.now():
                expired += 1
            else:
                valid += 1
        except Exception:
            expired += 1
    return {"total": len(files), "valid": valid, "expired": expired}


if __name__ == "__main__":
    print("=== CACHE TEST ===\n")
    test_queries = [
        "payments fintech brazil 2025 market",
        "higher education edtech brazil growth",
        "payments fintech brazil 2025 market",  # Repeated: should use the cache
    ]
    for q in test_queries:
        print(f"Query: '{q}'")
        results = search_with_cache(q, max_results=3)
        print(f"Results: {len(results)} items\n")

    print(f"\nCache status: {cache_status()}")
    removed = clear_expired_cache()
    print(f"Expired entries removed: {removed}")
