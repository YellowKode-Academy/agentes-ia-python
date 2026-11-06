import time
import random
import hashlib
import json
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import TypeVar, Callable, Optional

T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff and jitter."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    print(f"Error ({type(e).__name__}). Attempt {attempt + 1}/{max_attempts}. "
                          f"Waiting {delay:.1f}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def create_resilient_llm():
    """Creates an LLM with automatic fallback from Claude Sonnet to Groq Llama."""
    from langchain_anthropic import ChatAnthropic
    try:
        from langchain_groq import ChatGroq
        groq = ChatGroq(model="llama-3.3-70b-versatile", max_retries=2, timeout=30)
        return ChatAnthropic(model="claude-sonnet-4-6", max_retries=2, timeout=30).with_fallbacks([groq])
    except ImportError:
        return ChatAnthropic(model="claude-sonnet-4-6", max_retries=3, timeout=60)


class ResearchCache:
    """Disk cache with TTL for search results."""

    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        path = self.cache_dir / f"{self._key(query)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(data["created_at"])
            if created + self.ttl < datetime.now():
                path.unlink()
                return None
            return data["result"]
        except (json.JSONDecodeError, KeyError):
            return None

    def set(self, query: str, result: str) -> None:
        path = self.cache_dir / f"{self._key(query)}.json"
        path.write_text(
            json.dumps({"created_at": datetime.now().isoformat(), "result": result},
                       ensure_ascii=False),
            encoding="utf-8"
        )

    def clear_expired(self) -> int:
        """Removes expired entries. Returns the number of files removed."""
        removed = 0
        for file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                created = datetime.fromisoformat(data["created_at"])
                if created + self.ttl < datetime.now():
                    file.unlink()
                    removed += 1
            except Exception:
                pass
        return removed


class CircuitBreaker:
    """Stops calls to an API that keeps failing."""

    def __init__(self, failure_threshold: int = 5, wait_time_s: int = 60):
        self.threshold = failure_threshold
        self.wait_time = timedelta(seconds=wait_time_s)
        self.failures = 0
        self.open = False
        self.last_failure: Optional[datetime] = None

    def can_call(self) -> bool:
        if not self.open:
            return True
        if self.last_failure and datetime.now() - self.last_failure > self.wait_time:
            self.open = False
            self.failures = 0
            return True
        return False

    def record_success(self):
        self.failures = 0
        self.open = False

    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.now()
        if self.failures >= self.threshold:
            self.open = True
            print(f"Circuit breaker OPEN after {self.failures} failures. "
                  f"Waiting {self.wait_time.seconds}s.")

    @property
    def state(self) -> str:
        if not self.open:
            return "CLOSED"
        if self.last_failure and datetime.now() - self.last_failure > self.wait_time:
            return "HALF_OPEN"
        return "OPEN"


class DegradationLevel(Enum):
    FULL = "full"
    NO_SEARCH = "no_search"
    FALLBACK_MODEL = "fallback"
    MINIMAL = "minimal"


def build_degradable_agent():
    """
    Tries to create the full agent; degrades gracefully if components fail.
    Returns (agent, degradation_level).
    """
    from langchain_anthropic import ChatAnthropic
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import MemorySaver

    tools = []
    level = DegradationLevel.FULL

    try:
        from langchain_tavily import TavilySearch
        search = TavilySearch(max_results=3)
        search.invoke("test")
        tools.append(search)
    except Exception as e:
        print(f"Tavily unavailable ({type(e).__name__}). Continuing without search.")
        level = DegradationLevel.NO_SEARCH

    try:
        llm = ChatAnthropic(model="claude-sonnet-4-6", max_retries=1, timeout=10)
        llm.invoke([{"role": "user", "content": "ok"}])
    except Exception:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model="llama-3.3-70b-versatile")
            level = DegradationLevel.FALLBACK_MODEL
        except Exception:
            raise RuntimeError("No model available. The system cannot operate.")

    agent = create_agent(llm, tools, checkpointer=MemorySaver())
    return agent, level
