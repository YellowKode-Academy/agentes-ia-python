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


def com_retry(
    max_tentativas: int = 3,
    delay_base: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Decorator de retry com backoff exponencial e jitter."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for tentativa in range(max_tentativas):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if tentativa == max_tentativas - 1:
                        raise
                    delay = delay_base * (2 ** tentativa) + random.uniform(0, 0.5)
                    print(f"Erro ({type(e).__name__}). Tentativa {tentativa + 1}/{max_tentativas}. "
                          f"Aguardando {delay:.1f}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def criar_llm_resiliente():
    """Cria LLM com fallback automático Claude Sonnet → Groq Llama."""
    from langchain_anthropic import ChatAnthropic
    try:
        from langchain_groq import ChatGroq
        groq = ChatGroq(model="llama-3.3-70b-versatile", max_retries=2, timeout=30)
        return ChatAnthropic(model="claude-sonnet-4-6", max_retries=2, timeout=30).with_fallbacks([groq])
    except ImportError:
        return ChatAnthropic(model="claude-sonnet-4-6", max_retries=3, timeout=60)


class CachePesquisa:
    """Cache em disco com TTL para resultados de busca."""

    def __init__(self, cache_dir: str = "cache", ttl_horas: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_horas)

    def _chave(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        caminho = self.cache_dir / f"{self._chave(query)}.json"
        if not caminho.exists():
            return None
        try:
            dados = json.loads(caminho.read_text(encoding="utf-8"))
            criado = datetime.fromisoformat(dados["criado_em"])
            if criado + self.ttl < datetime.now():
                caminho.unlink()
                return None
            return dados["resultado"]
        except (json.JSONDecodeError, KeyError):
            return None

    def set(self, query: str, resultado: str) -> None:
        caminho = self.cache_dir / f"{self._chave(query)}.json"
        caminho.write_text(
            json.dumps({"criado_em": datetime.now().isoformat(), "resultado": resultado},
                       ensure_ascii=False),
            encoding="utf-8"
        )

    def limpar_expirados(self) -> int:
        """Remove entradas expiradas. Retorna número de arquivos removidos."""
        removidos = 0
        for arquivo in self.cache_dir.glob("*.json"):
            try:
                dados = json.loads(arquivo.read_text(encoding="utf-8"))
                criado = datetime.fromisoformat(dados["criado_em"])
                if criado + self.ttl < datetime.now():
                    arquivo.unlink()
                    removidos += 1
            except Exception:
                pass
        return removidos


class CircuitBreaker:
    """Interrompe chamadas para uma API que está falhando repetidamente."""

    def __init__(self, limite_falhas: int = 5, tempo_espera_s: int = 60):
        self.limite = limite_falhas
        self.tempo_espera = timedelta(seconds=tempo_espera_s)
        self.falhas = 0
        self.aberto = False
        self.ultima_falha: Optional[datetime] = None

    def pode_chamar(self) -> bool:
        if not self.aberto:
            return True
        if self.ultima_falha and datetime.now() - self.ultima_falha > self.tempo_espera:
            self.aberto = False
            self.falhas = 0
            return True
        return False

    def registrar_sucesso(self):
        self.falhas = 0
        self.aberto = False

    def registrar_falha(self):
        self.falhas += 1
        self.ultima_falha = datetime.now()
        if self.falhas >= self.limite:
            self.aberto = True
            print(f"Circuit breaker ABERTO após {self.falhas} falhas. "
                  f"Aguardando {self.tempo_espera.seconds}s.")

    @property
    def estado(self) -> str:
        if not self.aberto:
            return "FECHADO"
        if self.ultima_falha and datetime.now() - self.ultima_falha > self.tempo_espera:
            return "SEMI_ABERTO"
        return "ABERTO"


class NivelDegradacao(Enum):
    COMPLETO = "completo"
    SEM_BUSCA = "sem_busca"
    MODELO_FALLBACK = "fallback"
    MINIMO = "minimo"


def criar_agente_degradavel():
    """
    Tenta criar o agente completo; degrada graciosamente se componentes falharem.
    Retorna (agente, nivel_degradacao).
    """
    from langchain_anthropic import ChatAnthropic
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub
    from langgraph.checkpoint.memory import MemorySaver

    ferramentas = []
    nivel = NivelDegradacao.COMPLETO

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        busca = TavilySearchResults(max_results=3)
        busca.invoke("teste")
        ferramentas.append(busca)
    except Exception as e:
        print(f"Tavily indisponível ({type(e).__name__}). Continuando sem busca.")
        nivel = NivelDegradacao.SEM_BUSCA

    try:
        llm = ChatAnthropic(model="claude-sonnet-4-6", max_retries=1, timeout=10)
        llm.invoke([{"role": "user", "content": "ok"}])
    except Exception:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model="llama-3.3-70b-versatile")
            nivel = NivelDegradacao.MODELO_FALLBACK
        except Exception:
            raise RuntimeError("Nenhum modelo disponível. Sistema não pode operar.")

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, ferramentas, prompt)
    executor = AgentExecutor(
        agent=agent, tools=ferramentas,
        handle_parsing_errors=True, max_iterations=5
    )
    return executor, nivel
