import logging
import logging.handlers
import structlog
from datetime import datetime
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler

log = structlog.get_logger()


class AgenteObservavel(BaseCallbackHandler):
    """Callback unificado: logs, métricas e alertas de custo."""

    CUSTO_INPUT = 0.003 / 1000   # USD por token — Claude Sonnet
    CUSTO_OUTPUT = 0.015 / 1000  # USD por token — Claude Sonnet

    def __init__(self, nome_run: str = "agente", limite_custo_usd: float = 0.50):
        self.nome = nome_run
        self.limite = limite_custo_usd
        self.inicio = None
        self.tokens_input = 0
        self.tokens_output = 0
        self.tool_calls: list[str] = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.inicio = datetime.now()
        log.info("run_start", agente=self.nome, inputs_len=len(str(inputs)))

    def on_tool_start(self, serialized, input_str, **kwargs):
        nome = serialized.get("name", "unknown")
        self.tool_calls.append(nome)
        log.info("tool", nome=nome, input=str(input_str)[:100])

    def on_tool_end(self, output, **kwargs):
        log.info("tool_end", output_len=len(str(output)))

    def on_llm_end(self, response, **kwargs):
        usage = getattr(response.llm_output, "token_usage", {}) or {}
        self.tokens_input += usage.get("prompt_tokens", 0)
        self.tokens_output += usage.get("completion_tokens", 0)
        custo_atual = (self.tokens_input * self.CUSTO_INPUT +
                       self.tokens_output * self.CUSTO_OUTPUT)
        if custo_atual > self.limite:
            print(f"ALERTA: custo acumulado ${custo_atual:.4f} excedeu ${self.limite:.2f}")

    def on_chain_end(self, outputs, **kwargs):
        latencia = (datetime.now() - self.inicio).total_seconds() if self.inicio else 0
        custo = self.tokens_input * self.CUSTO_INPUT + self.tokens_output * self.CUSTO_OUTPUT
        log.info(
            "run_end",
            agente=self.nome,
            latencia_s=round(latencia, 2),
            tokens=self.tokens_input + self.tokens_output,
            custo_usd=round(custo, 4),
            ferramentas_usadas=self.tool_calls
        )

    def on_chain_error(self, error, **kwargs):
        log.error("run_error", agente=self.nome, erro=str(error))

    def on_agent_action(self, action, **kwargs):
        log.info("agent_action", tool=action.tool, input=str(action.tool_input)[:100])

    @property
    def custo_total_usd(self) -> float:
        return self.tokens_input * self.CUSTO_INPUT + self.tokens_output * self.CUSTO_OUTPUT


def criar_config_observavel(thread_id: str, nome_run: str = "agente") -> dict:
    """Cria config com callback de observabilidade para uso em agent.invoke."""
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": [AgenteObservavel(nome_run)]
    }


def configurar_logging(log_dir: str = "logs") -> None:
    """Configura logging em arquivo com rotação diária."""
    Path(log_dir).mkdir(exist_ok=True)
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=f"{log_dir}/agente.log",
        when="midnight",
        backupCount=30,
        encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
