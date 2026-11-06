import logging
import logging.handlers
import structlog
from datetime import datetime
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler

log = structlog.get_logger()


class ObservableAgent(BaseCallbackHandler):
    """Unified callback: logs, metrics and cost alerts."""

    INPUT_COST = 0.003 / 1000   # USD per token, Claude Sonnet
    OUTPUT_COST = 0.015 / 1000  # USD per token, Claude Sonnet

    def __init__(self, run_name: str = "agent", cost_limit_usd: float = 0.50):
        self.name = run_name
        self.limit = cost_limit_usd
        self.start = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls: list[str] = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.start = datetime.now()
        log.info("run_start", agent=self.name, inputs_len=len(str(inputs)))

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown")
        self.tool_calls.append(name)
        log.info("tool", name=name, input=str(input_str)[:100])

    def on_tool_end(self, output, **kwargs):
        log.info("tool_end", output_len=len(str(output)))

    def on_llm_end(self, response, **kwargs):
        usage = getattr(response.llm_output, "token_usage", {}) or {}
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        current_cost = (self.input_tokens * self.INPUT_COST +
                        self.output_tokens * self.OUTPUT_COST)
        if current_cost > self.limit:
            print(f"ALERT: accumulated cost ${current_cost:.4f} exceeded ${self.limit:.2f}")

    def on_chain_end(self, outputs, **kwargs):
        latency = (datetime.now() - self.start).total_seconds() if self.start else 0
        cost = self.input_tokens * self.INPUT_COST + self.output_tokens * self.OUTPUT_COST
        log.info(
            "run_end",
            agent=self.name,
            latency_s=round(latency, 2),
            tokens=self.input_tokens + self.output_tokens,
            cost_usd=round(cost, 4),
            tools_used=self.tool_calls
        )

    def on_chain_error(self, error, **kwargs):
        log.error("run_error", agent=self.name, error=str(error))

    def on_agent_action(self, action, **kwargs):
        log.info("agent_action", tool=action.tool, input=str(action.tool_input)[:100])

    @property
    def total_cost_usd(self) -> float:
        return self.input_tokens * self.INPUT_COST + self.output_tokens * self.OUTPUT_COST


def build_observable_config(thread_id: str, run_name: str = "agent") -> dict:
    """Creates a config with the observability callback for use in agent.invoke."""
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": [ObservableAgent(run_name)]
    }


def configure_logging(log_dir: str = "logs") -> None:
    """Configures file logging with daily rotation."""
    Path(log_dir).mkdir(exist_ok=True)
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=f"{log_dir}/agent.log",
        when="midnight",
        backupCount=30,
        encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
