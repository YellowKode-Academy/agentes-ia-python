"""
Capítulo 6 — Solução do exercício
Dashboard que lê logs/agente.log e exibe estatísticas de uso:
- Número total de execuções
- Média de latência por execução
- Ferramenta mais usada
- Custo total estimado do dia
"""
import re
from pathlib import Path
from datetime import datetime, date
from collections import Counter


LOG_FILE = Path("logs/agente.log")


def parse_log_linha(linha: str) -> dict | None:
    """Extrai campos estruturados de uma linha de log structlog."""
    if not linha.strip():
        return None
    try:
        campos = {}
        partes = linha.strip().split(" ")
        for parte in partes:
            if "=" in parte:
                k, v = parte.split("=", 1)
                campos[k] = v.strip("'\"")
        if "timestamp" not in campos and len(partes) > 1:
            campos["_raw"] = linha
        return campos if campos else None
    except Exception:
        return None


def ler_eventos_log(caminho: Path, data_filtro: date | None = None) -> list[dict]:
    """Lê o arquivo de log e retorna lista de eventos estruturados."""
    if not caminho.exists():
        return []
    eventos = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            evento = parse_log_linha(linha)
            if evento:
                if data_filtro:
                    ts = evento.get("timestamp", "")
                    if ts and not ts.startswith(str(data_filtro)):
                        continue
                eventos.append({"_raw": linha.strip(), **evento})
    return eventos


def calcular_metricas(eventos: list[dict]) -> dict:
    """Calcula métricas agregadas dos eventos de log."""
    runs_start = [e for e in eventos if "run_start" in e.get("_raw", "")]
    runs_end = [e for e in eventos if "run_end" in e.get("_raw", "")]
    runs_error = [e for e in eventos if "run_error" in e.get("_raw", "")]

    latencias = []
    custos = []
    ferramentas: list[str] = []

    for evento in runs_end:
        raw = evento.get("_raw", "")
        m_lat = re.search(r"latencia_s=([\d.]+)", raw)
        if m_lat:
            latencias.append(float(m_lat.group(1)))
        m_custo = re.search(r"custo_usd=([\d.]+)", raw)
        if m_custo:
            custos.append(float(m_custo.group(1)))
        m_tools = re.search(r"ferramentas_usadas=\[([^\]]*)\]", raw)
        if m_tools:
            tools_str = m_tools.group(1)
            tools_lista = [t.strip().strip("'\"") for t in tools_str.split(",") if t.strip()]
            ferramentas.extend(tools_lista)

    for evento in eventos:
        raw = evento.get("_raw", "")
        if "tool " in raw or "tool=" in raw:
            m_tool = re.search(r"nome='?([^'\s]+)'?", raw)
            if m_tool:
                ferramentas.append(m_tool.group(1))

    ferramenta_mais_usada = Counter(ferramentas).most_common(1)

    return {
        "total_execucoes": len(runs_start),
        "execucoes_concluidas": len(runs_end),
        "execucoes_com_erro": len(runs_error),
        "latencia_media_s": round(sum(latencias) / len(latencias), 2) if latencias else 0,
        "latencia_max_s": max(latencias) if latencias else 0,
        "custo_total_usd": round(sum(custos), 4),
        "ferramenta_mais_usada": ferramenta_mais_usada[0][0] if ferramenta_mais_usada else "N/A",
        "total_chamadas_ferramentas": len(ferramentas),
    }


def exibir_dashboard(caminho: Path = LOG_FILE, apenas_hoje: bool = True) -> None:
    """Exibe o dashboard de métricas no terminal."""
    data_filtro = date.today() if apenas_hoje else None
    periodo = f"hoje ({date.today()})" if apenas_hoje else "todo o histórico"

    print(f"\n{'='*50}")
    print(f"DASHBOARD DO AGENTE — {periodo.upper()}")
    print(f"Arquivo: {caminho}")
    print(f"{'='*50}")

    if not caminho.exists():
        print(f"\nArquivo de log não encontrado: {caminho}")
        print("Execute o agente com AgenteObservavel para gerar logs.")
        return

    eventos = ler_eventos_log(caminho, data_filtro)
    if not eventos:
        print(f"\nNenhum evento registrado para {periodo}.")
        return

    m = calcular_metricas(eventos)

    print(f"\n📊 Execuções: {m['total_execucoes']} iniciadas, {m['execucoes_concluidas']} concluídas, {m['execucoes_com_erro']} com erro")
    print(f"⏱  Latência média: {m['latencia_media_s']}s | Máxima: {m['latencia_max_s']}s")
    print(f"🔧 Ferramenta mais usada: {m['ferramenta_mais_usada']} ({m['total_chamadas_ferramentas']} chamadas total)")
    print(f"💰 Custo estimado {periodo}: ${m['custo_total_usd']:.4f} USD")

    if m['custo_total_usd'] > 5.0:
        print(f"\n⚠️  ALERTA: custo acumulado alto (${m['custo_total_usd']:.4f})")

    print()


if __name__ == "__main__":
    exibir_dashboard()
