"""
Capítulo 12 — Sistema Completo: Agente de Inteligência de Mercado
Integra: LangGraph (orquestração) + CrewAI (multi-agente) + ChromaDB (memória longa)
         + Pydantic (validação) + FastAPI (API REST)
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")


class EstadoSistema(TypedDict):
    tema: str
    contexto_anterior: str
    relatorio_bruto: str
    relatorio_validado: Optional[dict]
    tentativas_validacao: int
    status: str


def verificar_contexto(e: EstadoSistema) -> EstadoSistema:
    """Recupera análises anteriores do ChromaDB."""
    try:
        from cap04.memory_store import buscar_com_filtro
        docs = buscar_com_filtro(e["tema"], filtros={}, k=3)
        contexto = "\n\n".join(d.page_content for d in docs) if docs else ""
        return {**e, "contexto_anterior": contexto, "status": "pesquisando"}
    except Exception:
        return {**e, "contexto_anterior": "", "status": "pesquisando"}


def executar_crew(e: EstadoSistema) -> EstadoSistema:
    """Executa a Crew de 3 agentes para produzir o relatório."""
    from cap08.crew_completa import criar_crew_inteligencia_mercado
    crew = criar_crew_inteligencia_mercado()
    prompt_adicional = ""
    if e.get("relatorio_bruto") and e.get("tentativas_validacao", 0) > 0:
        prompt_adicional = (
            "\n\nIMPORTANTE: A análise anterior foi rejeitada por validação. "
            "Responda APENAS em JSON válido com o schema: "
            "{\"tema\": str, \"resumo_executivo\": str, \"principais_players\": [str], "
            "\"tamanho_mercado\": str, \"crescimento_anual\": str, "
            "\"oportunidades\": [str], \"riscos\": [str], \"recomendacao\": str}"
        )
    resultado = crew.kickoff(inputs={"tema": e["tema"] + prompt_adicional})
    return {**e, "relatorio_bruto": resultado.raw, "status": "validando"}


def validar_relatorio_node(e: EstadoSistema) -> EstadoSistema:
    """Tenta extrair JSON estruturado do relatório bruto."""
    texto = e["relatorio_bruto"]

    # Tenta extrair JSON direto
    import re
    m = re.search(r'\{[^{}]*"tema"[^{}]*\}', texto, re.DOTALL)
    if m:
        try:
            dados = json.loads(m.group())
            if all(k in dados for k in ["tema", "principais_players", "oportunidades", "riscos", "recomendacao"]):
                return {**e, "relatorio_validado": dados, "status": "validado",
                        "tentativas_validacao": e.get("tentativas_validacao", 0) + 1}
        except json.JSONDecodeError:
            pass

    # Extrai campos estruturados do texto livre
    relatorio_estruturado = {
        "tema": e["tema"],
        "resumo_executivo": texto[:300] if texto else "",
        "principais_players": _extrair_lista(texto, ["player", "empresa", "concorrente"]),
        "tamanho_mercado": _extrair_valor(texto, ["mercado", "bilhões", "milhões", "R$"]),
        "crescimento_anual": _extrair_valor(texto, ["crescimento", "CAGR", "%"]),
        "oportunidades": _extrair_lista(texto, ["oportunidade", "potencial", "crescimento"]),
        "riscos": _extrair_lista(texto, ["risco", "desafio", "ameaça"]),
        "recomendacao": _extrair_recomendacao(texto)
    }

    valido = (len(relatorio_estruturado["principais_players"]) >= 1 and
              len(relatorio_estruturado["oportunidades"]) >= 1 and
              len(relatorio_estruturado["riscos"]) >= 1)

    return {
        **e,
        "relatorio_validado": relatorio_estruturado if valido else None,
        "status": "validado" if valido else "invalido",
        "tentativas_validacao": e.get("tentativas_validacao", 0) + 1
    }


def _extrair_lista(texto: str, palavras_chave: list[str]) -> list[str]:
    """Extrai frases relevantes do texto baseado em palavras-chave."""
    linhas = texto.split("\n")
    encontrados = []
    for linha in linhas:
        linha_lower = linha.lower()
        if any(kw in linha_lower for kw in palavras_chave):
            linha_limpa = linha.strip().lstrip("-•*123456789. ")
            if 10 < len(linha_limpa) < 200:
                encontrados.append(linha_limpa)
    return encontrados[:5] if encontrados else ["Informação não disponível"]


def _extrair_valor(texto: str, palavras_chave: list[str]) -> str:
    """Extrai primeiro valor numérico relevante do texto."""
    import re
    linhas = texto.split("\n")
    for linha in linhas:
        if any(kw.lower() in linha.lower() for kw in palavras_chave):
            m = re.search(r'R?\$?\s*[\d,\.]+\s*(bilh[õo]es?|milh[õo]es?|%)?', linha)
            if m:
                return m.group().strip()
    return "Dados não disponíveis"


def _extrair_recomendacao(texto: str) -> str:
    """Extrai recomendação final do texto."""
    linhas = texto.split("\n")
    for i, linha in enumerate(linhas):
        if "recomenda" in linha.lower():
            partes = linhas[i:i+3]
            return " ".join(p.strip() for p in partes if p.strip())[:300]
    return texto[-300:].strip() if texto else "Análise inconclusiva."


def salvar_resultado(e: EstadoSistema) -> EstadoSistema:
    """Salva relatório na memória longa (ChromaDB)."""
    try:
        from cap04.memory_store import ingerir_texto
        conteudo = json.dumps(e.get("relatorio_validado") or {"bruto": e["relatorio_bruto"][:500]},
                              ensure_ascii=False)
        ingerir_texto(conteudo, {
            "empresa": e["tema"],
            "tipo": "relatorio",
            "data": datetime.now().strftime("%Y-%m-%d"),
            "fonte": "sistema_completo"
        })
    except Exception:
        pass
    return {**e, "status": "concluido"}


def rotear_validacao(e: EstadoSistema) -> str:
    if e["status"] == "validado":
        return "salvar"
    elif e.get("tentativas_validacao", 0) >= 2:
        return "salvar"
    return "crew"


def criar_sistema_completo():
    """Cria e compila o grafo do sistema completo."""
    g = StateGraph(EstadoSistema)
    g.add_node("verificar_contexto", verificar_contexto)
    g.add_node("crew", executar_crew)
    g.add_node("validar", validar_relatorio_node)
    g.add_node("salvar", salvar_resultado)
    g.add_edge(START, "verificar_contexto")
    g.add_edge("verificar_contexto", "crew")
    g.add_edge("crew", "validar")
    g.add_conditional_edges("validar", rotear_validacao,
                            {"salvar": "salvar", "crew": "crew"})
    g.add_edge("salvar", END)
    return g.compile(checkpointer=MemorySaver())


def analisar_tema(tema: str) -> dict:
    """Executa análise completa e retorna o relatório."""
    app = criar_sistema_completo()
    estado_inicial: EstadoSistema = {
        "tema": tema,
        "contexto_anterior": "",
        "relatorio_bruto": "",
        "relatorio_validado": None,
        "tentativas_validacao": 0,
        "status": "iniciando"
    }
    config = {"configurable": {"thread_id": f"analise-{tema[:20].replace(' ', '-')}"}}
    resultado = app.invoke(estado_inicial, config=config)
    return resultado.get("relatorio_validado") or {
        "tema": tema,
        "erro": "Validação falhou após tentativas",
        "bruto": resultado.get("relatorio_bruto", "")[:500]
    }


def comparar_empresas(temas: list[str]) -> dict:
    """Analisa múltiplos temas e retorna comparativo."""
    resultados = {}
    for tema in temas:
        print(f"Analisando: {tema}...")
        resultados[tema] = analisar_tema(tema)

    comparativo = {
        "temas_analisados": temas,
        "data": datetime.now().strftime("%Y-%m-%d"),
        "comparativo_players": {},
        "comparativo_oportunidades": {},
        "resumo": []
    }
    for tema, r in resultados.items():
        if r and "erro" not in r:
            comparativo["comparativo_players"][tema] = r.get("principais_players", [])[:3]
            comparativo["comparativo_oportunidades"][tema] = r.get("oportunidades", [])[:2]
            comparativo["resumo"].append({
                "tema": tema,
                "recomendacao": r.get("recomendacao", "N/A")[:200],
                "tamanho": r.get("tamanho_mercado", "N/A")
            })
    return comparativo


class MonitoramentoMercado:
    """Monitoramento contínuo de múltiplos temas com detecção de mudanças."""

    def __init__(self, temas: list[str], intervalo_horas: int = 168):
        self.temas = temas
        self.intervalo = timedelta(hours=intervalo_horas)
        self.ultima_execucao: dict[str, datetime] = {}
        self.historico: dict[str, list[dict]] = {t: [] for t in temas}

    def precisa_atualizar(self, tema: str) -> bool:
        ultima = self.ultima_execucao.get(tema)
        return ultima is None or datetime.now() - ultima > self.intervalo

    def executar_monitoramento(self, tema: str) -> dict:
        relatorio = analisar_tema(tema)
        historico = self.historico.get(tema, [])
        mudancas = []

        if historico:
            players_anteriores = set(historico[-1].get("principais_players", []))
            players_atuais = set(relatorio.get("principais_players", []))
            novos = players_atuais - players_anteriores
            removidos = players_anteriores - players_atuais
            if novos:
                mudancas.append(f"Novos players: {', '.join(list(novos)[:3])}")
                print(f"[{tema}] Novos players: {novos}")
            if removidos:
                mudancas.append(f"Players saíram: {', '.join(list(removidos)[:3])}")

        relatorio["_mudancas"] = mudancas
        relatorio["_data_analise"] = datetime.now().isoformat()
        self.historico[tema].append(relatorio)
        self.ultima_execucao[tema] = datetime.now()
        return relatorio

    async def loop_monitoramento(self):
        """Loop assíncrono que verifica e atualiza análises periodicamente."""
        while True:
            for tema in self.temas:
                if self.precisa_atualizar(tema):
                    print(f"Atualizando análise: {tema}")
                    await asyncio.to_thread(self.executar_monitoramento, tema)
            await asyncio.sleep(3600)


if __name__ == "__main__":
    print("=== ANÁLISE DE CONCORRENTE ===")
    r1 = analisar_tema("plataformas de gestão financeira para PMEs")
    print(f"Tema: {r1.get('tema')}")
    print(f"Players: {r1.get('principais_players', [])[:3]}")
    print(f"Recomendação: {str(r1.get('recomendacao', ''))[:200]}")

    print("\n=== COMPARATIVO DE SETORES ===")
    comp = comparar_empresas(["healthtechs de telemedicina", "agritechs de precisão"])
    print(json.dumps(comp, ensure_ascii=False, indent=2))
