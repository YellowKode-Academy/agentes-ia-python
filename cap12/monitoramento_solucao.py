"""
Capítulo 12 — Solução do exercício final
Endpoint POST /monitorar que:
1. Recebe lista de temas
2. Agenda análises semanais para cada tema
3. Compara relatório atual com o anterior e destaca mudanças
4. Envia diff por email via smtplib

Uso:
    uvicorn cap12.monitoramento_solucao:app --port 8002
    POST /monitorar {"temas": ["fintech", "healthtech"], "email": "seu@email.com"}
"""
import json
import asyncio
import smtplib
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Monitoramento de Mercado — Solução Cap 12", version="1.0.0")
API_KEY = os.getenv("API_KEY", "chave-dev-apenas")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

_monitoramentos: dict[str, dict] = {}
_historico: dict[str, list[dict]] = {}


async def auth(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Não autorizado")
    return key


class MonitorarRequest(BaseModel):
    temas: list[str]
    email: str  # Ex: "usuario@dominio.com"
    intervalo_dias: int = 7


def gerar_diff(anterior: dict, atual: dict) -> str:
    """Gera texto comparativo entre dois relatórios."""
    diff_partes = []

    players_ant = set(anterior.get("principais_players", []))
    players_atu = set(atual.get("principais_players", []))
    novos = players_atu - players_ant
    removidos = players_ant - players_atu

    if novos:
        diff_partes.append(f"NOVOS PLAYERS: {', '.join(list(novos)[:5])}")
    if removidos:
        diff_partes.append(f"PLAYERS SAÍRAM: {', '.join(list(removidos)[:5])}")

    tam_ant = anterior.get("tamanho_mercado", "")
    tam_atu = atual.get("tamanho_mercado", "")
    if tam_ant and tam_atu and tam_ant != tam_atu:
        diff_partes.append(f"TAMANHO DO MERCADO: {tam_ant} → {tam_atu}")

    rec_ant = str(anterior.get("recomendacao", ""))[:100]
    rec_atu = str(atual.get("recomendacao", ""))[:100]
    if rec_ant != rec_atu:
        diff_partes.append(f"MUDANÇA NA RECOMENDAÇÃO:\n  Antes: {rec_ant}\n  Agora:  {rec_atu}")

    return "\n".join(diff_partes) if diff_partes else "Nenhuma mudança significativa detectada."


def enviar_email_diff(destinatario: str, tema: str, diff: str, relatorio_atual: dict) -> bool:
    """Envia email com o diff do relatório via SMTP."""
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        print(f"[EMAIL] SMTP não configurado. Diff para '{tema}':\n{diff}")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Monitoramento] Atualização: {tema}"
        msg["From"] = smtp_user
        msg["To"] = destinatario

        corpo_texto = (
            f"RELATÓRIO SEMANAL — {tema.upper()}\n"
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"MUDANÇAS DETECTADAS:\n{diff}\n\n"
            f"PLAYERS ATUAIS: {', '.join(relatorio_atual.get('principais_players', [])[:5])}\n\n"
            f"RECOMENDAÇÃO ATUAL:\n{relatorio_atual.get('recomendacao', 'N/A')}"
        )
        corpo_html = f"""
        <html><body>
        <h2>Monitoramento de Mercado — {tema}</h2>
        <p><em>{datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>
        <h3>Mudanças Detectadas</h3>
        <pre>{diff}</pre>
        <h3>Players Atuais</h3>
        <p>{', '.join(relatorio_atual.get('principais_players', [])[:5])}</p>
        <h3>Recomendação Atual</h3>
        <p>{relatorio_atual.get('recomendacao', 'N/A')}</p>
        </body></html>
        """
        msg.attach(MIMEText(corpo_texto, "plain"))
        msg.attach(MIMEText(corpo_html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, destinatario, msg.as_string())
        print(f"[EMAIL] Enviado para {destinatario}: {tema}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False


async def executar_ciclo_monitoramento(monitor_id: str):
    """Loop que executa análises semanais e envia diffs por email."""
    from cap12.sistema_completo import analisar_tema

    config = _monitoramentos[monitor_id]
    temas = config["temas"]
    email = config["email"]
    intervalo = timedelta(days=config["intervalo_dias"])

    while monitor_id in _monitoramentos and _monitoramentos[monitor_id]["ativo"]:
        for tema in temas:
            ultima = _monitoramentos[monitor_id].get(f"ultima_{tema}")
            agora = datetime.now()

            if ultima is None or agora - datetime.fromisoformat(ultima) >= intervalo:
                print(f"[MONITOR {monitor_id}] Analisando: {tema}")
                _monitoramentos[monitor_id]["status"] = f"analisando: {tema}"

                try:
                    relatorio_atual = await asyncio.to_thread(analisar_tema, tema)
                    historico = _historico.setdefault(f"{monitor_id}:{tema}", [])

                    if historico:
                        diff = gerar_diff(historico[-1], relatorio_atual)
                        await asyncio.to_thread(enviar_email_diff, email, tema, diff, relatorio_atual)
                    else:
                        await asyncio.to_thread(
                            enviar_email_diff, email, tema,
                            "Primeira análise — sem comparativo anterior.", relatorio_atual
                        )

                    historico.append(relatorio_atual)
                    _monitoramentos[monitor_id][f"ultima_{tema}"] = agora.isoformat()

                except Exception as e:
                    print(f"[MONITOR {monitor_id}] Erro em '{tema}': {e}")

        _monitoramentos[monitor_id]["status"] = "aguardando_proximo_ciclo"
        await asyncio.sleep(3600)


@app.post("/monitorar")
async def iniciar_monitoramento(req: MonitorarRequest, _=Depends(auth)):
    """Inicia monitoramento semanal de temas com notificação por email."""
    monitor_id = f"monitor-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    _monitoramentos[monitor_id] = {
        "id": monitor_id,
        "temas": req.temas,
        "email": req.email,
        "intervalo_dias": req.intervalo_dias,
        "criado_em": datetime.now().isoformat(),
        "ativo": True,
        "status": "iniciando"
    }
    asyncio.create_task(executar_ciclo_monitoramento(monitor_id))
    return {
        "monitor_id": monitor_id,
        "temas": req.temas,
        "email": req.email,
        "intervalo": f"a cada {req.intervalo_dias} dia(s)",
        "mensagem": "Monitoramento iniciado. Primeiro relatório será gerado em breve."
    }


@app.get("/monitorar")
async def listar_monitoramentos(_=Depends(auth)):
    """Lista todos os monitoramentos ativos."""
    return [
        {
            "id": m["id"],
            "temas": m["temas"],
            "email": m["email"],
            "ativo": m["ativo"],
            "status": m.get("status", "desconhecido"),
            "criado_em": m["criado_em"]
        }
        for m in _monitoramentos.values()
    ]


@app.delete("/monitorar/{monitor_id}")
async def parar_monitoramento(monitor_id: str, _=Depends(auth)):
    """Para um monitoramento ativo."""
    if monitor_id not in _monitoramentos:
        raise HTTPException(404, "Monitoramento não encontrado")
    _monitoramentos[monitor_id]["ativo"] = False
    return {"mensagem": f"Monitoramento {monitor_id} parado"}


@app.get("/monitorar/{monitor_id}/historico")
async def historico_monitoramento(monitor_id: str, _=Depends(auth)):
    """Retorna histórico de relatórios de um monitoramento."""
    if monitor_id not in _monitoramentos:
        raise HTTPException(404, "Monitoramento não encontrado")
    config = _monitoramentos[monitor_id]
    historicos = {}
    for tema in config["temas"]:
        chave = f"{monitor_id}:{tema}"
        historicos[tema] = len(_historico.get(chave, []))
    return {"monitor_id": monitor_id, "temas": historicos}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "monitoramentos_ativos": sum(1 for m in _monitoramentos.values() if m["ativo"])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cap12.monitoramento_solucao:app", host="0.0.0.0", port=8002, reload=True)
