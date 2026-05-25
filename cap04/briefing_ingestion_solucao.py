"""
Capítulo 4 — Solução do exercício
Lê todos os .txt e .json de dados/, ingere no vector store com metadados,
testa 5 consultas e imprime total de chunks armazenados.
"""
import json
from pathlib import Path
from dotenv import load_dotenv
from cap04.memory_store import ingerir_arquivo, ingerir_texto, buscar_com_score

load_dotenv()

PASTA_DADOS = Path("dados")

CONSULTAS_TESTE = [
    "crescimento e expansão da empresa",
    "principais concorrentes e market share",
    "riscos e desafios do mercado",
    "tamanho da equipe e número de funcionários",
    "produto e diferenciais competitivos",
]


def ingerir_json_estruturado(caminho: Path) -> int:
    """Ingere arquivo JSON convertendo para texto legível."""
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    texto = json.dumps(dados, ensure_ascii=False, indent=2)
    empresa = dados.get("empresa", caminho.stem)
    return ingerir_texto(texto, {
        "empresa": empresa,
        "tipo": "dados_brutos",
        "fonte": str(caminho),
        "arquivo": caminho.name
    })


if __name__ == "__main__":
    total_chunks = 0

    print("=== INGESTÃO DE DADOS ===\n")
    if not PASTA_DADOS.exists():
        print(f"Pasta '{PASTA_DADOS}' não encontrada. Crie a pasta e adicione arquivos.")
    else:
        for arq in PASTA_DADOS.glob("*.json"):
            n = ingerir_json_estruturado(arq)
            total_chunks += n
            print(f"✓ {arq.name} — {n} chunks")

        for arq in PASTA_DADOS.glob("*.txt"):
            n = ingerir_arquivo(str(arq), {"tipo": "texto", "fonte": arq.name})
            total_chunks += n
            print(f"✓ {arq.name} — {n} chunks")

    print(f"\nTotal de chunks armazenados: {total_chunks}")

    print("\n=== TESTES DE BUSCA (com scores) ===\n")
    for consulta in CONSULTAS_TESTE:
        print(f"Consulta: '{consulta}'")
        resultados = buscar_com_score(consulta, k=2)
        for doc, score in resultados:
            print(f"  Score: {score:.3f} | {doc.page_content[:120]}...")
        print()
