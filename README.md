# Agentes de IA com Python — Repositório Companion

Código de referência do livro **Agentes de IA com Python** (Kelvin Biffi / YellowKode).

## Setup

```bash
git clone https://github.com/YellowKode-Academy/agentes-ia-python
cd agentes-ia-python
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edite .env com suas chaves de API
```

## Variáveis de ambiente

Copie `.env.example` para `.env` e preencha:

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_API_KEY=ls__...        # opcional — para LangSmith
LANGCHAIN_TRACING_V2=true        # opcional — para LangSmith
LANGCHAIN_PROJECT=agente-mercado # opcional — para LangSmith
GROQ_API_KEY=gsk_...             # opcional — para fallback de modelo
API_KEY=sua-chave-de-producao    # para a API REST do cap11
```

## Estrutura por capítulo

| Capítulo | Módulo | Conteúdo |
|---|---|---|
| 1 | `cap01/` | Agente base com `create_agent` + `ChatAnthropic` |
| 2 | `cap02/` | Ferramentas: Tavily, calculadora, leitura de JSON |
| 3 | `cap03/` | Memória de curto prazo com `MemorySaver` |
| 4 | `cap04/` | Memória longa com ChromaDB + embeddings |
| 5 | `cap05/` | RAG com documentos: PDF, CSV, web |
| 6 | `cap06/` | Observabilidade: LangSmith + callbacks |
| 7 | `cap07/` | CrewAI: pesquisador + analista (2 agentes) |
| 8 | `cap08/` | CrewAI: pesquisador + analista + redator (3 agentes) |
| 9 | `cap09/` | Fluxos condicionais com LangGraph |
| 10 | `cap10/` | Resiliência: retry, fallback, circuit breaker, cache |
| 11 | `cap11/` | API REST com FastAPI + Docker |
| 12 | `cap12/` | Sistema completo integrado |

## Pastas de dados

- `dados/` — arquivos JSON de exemplo com dados de concorrentes
- `resultados/` — outputs gerados pelo agente (criado automaticamente)
- `cache/` — cache de buscas com TTL (criado automaticamente)
- `logs/` — logs de execução (criado automaticamente)
- `memoria_longa/` — ChromaDB persistente (criado automaticamente)
- `indices/` — índices RAG por empresa (criado automaticamente)

## Versões

Todas as dependências estão em `requirements.txt` com versões fixadas.
Use exatamente essas versões para garantir compatibilidade com o código do livro.
