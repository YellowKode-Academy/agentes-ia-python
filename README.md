<p align="center">
  <img src="capa.jpg" alt="Agentes de IA com Python" width="320"/>
</p>

<h1 align="center">Agentes de IA com Python</h1>

<p align="center">
  Repositório companion do livro publicado por
  <a href="https://github.com/kelvinbiffi">@kelvinbiffi</a>
  e <a href="https://github.com/YellowKode-Academy">@YellowKode-Academy</a>
</p>

<p align="center">
  <a href="https://www.amazon.com.br/Agentes-com-Python-aut%C3%B4nomos-LangChain-ebook/dp/B0H2RZ1VBM/">Disponivel na Amazon KDP</a>
</p>

---

## Sobre este repositório

Contém todo o código Python referenciado no livro, do capítulo 1 ao 12. Cada módulo corresponde diretamente a um capítulo, com os arquivos de exercício e solução mencionados no texto.

O projeto central do livro é um agente de inteligência de mercado, construído peça por peça. No final, o sistema integra busca na web, memória longa, RAG em documentos, orquestração multi-agente e uma API REST completa.

## Setup

```bash
git clone https://github.com/YellowKode-Academy/agentes-ia-python
cd agentes-ia-python
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edite .env com suas chaves de API
```

## Variáveis de ambiente

Copie `.env.example` para `.env` e preencha:

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_API_KEY=ls__...         # opcional: para LangSmith
LANGCHAIN_TRACING_V2=true         # opcional: para LangSmith
LANGCHAIN_PROJECT=agente-mercado  # opcional: para LangSmith
GROQ_API_KEY=gsk_...              # opcional: fallback de modelo (cap10)
API_KEY=sua-chave-de-producao     # para a API REST do cap11
```

## Onde obter as chaves

| Variável | Onde criar | Plano gratuito |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) | Não (pay-as-you-go) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com/home) | Sim (1.000 req/mês) |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys | Sim |
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | Sim (rate limits) |
| `API_KEY` | Gere localmente: `python -c "import secrets; print(secrets.token_hex(32))"` | — |

## Estrutura por capítulo

| Capítulo | Modulo | O que voce constroi |
|---|---|---|
| 1 | `cap01/` | Primeiro agente com `create_react_agent` e `ChatAnthropic` |
| 2 | `cap02/` | Ferramentas customizadas: busca, calculadora, leitura de JSON |
| 3 | `cap03/` | Memoria de curto prazo com historico de conversas |
| 4 | `cap04/` | Memoria longa com ChromaDB e embeddings locais |
| 5 | `cap05/` | RAG: consulta a PDFs, CSVs e paginas web como ferramenta |
| 6 | `cap06/` | Observabilidade com LangSmith e callbacks customizados |
| 7 | `cap07/` | CrewAI: pesquisador e analista em processo sequencial |
| 8 | `cap08/` | CrewAI: tres agentes especializados com verificacao de qualidade |
| 9 | `cap09/` | Fluxos condicionais com LangGraph: loop de qualidade e retry |
| 10 | `cap10/` | Resiliencia: retry, fallback Claude para Groq, circuit breaker, cache |
| 11 | `cap11/` | API REST com FastAPI, job queue assincrono e Docker |
| 12 | `cap12/` | Sistema completo integrado com monitoramento continuo |

## Arquivos por modulo

Cada modulo tem, no minimo, dois arquivos:

- `[modulo].py` — implementacao principal usada no livro
- `exercicio_solucao.py` ou `*_solucao.py` — solucao completa do exercicio do capitulo

## Pastas de dados

```
dados/          JSON de exemplo com dados de concorrentes
resultados/     outputs gerados pelo agente (criado automaticamente)
cache/          cache de buscas com TTL (criado automaticamente)
logs/           logs de execucao (criado automaticamente)
memoria_longa/  ChromaDB persistente (criado automaticamente)
indices/        indices RAG por empresa (criado automaticamente)
```

## Versoes

Todas as dependencias estao em `requirements.txt` com versoes fixadas. Use exatamente essas versoes para garantir compatibilidade com o codigo do livro.

## Autor

Criado por [@kelvinbiffi](https://github.com/kelvinbiffi) para a serie de livros tecnicos da [@YellowKode-Academy](https://github.com/YellowKode-Academy).
