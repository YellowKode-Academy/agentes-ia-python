FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema para sentence-transformers e pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cria diretórios necessários para o agente
RUN mkdir -p logs cache indices memoria_longa dados resultados

EXPOSE 8000

CMD ["uvicorn", "cap11.api:app", "--host", "0.0.0.0", "--port", "8000"]
