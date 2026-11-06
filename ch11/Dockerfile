FROM python:3.11-slim

WORKDIR /app

# System dependencies for sentence-transformers and pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Creates the directories the agent needs
RUN mkdir -p logs cache indexes long_term_memory data results

EXPOSE 8000

CMD ["uvicorn", "ch11.api:app", "--host", "0.0.0.0", "--port", "8000"]
