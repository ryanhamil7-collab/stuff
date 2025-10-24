FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/models data/knowledge_base logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["python", "main.py", "dashboard"]
