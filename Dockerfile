FROM python:3.9-slim

# HuggingFace Spaces metadata
LABEL hf_space="true"
LABEL openenv="true"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/state')" || exit 1

# Judges: pass your .env file at runtime:
#   docker run --env-file .env -p 8000:8000 cloudfinops-env
CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "8000"]
