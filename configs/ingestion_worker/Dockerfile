# --------------------------
# Stage 1: Install dependencies: PyTorch (+CUDA)
# --------------------------
FROM python:3.12-slim AS base
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# --------------------------
# Stage 2: Runtime
# --------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy packages
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Install ffmpeg, pytesseract
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      ollama
 && rm -rf /var/lib/apt/lists/*

RUN ollama pull gemma3:12b

# Copy your code
COPY nn_ideas_manager ./nn_ideas_manager

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
CMD ["python", "-m", "nn_ideas_manager.core.workers.process_undigested_urls"]