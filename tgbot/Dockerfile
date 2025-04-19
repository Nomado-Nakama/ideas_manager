# --------------------------
# Stage 1: PyTorch (+CUDA)
# --------------------------
FROM python:3.12-slim AS base
WORKDIR /install

COPY requirements.cuda.txt .
RUN pip install -r requirements.cuda.txt --extra-index-url https://download.pytorch.org/whl/cu126

# --------------------------
# Stage 2: Other deps (+git)
# --------------------------
FROM python:3.12-slim AS deps
WORKDIR /app

# Install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# Stage 3: Final runtime
# --------------------------
FROM python:3.12-slim AS runtime
WORKDIR /app

# Copy other packages
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy PyTorch libs
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

RUN apt get install ffmpeg

# Copy your code
COPY nn_ideas_manager ./nn_ideas_manager

CMD ["python", "-m", "nn_ideas_manager.main"]
