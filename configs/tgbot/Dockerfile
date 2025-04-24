# --------------------------
# Stage 1: Install deps
# --------------------------
FROM python:3.12-slim AS deps
WORKDIR /app

COPY ./configs/tgbot/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# Stage 2: Runtime
# --------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy other packages
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy your code
COPY nn_ideas_manager ./nn_ideas_manager

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
CMD ["python", "-m", "nn_ideas_manager.main"]
