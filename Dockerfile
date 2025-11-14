# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build deps only here
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages in a local user folder
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

# Install runtime-only deps
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY app /app/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]