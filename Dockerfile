# ── Stage 1: build the React frontend ──────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + built frontend ────────────────────
FROM python:3.12-slim
WORKDIR /app

# ffmpeg needed by faster-whisper/gTTS audio handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Cloud Run injects PORT at runtime; server.py already reads it correctly
ENV PORT=8080
EXPOSE 8080

CMD ["python", "server.py"]