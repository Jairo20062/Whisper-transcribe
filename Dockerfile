# ---- Build stage: pre-download heavy models --------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Whisper large-v3 (~3 GB) at build time so startup is instant.
# Models land in /root/.cache/huggingface/hub/
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# Download Coqui TTS nl/css10/vits model (~100 MB).
# Models land in /root/.local/share/tts/
ENV TTS_HOME=/root/.local/share/tts
RUN python -c "from TTS.api import TTS; TTS('tts_models/nl/css10/vits', gpu=False)"

# ---- Runtime stage ---------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages and cached models from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /root/.cache
COPY --from=builder /root/.local /root/.local

# Application source
COPY main.py .

ENV TTS_HOME=/root/.local/share/tts \
    PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" \
    || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
