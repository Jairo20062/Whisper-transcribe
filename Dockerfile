# ---- Build stage: install deps and pre-download heavy models ---------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first so neither TTS nor pyannote pulls in CUDA weights.
# The +cpu local-version suffix satisfies torch>=x.y.z requirements from both packages.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch already present, pip skips it)
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper large-v3 (~3 GB) so container startup is instant.
# Model lands in /root/.cache/huggingface/hub/
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# Pre-download Coqui TTS nl/css10/vits (~100 MB).
# Model lands in /root/.local/share/tts/
ENV TTS_HOME=/root/.local/share/tts
RUN python -c "from TTS.api import TTS; TTS('tts_models/nl/css10/vits', gpu=False)"

# NOTE: pyannote/speaker-diarization-3.1 requires a Hugging Face token and
# cannot be downloaded at build time. It is fetched on first use and cached in
# /root/.cache/huggingface — mount this path as a Docker volume so the model
# persists across redeploys:
#
#   docker run -v hf-cache:/root/.cache/huggingface jb-whisper
#   (Coolify: add a persistent volume mapped to /root/.cache/huggingface)

# ---- Runtime stage ---------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages and pre-downloaded model caches from builder
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
