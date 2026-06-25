# jb-whisper

Audio transcription and translation service for JB News podcast episodes.

**Stack:** FastAPI · faster-whisper large-v3 · Kimi/Moonshot LLM · Coqui TTS · ffmpeg · Docker

---

## How it works

1. Client POSTs an audio URL to `/transcribe` or `/translate`.
2. The service downloads the audio, transcribes it with Whisper large-v3 (CPU, int8).
3. For `/translate`: segments are translated to Dutch via the Kimi API (10 at a time), then Coqui TTS generates a Dutch MP3.
4. `/translate` returns a `job_id` immediately (HTTP 202). The client polls `/jobs/{job_id}` until `status = done`.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SERVICE_KEY` | `change-me-random-string` | Shared secret sent in `X-Service-Key` header |
| `PORT` | `8001` | Port the server listens on |
| `MAX_CONCURRENT_JOBS` | `2` | Max parallel translate jobs (each needs ~6 GB RAM) |

Copy `.env.example` to `.env` and set a real `SERVICE_KEY`:

```bash
cp .env.example .env
# edit .env and set SERVICE_KEY=$(openssl rand -hex 32)
```

---

## Run locally

### Without Docker (development)

```bash
# Prerequisites: Python 3.11, ffmpeg
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu

pip install -r requirements.txt

cp .env.example .env
# edit .env

uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Models download automatically on first startup (~3 GB Whisper + ~100 MB TTS).

### With Docker

```bash
# Build (downloads models — takes 10-20 min and ~15 GB disk)
docker build -t jb-whisper .

# Run
docker run -d \
  --name jb-whisper \
  -p 8001:8001 \
  -e SERVICE_KEY=your-secret-here \
  -e MAX_CONCURRENT_JOBS=2 \
  -v jb-whisper-audio:/tmp/audio \
  jb-whisper
```

---

## Deploy on Coolify

1. Push this repo to GitHub / GitLab.
2. In Coolify → New Service → **Dockerfile**.
3. Set the **port** to `8001`.
4. Add environment variables:
   - `SERVICE_KEY` — a strong random string
   - `MAX_CONCURRENT_JOBS` — `1` or `2` depending on server RAM
5. Set **memory limit** to at least `12 GB` (Whisper large-v3 needs ~6 GB per job).
6. Enable **persistent storage**: mount a volume to `/tmp/audio` so MP3 files survive restarts.
7. Deploy. First build takes ~15-20 minutes (Whisper download is ~3 GB).

---

## API reference

All endpoints require the header:

```
X-Service-Key: <your SERVICE_KEY>
```

### GET /health

```bash
curl http://localhost:8001/health
# {"status":"ok","model":"large-v3"}
```

---

### POST /transcribe

Transcribe audio to text. Synchronous — waits for the result.

```bash
curl -X POST http://localhost:8001/transcribe \
  -H "X-Service-Key: your-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/episode.mp3",
    "language": "en"
  }'
```

Response:

```json
{
  "transcript": "Hello and welcome to the show...",
  "segments": [
    { "start": 0.0, "end": 5.2, "text": "Hello and welcome to the show." }
  ],
  "language": "en",
  "duration": 3600.0
}
```

---

### POST /translate

Transcribe English audio, translate to Dutch, generate Dutch MP3.
Returns immediately with a `job_id`; poll `/jobs/{job_id}` for progress.

```bash
curl -X POST http://localhost:8001/translate \
  -H "X-Service-Key: your-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/episode.mp3",
    "kimi_api_key": "sk-...",
    "kimi_base_url": "https://api.moonshot.cn/v1"
  }'
```

Response (HTTP 202):

```json
{ "job_id": "3f8a1c2d-...", "status": "pending" }
```

---

### GET /jobs/{job_id}

Poll job status. `progress` is 0-100.

```bash
JOB_ID="3f8a1c2d-..."

curl http://localhost:8001/jobs/$JOB_ID \
  -H "X-Service-Key: your-secret"
```

While processing:

```json
{ "job_id": "...", "status": "processing", "progress": 65, "result": null }
```

When done:

```json
{
  "job_id": "...",
  "status": "done",
  "progress": 100,
  "result": {
    "transcript_en": "Hello and welcome...",
    "transcript_nl": "Hallo en welkom...",
    "segments_nl": [
      { "start": 0.0, "end": 5.2, "text": "Hallo en welkom bij de show." }
    ],
    "audio_url": "/audio/3f8a1c2d-....mp3",
    "duration": 3600.0
  }
}
```

---

### GET /audio/{job_id}.mp3

Download the generated Dutch audio file.

```bash
curl -O http://localhost:8001/audio/$JOB_ID.mp3 \
  -H "X-Service-Key: your-secret"
```

---

## Full translate + poll loop (bash)

```bash
SERVICE_KEY="your-secret"
BASE_URL="https://jb-whisper.yourdomain.com"

# Start job
RESPONSE=$(curl -s -X POST $BASE_URL/translate \
  -H "X-Service-Key: $SERVICE_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"audio_url\": \"https://cdn.jbnews.nl/episodes/ep42.mp3\",
    \"kimi_api_key\": \"sk-your-kimi-key\"
  }")

JOB_ID=$(echo $RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job: $JOB_ID"

# Poll until done
while true; do
  STATUS=$(curl -s $BASE_URL/jobs/$JOB_ID -H "X-Service-Key: $SERVICE_KEY")
  PROGRESS=$(echo $STATUS | python3 -c "import sys,json; print(json.load(sys.stdin)['progress'])")
  STATE=$(echo $STATUS  | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "[$STATE] $PROGRESS%"
  [ "$STATE" = "done" ] || [ "$STATE" = "failed" ] && break
  sleep 15
done

# Download Dutch audio
curl -o dutch_episode.mp3 $BASE_URL/audio/$JOB_ID.mp3 \
  -H "X-Service-Key: $SERVICE_KEY"
```

---

## Connecting to JB News

The typical integration from a JB News CMS or automation:

1. When a new English podcast is published, POST the CDN URL to `/translate` with your Kimi API key.
2. Store the returned `job_id` alongside the episode record.
3. Poll `/jobs/{job_id}` every 30 seconds until `status = done`.
4. Upload the Dutch MP3 from `/audio/{job_id}.mp3` to your CDN/storage.
5. Store `transcript_nl` and `segments_nl` in your CMS for subtitles / show notes.

---

## Performance on CPU

| Audio length | Transcription | Translation (API) | TTS | Total |
|---|---|---|---|---|
| 10 min | ~3 min | ~1 min | ~2 min | ~6 min |
| 30 min | ~9 min | ~3 min | ~6 min | ~18 min |
| 60 min | ~18 min | ~6 min | ~12 min | ~36 min |

*Estimated on a 4-core CPU with int8 quantisation. Times vary by audio clarity.*

---

## Docker build time estimate

| Step | Time | Size |
|---|---|---|
| Install Python deps (torch, etc.) | 5-8 min | ~4 GB |
| Download Whisper large-v3 | 5-10 min | ~3 GB |
| Download Coqui TTS nl model | 1-2 min | ~100 MB |
| **Total image size** | — | **~15 GB** |

Build on a fast internet connection: **~15 minutes**.  
Subsequent builds use Docker layer cache — only changed layers rebuild.

---

## Supported audio formats

mp3, m4a, wav, ogg — passed directly to ffmpeg/Whisper.  
Maximum file size: **500 MB**.
