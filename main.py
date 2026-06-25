import asyncio
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("jb-whisper")

SERVICE_KEY = os.getenv("SERVICE_KEY", "change-me-random-string")
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
MAX_AUDIO_BYTES = 500 * 1_024 * 1_024  # 500 MB

AUDIO_DIR = Path("/tmp/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}
_semaphore: asyncio.Semaphore | None = None  # created on startup (needs running loop)

whisper_model = None  # faster_whisper.WhisperModel
tts_model = None      # TTS.api.TTS

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="jb-whisper", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    global whisper_model, tts_model, _semaphore

    _semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

    logger.info("Loading Whisper large-v3 (CPU / int8)…")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    logger.info("Whisper ready.")

    logger.info("Loading Coqui TTS tts_models/nl/css10/vits…")
    from TTS.api import TTS
    tts_model = TTS("tts_models/nl/css10/vits", gpu=False)
    logger.info("TTS ready.")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _require_key(key: Optional[str]) -> None:
    if key != SERVICE_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Service-Key header")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    audio_url: str
    language: str = "en"


class TranslateRequest(BaseModel):
    audio_url: str
    kimi_api_key: str
    kimi_base_url: str = "https://api.moonshot.cn/v1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _download(url: str, dest: Path) -> None:
    """Stream-download audio to *dest*, enforcing the 500 MB cap."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = 0
            with dest.open("wb") as fh:
                async for chunk in resp.aiter_bytes(65_536):
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        dest.unlink(missing_ok=True)
                        raise HTTPException(413, "Audio file exceeds 500 MB limit")
                    fh.write(chunk)


def _transcribe(path: Path, language: Optional[str] = None) -> dict:
    """Run Whisper in a thread (called via asyncio.to_thread)."""
    segments_iter, info = whisper_model.transcribe(
        str(path),
        language=language or None,
        beam_size=5,
        vad_filter=True,
    )
    segments: list[dict] = []
    texts: list[str] = []
    for seg in segments_iter:
        text = seg.text.strip()
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        texts.append(text)
    return {
        "transcript": " ".join(texts),
        "segments": segments,
        "language": info.language,
        "duration": round(info.duration, 2),
    }


def _translate_chunks(
    segments: list[dict],
    kimi_api_key: str,
    kimi_base_url: str,
) -> tuple[str, list[dict]]:
    """Translate English segments to Dutch via Kimi API (10 segments per call)."""
    from openai import OpenAI

    client = OpenAI(api_key=kimi_api_key, base_url=kimi_base_url)
    translated: list[dict] = []

    for i in range(0, len(segments), 10):
        chunk = segments[i : i + 10]
        numbered = "\n".join(f"{j + 1}. {seg['text']}" for j, seg in enumerate(chunk))

        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional English-to-Dutch translator for broadcast journalism. "
                        "Translate the numbered sentences below to natural, spoken Dutch. "
                        "Return ONLY the numbered translations, one per line, in the exact same format. "
                        "No explanations, no extra text."
                    ),
                },
                {"role": "user", "content": numbered},
            ],
        )

        lines = [
            l.strip()
            for l in response.choices[0].message.content.strip().splitlines()
            if l.strip()
        ]

        for j, seg in enumerate(chunk):
            raw = lines[j] if j < len(lines) else seg["text"]
            # Strip leading "N. " prefix that the model echoes back
            nl_text = re.sub(r"^\d+\.\s*", "", raw)
            translated.append({"start": seg["start"], "end": seg["end"], "text": nl_text})

    transcript_nl = " ".join(s["text"] for s in translated)
    return transcript_nl, translated


def _synthesize_to_wav(text: str, wav_path: Path) -> None:
    """Generate Dutch speech with Coqui TTS (called via asyncio.to_thread)."""
    tts_model.tts_to_file(text=text, file_path=str(wav_path))


def _wav_to_mp3(wav: Path, mp3: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "4", str(mp3)],
        check=True,
        capture_output=True,
    )
    wav.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def _run_translate_job(job_id: str, req: TranslateRequest) -> None:
    src = Path(f"/tmp/{job_id}.audio")
    out_wav = AUDIO_DIR / f"{job_id}.wav"
    out_mp3 = AUDIO_DIR / f"{job_id}.mp3"

    def _progress(pct: int, status: str = "processing") -> None:
        jobs[job_id]["status"] = status
        jobs[job_id]["progress"] = pct

    async with _semaphore:
        try:
            _progress(2)
            logger.info("[%s] Downloading audio…", job_id)
            await _download(req.audio_url, src)

            _progress(10)
            logger.info("[%s] Transcribing…", job_id)
            result = await asyncio.to_thread(_transcribe, src, "en")
            src.unlink(missing_ok=True)

            _progress(35)
            logger.info("[%s] Translating %d segments…", job_id, len(result["segments"]))
            transcript_nl, segments_nl = await asyncio.to_thread(
                _translate_chunks,
                result["segments"],
                req.kimi_api_key,
                req.kimi_base_url,
            )

            _progress(65)
            logger.info("[%s] Synthesising Dutch audio…", job_id)
            await asyncio.to_thread(_synthesize_to_wav, transcript_nl, out_wav)

            _progress(85)
            logger.info("[%s] Converting WAV → MP3…", job_id)
            await asyncio.to_thread(_wav_to_mp3, out_wav, out_mp3)

            jobs[job_id].update(
                {
                    "status": "done",
                    "progress": 100,
                    "result": {
                        "transcript_en": result["transcript"],
                        "transcript_nl": transcript_nl,
                        "segments_nl": segments_nl,
                        "audio_url": f"/audio/{job_id}.mp3",
                        "duration": result["duration"],
                    },
                }
            )
            logger.info("[%s] Done.", job_id)

        except Exception as exc:
            logger.exception("[%s] Job failed", job_id)
            src.unlink(missing_ok=True)
            jobs[job_id].update(
                {"status": "failed", "progress": 0, "result": {"error": str(exc)}}
            )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "large-v3"}


@app.post("/transcribe")
async def transcribe(
    req: TranscribeRequest,
    x_service_key: Optional[str] = Header(None),
) -> dict:
    _require_key(x_service_key)
    tmp = Path(f"/tmp/{uuid.uuid4()}.audio")
    try:
        await _download(req.audio_url, tmp)
        return await asyncio.to_thread(_transcribe, tmp, req.language)
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/translate", status_code=202)
async def translate(
    req: TranslateRequest,
    background_tasks: BackgroundTasks,
    x_service_key: Optional[str] = Header(None),
) -> dict:
    _require_key(x_service_key)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"job_id": job_id, "status": "pending", "progress": 0, "result": None}
    background_tasks.add_task(_run_translate_job, job_id, req)
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    x_service_key: Optional[str] = Header(None),
) -> dict:
    _require_key(x_service_key)
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/audio/{job_id}.mp3")
async def get_audio(
    job_id: str,
    x_service_key: Optional[str] = Header(None),
) -> FileResponse:
    _require_key(x_service_key)
    path = AUDIO_DIR / f"{job_id}.mp3"
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(str(path), media_type="audio/mpeg", filename=f"{job_id}.mp3")
