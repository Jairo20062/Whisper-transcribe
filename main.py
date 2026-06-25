import asyncio
import io
import json
import logging
import os
import queue as _queue
import re
import subprocess
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
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
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_AUDIO_BYTES = 500 * 1_024 * 1_024  # 500 MB

AUDIO_DIR = Path("/tmp/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Thread pool for CPU-bound work (Whisper + pyannote)
_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jb-worker")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}
_semaphore: asyncio.Semaphore | None = None  # created on startup (needs running loop)

whisper_model = None        # faster_whisper.WhisperModel
tts_model = None            # TTS.api.TTS
diarization_pipeline = None # pyannote.audio.Pipeline (optional)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="jb-whisper", version="1.1.0")


@app.on_event("startup")
async def _startup() -> None:
    global whisper_model, tts_model, diarization_pipeline, _semaphore

    _semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

    logger.info("Loading Whisper large-v3 (CPU / int8)…")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    logger.info("Whisper ready.")

    logger.info("Loading Coqui TTS tts_models/nl/css10/vits…")
    from TTS.api import TTS
    tts_model = TTS("tts_models/nl/css10/vits", gpu=False)
    logger.info("TTS ready.")

    if HF_TOKEN:
        logger.info("Loading pyannote speaker-diarization-3.1…")
        try:
            from pyannote.audio import Pipeline
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN,
            )
            logger.info("Diarization pipeline ready.")
        except Exception:
            logger.exception("Failed to load diarization pipeline — continuing without it.")
    else:
        logger.info("HF_TOKEN not set — speaker diarization disabled.")


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
# HTTP helpers (unchanged)
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
# WebSocket — session state
# ---------------------------------------------------------------------------

@dataclass
class _TranscriptionSession:
    language: str
    segments: list[dict] = field(default_factory=list)
    # All PCM data accumulated for post-processing diarization
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    # Running total of audio seconds received (used to offset chunk timestamps)
    cumulative_offset: float = 0.0
    # Last N chars of transcript — fed as initial_prompt to improve continuity
    previous_text: str = ""


# ---------------------------------------------------------------------------
# WebSocket — audio helpers
# ---------------------------------------------------------------------------

def _decode_wav_bytes(data: bytes) -> np.ndarray:
    """Decode WAV bytes (16 kHz, mono, int16) → float32 numpy array."""
    with wave.open(io.BytesIO(data)) as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32_768.0


async def _stream_transcribe_chunk(
    audio: np.ndarray,
    language: str,
    offset: float,
    initial_prompt: str = "",
) -> AsyncIterator[dict]:
    """
    Async generator — runs faster-whisper in the thread pool and yields each
    segment dict as soon as the model produces it, keeping the event loop free.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _worker() -> None:
        try:
            segs, _ = whisper_model.transcribe(
                audio,
                language=language,
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=True,
                initial_prompt=initial_prompt or None,
            )
            for seg in segs:
                item = {
                    "start": round(seg.start + offset, 2),
                    "end": round(seg.end + offset, 2),
                    "text": seg.text.strip(),
                }
                loop.call_soon_threadsafe(q.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

    future = loop.run_in_executor(_thread_pool, _worker)
    try:
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await future  # always join the thread


def _write_wav_temp(audio: np.ndarray, sample_rate: int = 16_000) -> Path:
    """Write float32 audio to a temp WAV file. Caller is responsible for deletion."""
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    pcm = (np.clip(audio, -1.0, 1.0) * 32_767).astype(np.int16)
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return tmp


# ---------------------------------------------------------------------------
# WebSocket — diarization helpers
# ---------------------------------------------------------------------------

def _diarize_sync(audio: np.ndarray) -> list[tuple[float, float, str]]:
    """
    Blocking: write audio to temp WAV, run pyannote, return
    (start, end, "Spreker N") turns sorted by start time.
    """
    tmp = _write_wav_temp(audio)
    try:
        diarization = diarization_pipeline(str(tmp))
        speaker_map: dict[str, str] = {}
        counter = 1
        turns: list[tuple[float, float, str]] = []
        for turn, _, raw_label in diarization.itertracks(yield_label=True):
            if raw_label not in speaker_map:
                speaker_map[raw_label] = f"Spreker {counter}"
                counter += 1
            turns.append((turn.start, turn.end, speaker_map[raw_label]))
        return turns
    finally:
        tmp.unlink(missing_ok=True)


def _assign_speakers(
    segments: list[dict],
    turns: list[tuple[float, float, str]],
) -> list[dict]:
    """
    For each segment, find the diarization turn with the most timestamp overlap.
    Falls back to the previous segment's speaker when there is no overlap.
    """
    result: list[dict] = []
    prev_speaker = "Spreker 1"
    for seg in segments:
        s, e = seg["start"], seg["end"]
        best_label, best_overlap = prev_speaker, 0.0
        for ts, te, label in turns:
            overlap = max(0.0, min(e, te) - max(s, ts))
            if overlap > best_overlap:
                best_overlap, best_label = overlap, label
        prev_speaker = best_label
        result.append({**seg, "speaker": best_label})
    return result


# ---------------------------------------------------------------------------
# WebSocket — stop handler
# ---------------------------------------------------------------------------

async def _handle_stop(
    websocket: WebSocket,
    session: _TranscriptionSession,
    conn_id: str,
) -> None:
    await websocket.send_json({
        "type": "processing",
        "message": "Spreker detectie wordt uitgevoerd...",
    })

    updated = list(session.segments)  # defensive copy

    if diarization_pipeline is not None and session.audio_buffer.size > 0:
        try:
            turns = await asyncio.to_thread(_diarize_sync, session.audio_buffer)
            updated = _assign_speakers(updated, turns)
            logger.info("[ws:%s] Diarization complete: %d speaker turns", conn_id, len(turns))
        except Exception:
            logger.exception("[ws:%s] Diarization failed — keeping Spreker 1 for all", conn_id)
            # updated already has "Spreker 1" from the real-time phase; keep it
    else:
        reason = "no HF_TOKEN" if diarization_pipeline is None else "empty audio buffer"
        logger.info("[ws:%s] Diarization skipped (%s)", conn_id, reason)

    await websocket.send_json({"type": "speakers_updated", "segments": updated})
    await websocket.send_json({"type": "done"})


# ---------------------------------------------------------------------------
# Background job runner (unchanged)
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
# HTTP routes (unchanged)
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


# ---------------------------------------------------------------------------
# WebSocket route
# ---------------------------------------------------------------------------

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    """
    Real-time meeting transcription with post-processing speaker diarization.

    Connect: ws://host/ws/transcribe?key=SERVICE_KEY&lang=nl
    Send:    binary WAV frames (16 kHz, mono, int16, ~2 s each)
             OR JSON {"type": "stop"} to end the session
    Receive: {"type": "segment",          "text": "...", "speaker": "Spreker 1", "start": 0.0, "end": 2.0}
             {"type": "partial",          "text": "..."}       — interim segment
             {"type": "processing",       "message": "..."}    — diarization started
             {"type": "speakers_updated", "segments": [...]}   — final with speaker labels
             {"type": "done"}                                  — session complete
    """
    # Auth via query param (WebSocket handshake can't carry custom headers easily)
    key = websocket.query_params.get("key", "")
    lang = websocket.query_params.get("lang", "nl")

    await websocket.accept()

    if key != SERVICE_KEY:
        await websocket.close(code=1008, reason="Invalid service key")
        return

    conn_id = str(uuid.uuid4())[:8]
    logger.info("[ws:%s] Connected — lang=%s", conn_id, lang)

    session = _TranscriptionSession(language=lang)

    try:
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("[ws:%s] Client disconnected", conn_id)
                break

            if msg["type"] == "websocket.disconnect":
                logger.info("[ws:%s] Received disconnect frame", conn_id)
                break

            raw_bytes: bytes | None = msg.get("bytes")
            raw_text: str | None = msg.get("text")

            # ── Binary frame: audio chunk ────────────────────────────────
            if raw_bytes:
                try:
                    audio_chunk = _decode_wav_bytes(raw_bytes)
                except Exception as exc:
                    logger.warning("[ws:%s] WAV decode error, skipping frame: %s", conn_id, exc)
                    continue

                chunk_offset = session.cumulative_offset
                session.cumulative_offset += len(audio_chunk) / 16_000.0
                session.audio_buffer = np.concatenate([session.audio_buffer, audio_chunk])

                try:
                    async for seg in _stream_transcribe_chunk(
                        audio_chunk,
                        session.language,
                        chunk_offset,
                        session.previous_text,
                    ):
                        if not seg["text"]:
                            continue

                        # Update context for next chunk
                        session.previous_text = (
                            session.previous_text + " " + seg["text"]
                        )[-500:]

                        seg["speaker"] = "Spreker 1"
                        session.segments.append(seg)

                        await websocket.send_json({
                            "type": "segment",
                            "text": seg["text"],
                            "speaker": "Spreker 1",
                            "start": seg["start"],
                            "end": seg["end"],
                        })

                except Exception:
                    logger.exception("[ws:%s] Transcription error — skipping chunk", conn_id)
                    # Don't close; let the session continue

            # ── Text frame: control message ──────────────────────────────
            elif raw_text:
                try:
                    ctrl = json.loads(raw_text)
                except json.JSONDecodeError:
                    logger.warning("[ws:%s] Received non-JSON text, ignoring", conn_id)
                    continue

                if ctrl.get("type") == "stop":
                    await _handle_stop(websocket, session, conn_id)
                    break

    finally:
        # Free memory regardless of how the connection ended
        session.segments.clear()
        del session.audio_buffer
        logger.info("[ws:%s] Session cleaned up", conn_id)
