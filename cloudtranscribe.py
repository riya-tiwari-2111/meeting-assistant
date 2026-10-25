"""Whisper transcription via Groq's hosted API (chunked to stay under upload limits)."""
import os
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq
from scipy.signal import resample_poly

load_dotenv()
MODEL = "whisper-large-v3-turbo"
TARGET_SR = 16000
CHUNK_SECONDS = 600


def make_output_name(audio_path: str) -> str:
    p = Path(audio_path)
    return str(p.with_name(f"{p.stem}_transcript.txt"))


def _load_16k_mono(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, sr // g)
    return data.astype(np.float32)


def process_single_file(audio_path: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    audio = _load_16k_mono(audio_path)
    step = CHUNK_SECONDS * TARGET_SR
    lines = []
    for start in range(0, len(audio), step):
        chunk = audio[start:start + step]
        with tempfile.TemporaryDirectory() as tmp:
            chunk_path = os.path.join(tmp, "chunk.wav")
            sf.write(chunk_path, chunk, TARGET_SR, subtype="PCM_16")
            with open(chunk_path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=("chunk.wav", fh.read()),
                    model=MODEL,
                    response_format="verbose_json",
                )
        segments = getattr(resp, "segments", None) or []
        for seg in segments:
            text = (seg["text"] if isinstance(seg, dict) else seg.text).strip()
            if text:
                lines.append(f"UNKNOWN: {text}")
        if not segments and getattr(resp, "text", "").strip():
            lines.append(f"UNKNOWN: {resp.text.strip()}")
    out = make_output_name(audio_path)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    return out
