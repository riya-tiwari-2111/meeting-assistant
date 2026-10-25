"""Sarvam AI transcription (good for Hindi / Indian languages and code-mixed speech).
NOTE: verify endpoint/model names against https://docs.sarvam.ai - they change."""
import io
import os
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv

from cloudtranscribe import TARGET_SR, _load_16k_mono

load_dotenv()
URL = "https://api.sarvam.ai/speech-to-text"
MODEL = "saarika:v2.5"
CHUNK_SECONDS = 25  # sync API accepts short clips only


def make_output_name(audio_path: str) -> str:
    p = Path(audio_path)
    return str(p.with_name(f"{p.stem}_transcript.txt"))


def process_single_file(audio_path: str) -> str:
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}
    audio = _load_16k_mono(audio_path)
    step = CHUNK_SECONDS * TARGET_SR
    lines = []
    for start in range(0, len(audio), step):
        chunk = audio[start:start + step]
        if np.max(np.abs(chunk)) < 1e-3:  # skip silence
            continue
        buf = io.BytesIO()
        sf.write(buf, chunk, TARGET_SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        r = requests.post(
            URL, headers=headers,
            files={"file": ("chunk.wav", buf, "audio/wav")},
            data={"model": MODEL, "language_code": "unknown"},
            timeout=120,
        )
        r.raise_for_status()
        text = r.json().get("transcript", "").strip()
        if text:
            lines.append(f"UNKNOWN: {text}")
    out = make_output_name(audio_path)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    return out
