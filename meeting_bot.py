import argparse
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None

import cloudtranscribe
import sarvam_transcribe
from ingest import ingest
from meeting_library import get_meeting_dir, write_meeting_metadata

# Optional: force a loopback device by (partial) name, e.g. "CABLE Output (VB-Audio Virtual Cable)".
# None = use the default output device's loopback.
PREFERRED_LOOPBACK_DEVICE: str | None = None

LEAVE_SELECTORS = [
    'button[aria-label="Leave call"]',
    'button[aria-label*="Leave call"]',
    '[role="button"][aria-label*="Leave call"]',
]


def find_best_loopback_wasapi(pa):
    """Pick the WASAPI loopback endpoint matching the default output device."""
    if pyaudio is None:
        raise RuntimeError("pyaudiowpatch is not installed. Run `pip install pyaudiowpatch`.")
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except Exception as exc:
        raise RuntimeError(f"WASAPI host API is not available: {exc}") from exc

    default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    print(f"[*] Default output device: {default_out['name']}")

    loopbacks = [(i, pa.get_device_info_by_index(i)) for i in range(pa.get_device_count())
                 if pa.get_device_info_by_index(i).get("isLoopbackDevice")]
    if not loopbacks:
        raise RuntimeError("No WASAPI loopback device found. Enable 'Stereo Mix' or install VB-Cable.")
    for _, info in loopbacks:
        print(f"      - {info['name']}")

    if PREFERRED_LOOPBACK_DEVICE:
        pref = next(((i, info) for i, info in loopbacks
                     if PREFERRED_LOOPBACK_DEVICE.lower() in info["name"].lower()), None)
        if pref:
            return pref
        print(f"[!] Preferred device '{PREFERRED_LOOPBACK_DEVICE}' not found; falling back.")

    name = default_out["name"].lower()
    match = next(((i, info) for i, info in loopbacks
                  if name in info["name"].lower() or info["name"].lower() in name), None)
    return match or loopbacks[0]


class MeetingBot:
    def __init__(self, meeting_url: str, transcription_engine: str = "whisper"):
        self.meeting_url = meeting_url
        self.transcription_engine = transcription_engine
        self.started_at = datetime.now()
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", meeting_url).strip("-").lower()[-40:] or "google-meet"
        self.meeting_id = f"{self.started_at.strftime('%Y%m%d_%H%M%S')}_{slug}"
        self.meeting_dir = get_meeting_dir(self.meeting_id)
        self.documents_dir = self.meeting_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir = self.meeting_dir / "vector_store"
        self.collection_name = f"meeting_{self.started_at.strftime('%Y%m%d_%H%M%S')}"
        self.audio_file = str(self.meeting_dir / f"{self.meeting_id}.wav")

        self.recording = False
        self.recording_started = threading.Event()
        self.recording_error = None
        self.stop_recording = threading.Event()
        self._peak_lock = threading.Lock()
        self._last_peak = 0.0

    @staticmethod
    def _extract_participants(text: str) -> list[str]:
        """Best-effort: collect unique 'Speaker:' prefixes from transcript lines."""
        seen, out = set(), []
        for line in text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            speaker = line.split(":", 1)[0].strip()
            if speaker and speaker.lower() not in seen:
                seen.add(speaker.lower())
                out.append(speaker)
        return out

    # ── Audio recording ────────────────────────────────────────────────────
    def _record_audio(self):
        print(f"[*] Starting WASAPI loopback recording -> {self.audio_file}")
        self.recording = True
        try:
            if pyaudio is None:
                raise RuntimeError("pyaudiowpatch is not installed.")
            pa = pyaudio.PyAudio()
            stream = None
            chunks: list[np.ndarray] = []
            try:
                idx, info = find_best_loopback_wasapi(pa)
                rate = int(info["defaultSampleRate"])
                channels = min(int(info["maxInputChannels"]), 2)
                if channels < 1:
                    raise RuntimeError(f"Loopback device has no input channels: {info['name']}")
                print(f"[*] Recording: {info['name']} | {rate} Hz | {channels} ch")

                def callback(in_data, frame_count, time_info, status):
                    if in_data:
                        chunk = np.frombuffer(in_data, dtype=np.float32).copy()
                        if channels > 1:  # keep the louder channel
                            chunk = chunk.reshape(-1, channels)
                            chunk = chunk[:, int(np.argmax((chunk ** 2).sum(axis=0)))]
                        with self._peak_lock:
                            self._last_peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                        chunks.append(chunk)
                    return (None, pyaudio.paContinue)

                stream = pa.open(format=pyaudio.paFloat32, channels=channels, rate=rate, input=True,
                                 input_device_index=idx, frames_per_buffer=max(1024, int(rate * 0.1)),
                                 stream_callback=callback)
                self.recording_started.set()
                stream.start_stream()
                while not self.stop_recording.is_set():
                    time.sleep(0.2)

                if chunks:
                    audio = np.concatenate(chunks).astype(np.float32, copy=False)
                    peak = float(np.max(np.abs(audio)))
                    if peak > 0.99:
                        audio = audio / peak
                    sf.write(self.audio_file, audio, rate, subtype="PCM_16")
                    print(f"\n[*] Saved {len(audio) / rate:.1f}s of audio -> {self.audio_file}")
                else:
                    print("\n[!] No audio data captured.")
            finally:
                if stream is not None:
                    for fn in (stream.stop_stream, stream.close):
                        try:
                            fn()
                        except Exception:
                            pass
                pa.terminate()
        except Exception as exc:
            self.recording_error = exc
            self.recording_started.set()  # unblock main thread
            print(f"\n[!] Audio recording failed: {exc}")
        finally:
            self.recording = False
            print("\n[*] Audio recording stopped.")

    def _peak_monitor(self):
        """Live terminal level meter. Empty bar the whole time = audio isn't reaching the recorder."""
        while not self.stop_recording.is_set():
            time.sleep(1)
            with self._peak_lock:
                peak = self._last_peak
            n = min(30, int(peak * 150))
            print(f"\r  [{'█' * n}{'░' * (30 - n)}] {'SILENT' if n == 0 else 'AUDIO '}  peak={peak:.4f}",
                  end="", flush=True)

    # ── Google Meet automation ─────────────────────────────────────────────
    def _dismiss_prejoin_popups(self, page):
        selectors = ['button:has-text("Got it")', 'button:has-text("OK")',
                     'button:has-text("Continue without microphone and camera")',
                     'button[aria-label="Close"]', '[role="button"]:has-text("Got it")']
        for _ in range(4):
            clicked = False
            for sel in selectors:
                try:
                    loc = page.locator(sel).first
                    loc.wait_for(state="visible", timeout=1200)
                    loc.click(timeout=2000)
                    page.wait_for_timeout(600)
                    clicked = True
                    print(f"[*] Dismissed popup: {sel}")
                except Exception:
                    continue
            if not clicked:
                break

    def _turn_prejoin_control_off(self, page, name: str) -> bool:
        """'Turn off X' label => X is on (click once). 'Turn on X' => already off."""
        for pattern in (re.compile(rf"turn off {name}", re.I), re.compile(rf"turn on {name}", re.I),
                        re.compile(name, re.I)):
            try:
                loc = page.get_by_role("button", name=pattern).first
                loc.wait_for(state="visible", timeout=2500)
                label = f"{loc.get_attribute('aria-label') or ''} {loc.get_attribute('title') or ''}".lower()
                if f"turn on {name}" in label:
                    print(f"[*] {name.capitalize()} already off.")
                    return True
                loc.click(timeout=2500)
                page.wait_for_timeout(500)
                print(f"[*] Turned off {name}.")
                return True
            except Exception:
                continue
        print(f"[!] Could not find the {name} toggle.")
        return False

    def _disable_microphone_and_camera(self, page):
        print("[*] Muting bot camera and microphone...")
        try:
            page.locator("body").click(timeout=2000)
            page.keyboard.press("Control+d")
            page.wait_for_timeout(400)
            page.keyboard.press("Control+e")
            page.wait_for_timeout(400)
        except Exception as exc:
            print(f"[!] Could not mute via shortcut: {exc}")
        self._turn_prejoin_control_off(page, "microphone")
        self._turn_prejoin_control_off(page, "camera")

    def _enter_guest_name(self, page):
        for sel in ['input[aria-label="Your name"]', 'input[placeholder="Your name"]', 'input[type="text"]']:
            try:
                inp = page.locator(sel).first
                inp.wait_for(state="visible", timeout=1500)
                inp.fill("AI Meeting Bot", timeout=3000)
                print("[*] Entered bot name.")
                return
            except Exception:
                continue

    def _click_join_button(self, page) -> bool:
        for sel in ['button:has-text("Ask to join")', 'button:has-text("Join now")',
                    '[role="button"]:has-text("Ask to join")', '[role="button"]:has-text("Join now")']:
            try:
                btn = page.locator(sel).first
                btn.wait_for(state="visible", timeout=5000)
                btn.click(timeout=4000)
                print("[*] Clicked join button.")
                return True
            except Exception:
                continue
        return False

    def _wait_until_joined(self, page):
        for sel in LEAVE_SELECTORS:
            try:
                page.wait_for_selector(sel, state="visible", timeout=120000)
                return
            except PlaywrightTimeoutError:
                continue
        body = page.locator("body").inner_text(timeout=5000)
        if "can't join this video call" in body.lower():
            raise RuntimeError("Google Meet rejected the join request.")
        raise RuntimeError("Timed out waiting to be admitted to the meeting.")

    def _leave_meeting(self, page):
        print("[*] Leaving meeting...")
        for sel in LEAVE_SELECTORS:
            try:
                btn = page.locator(sel).first
                btn.wait_for(state="visible", timeout=1500)
                btn.click(timeout=2500)
                page.wait_for_timeout(1500)
                return
            except Exception:
                continue

    # ── Main flow ──────────────────────────────────────────────────────────
    def run(self):
        print(f"[*] Launching bot for: {self.meeting_url}")
        audio_thread = monitor_thread = None
        with sync_playwright() as p:
            # NOTE: do NOT add --use-fake-device-for-media-stream: it reroutes Meet's audio
            # to a fake device and the loopback recorder captures silence.
            browser = p.chromium.launch(headless=False, args=[
                "--use-fake-ui-for-media-stream",
                "--disable-blink-features=AutomationControlled"])
            context = browser.new_context(permissions=["camera", "microphone"])
            page = context.new_page()
            try:
                page.goto(self.meeting_url, wait_until="domcontentloaded", timeout=120000)
                page.wait_for_timeout(5000)
                self._dismiss_prejoin_popups(page)
                self._disable_microphone_and_camera(page)
                self._enter_guest_name(page)
                self._dismiss_prejoin_popups(page)
                if not self._click_join_button(page):
                    print("[!] Join button not found — click it manually in the browser.")
                print("[*] Waiting to be admitted...")
                self._wait_until_joined(page)
                print("[*] Joined the meeting!")
                page.wait_for_timeout(3000)

                audio_thread = threading.Thread(target=self._record_audio, daemon=True)
                audio_thread.start()
                if not self.recording_started.wait(timeout=15):
                    raise RuntimeError("Audio recorder did not start within 15 s.")
                if self.recording_error is not None:
                    raise RuntimeError(f"Recorder failed: {self.recording_error}")

                monitor_thread = threading.Thread(target=self._peak_monitor, daemon=True)
                monitor_thread.start()
                print("\n" + "=" * 55)
                print("  MEETING IS BEING RECORDED.")
                print("  Press Enter to STOP recording and begin transcription.")
                print("=" * 55 + "\n")
                input()
            except Exception as exc:
                print(f"\n[!] Error: {exc}")
            finally:
                self.stop_recording.set()
                if audio_thread:
                    audio_thread.join(timeout=10)
                if monitor_thread:
                    monitor_thread.join(timeout=3)
                try:
                    self._leave_meeting(page)
                except Exception:
                    pass
                browser.close()
                self._process_meeting()

    # ── Post-processing ────────────────────────────────────────────────────
    def _process_meeting(self):
        if self.recording_error is not None:
            print("[!] Skipping transcription — recording failed.")
            return
        if not os.path.exists(self.audio_file) or os.path.getsize(self.audio_file) < 1000:
            print("[!] No usable audio file — nothing to transcribe.")
            return

        engine = sarvam_transcribe if self.transcription_engine == "sarvam" else cloudtranscribe
        print("\n[*] Running cloud transcription...")
        try:
            engine.process_single_file(self.audio_file)
        except Exception as exc:
            print(f"[!] Transcription failed: {exc}")
            return

        transcript_file = engine.make_output_name(self.audio_file)
        if not os.path.exists(transcript_file):
            print("[!] Transcript file was not generated.")
            return

        content = Path(transcript_file).read_text(encoding="utf-8")
        participants = self._extract_participants(content)
        json_target = self.documents_dir / f"{Path(transcript_file).stem}.json"
        title = f"Meeting {self.started_at.strftime('%Y-%m-%d %H:%M')}"

        payload = {
            "title": title, "content": content,
            "source_audio": Path(self.audio_file).name,
            "source_transcript": Path(transcript_file).name,
            "type": "meeting_transcript", "meeting_id": self.meeting_id,
            "meeting_url": self.meeting_url, "started_at": self.started_at.isoformat(),
            "participants": participants, "transcription_engine": self.transcription_engine,
        }
        json_target.write_text(json.dumps(payload, indent=4), encoding="utf-8")

        write_meeting_metadata(self.meeting_dir, {
            "meeting_id": self.meeting_id, "title": title, "meeting_url": self.meeting_url,
            "started_at": self.started_at.isoformat(),
            "audio_file": Path(self.audio_file).name,
            "transcript_file": Path(transcript_file).name,
            "document_json": str(json_target.relative_to(self.meeting_dir)),
            "vector_store_dir": str(self.vector_store_dir),
            "collection_name": self.collection_name,
            "transcript_preview": content[:300],
            "participants": participants,
            "transcription_engine": self.transcription_engine,
        })

        print("[*] Ingesting transcript into this meeting's vector store...")
        ingest(pdf_dir=None, json_dir=str(self.documents_dir),
               persist_dir=str(self.vector_store_dir),
               collection_name=self.collection_name, reset_collection=True)
        print("\n*** Done. Run `streamlit run app2.py` to chat with your meeting ***\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Join a Google Meet, record, transcribe, and ingest it.")
    ap.add_argument("meeting_url", help="Google Meet URL")
    ap.add_argument("--transcriber", choices=["whisper", "sarvam"], default=None)
    return ap.parse_args()


def choose_transcriber(choice: str | None) -> str:
    if choice:
        return choice
    print("Choose transcription engine:\n1. whisper\n2. sarvam")
    while True:
        c = input("Enter 1 or 2 [default 1]: ").strip().lower()
        if c in {"", "1", "whisper"}:
            return "whisper"
        if c in {"2", "sarvam"}:
            return "sarvam"


if __name__ == "__main__":
    args = parse_args()
    engine = choose_transcriber(args.transcriber)
    print(f"[*] Selected transcription engine: {engine}")
    print("Tip: if the meter stays empty, set PREFERRED_LOOPBACK_DEVICE at the top of this file.")
    MeetingBot(args.meeting_url, transcription_engine=engine).run()
