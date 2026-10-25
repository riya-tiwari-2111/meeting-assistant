import json
from pathlib import Path

MEETINGS_ROOT = Path(__file__).parent / "data" / "meetings"


def get_meeting_dir(meeting_id: str) -> Path:
    return MEETINGS_ROOT / meeting_id


def write_meeting_metadata(meeting_dir, metadata: dict) -> None:
    with open(Path(meeting_dir) / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=4)


def list_meetings() -> list[dict]:
    meetings = []
    for meta in MEETINGS_ROOT.glob("*/metadata.json"):
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                meetings.append(json.load(fh))
        except Exception:
            continue
    return sorted(meetings, key=lambda m: m.get("started_at", ""), reverse=True)
