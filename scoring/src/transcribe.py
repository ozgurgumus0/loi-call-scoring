"""Phase 2 — two-track transcription.

- Zoom calls that have a VTT file: parse the VTT (already Italian, already
  diarized via speaker labels). No Whisper needed.
- Genesys calls (and the ~80 Zoom calls without VTT): faster-whisper
  large-v3, Italian-locked. Needs ffmpeg on PATH.

Output is uniform: outputs/transcripts/<source>__<call_id>.txt and .json.
The .json has segments [{start, end, speaker, text}, ...] so downstream
scoring can reason about turns.

Idempotent: skips calls whose .txt already exists in outputs/transcripts/.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import (
    AUDIO_DIR, CALLS_CSV, DATA, GENESYS_TRANSCRIPT_DIR, OUT_LOGS,
    OUT_TRANSCRIPTS, WHISPER_LANG, WHISPER_MODEL, ZOOM_VTT_DIR, ensure_dirs,
)

log = logging.getLogger("transcribe")

MIN_TRANSCRIPT_CHARS = 100  # below this, transcript is useless for LOI scoring


# ---------------------------------------------------------------- types ---
@dataclass
class CallRow:
    source: str
    call_id: str
    audio_file: str | None
    transcript_file: str | None  # path relative to data/, if Phase 1 set it
    duration_sec: int | None


# ---------------------------------------------------------------- I/O ----
def iter_calls() -> Iterator[CallRow]:
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                dur = int(row["duration_sec"]) if row.get("duration_sec") else None
            except ValueError:
                dur = None
            yield CallRow(
                source=row["source"],
                call_id=row["call_id"],
                audio_file=(row.get("audio_file") or None),
                transcript_file=(row.get("transcript_file") or None),
                duration_sec=dur,
            )


def load_existing_transcript(row: CallRow) -> tuple[str, list[dict]] | None:
    """Return (text, segments) if Phase 1 already produced a transcript.

    Genesys: data/transcripts/genesys/<call_id>.txt — diarized with
             [Agente]/[Cliente] line prefixes (so 'segments' is one entry
             per line with the speaker tag stripped into its own field).
    Zoom:    data/<row.transcript_file> — VTT with speaker labels.
    """
    if row.source == "genesys":
        path = GENESYS_TRANSCRIPT_DIR / f"{row.call_id}.txt"
        if path.exists() and path.stat().st_size >= MIN_TRANSCRIPT_CHARS:
            text = path.read_text(encoding="utf-8", errors="replace")
            segments: list[dict] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                speaker = None
                content = line
                if line.startswith("[") and "]" in line:
                    end = line.index("]")
                    speaker = line[1:end]
                    content = line[end + 1:].strip()
                segments.append({"start": None, "end": None,
                                 "speaker": speaker, "text": content})
            return text, segments
        return None

    if row.source == "zoom" and row.transcript_file:
        path = DATA / row.transcript_file
        if path.exists() and path.stat().st_size >= MIN_TRANSCRIPT_CHARS:
            try:
                return _parse_vtt(path)
            except Exception as e:  # noqa: BLE001
                log.warning("VTT parse failed for %s: %s", path, e)
                return None
    return None


def _out_paths(source: str, call_id: str) -> tuple[Path, Path]:
    stem = f"{source}__{call_id}"
    return OUT_TRANSCRIPTS / f"{stem}.txt", OUT_TRANSCRIPTS / f"{stem}.json"


# ------------------------------------------------------------- Zoom VTT --
def _parse_vtt(vtt_path: Path) -> tuple[str, list[dict]]:
    """Parse a Zoom VTT into (plain_text, segments).

    Zoom VTT speaker labels look like:  "Serena Pisano: Buongiorno..."
    """
    import webvtt  # local import — only needed in Phase 2+

    segments: list[dict] = []
    lines: list[str] = []
    for cap in webvtt.read(str(vtt_path)):
        text = cap.text.replace("\n", " ").strip()
        speaker = None
        if ": " in text and len(text.split(": ", 1)[0]) < 40:
            speaker, text = text.split(": ", 1)
        segments.append({
            "start": cap.start_in_seconds,
            "end": cap.end_in_seconds,
            "speaker": speaker,
            "text": text,
        })
        prefix = f"{speaker}: " if speaker else ""
        lines.append(prefix + text)
    return "\n".join(lines), segments


# ------------------------------------------------------------- Whisper ---
_WHISPER_MODEL_CACHE: dict[str, object] = {}


def _get_whisper():
    if "model" not in _WHISPER_MODEL_CACHE:
        from faster_whisper import WhisperModel  # local import
        # CPU default; switch device="cuda" if a GPU is present.
        _WHISPER_MODEL_CACHE["model"] = WhisperModel(
            WHISPER_MODEL, device="auto", compute_type="auto",
        )
    return _WHISPER_MODEL_CACHE["model"]


def _whisper_transcribe(audio_path: Path) -> tuple[str, list[dict]]:
    model = _get_whisper()
    segments_iter, _info = model.transcribe(
        str(audio_path),
        language=WHISPER_LANG,
        vad_filter=True,
        word_timestamps=True,
    )
    segments: list[dict] = []
    lines: list[str] = []
    for seg in segments_iter:
        text = seg.text.strip()
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "speaker": None,  # diarization is Phase 3 for Genesys
            "text": text,
        })
        lines.append(text)
    return "\n".join(lines), segments


# ---------------------------------------------------------------- main ---
def transcribe_one(row: CallRow) -> tuple[str, str]:
    """Returns (track, status) where track in
    {"existing","whisper","skip-no-source"} and status in
    {"done","cached","error:<msg>"}.
    """
    txt_out, json_out = _out_paths(row.source, row.call_id)
    if txt_out.exists() and json_out.exists():
        return ("cached", "cached")

    # Track A: a Phase-1 transcript already exists (Zoom VTT or Genesys .txt).
    loaded = load_existing_transcript(row)
    if loaded is not None:
        text, segs = loaded
        txt_out.write_text(text, encoding="utf-8")
        json_out.write_text(json.dumps(segs, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        return ("existing", "done")

    # Track B: Whisper on local audio (only the ~8% without a Phase-1 transcript).
    if not row.audio_file:
        return ("skip-no-source", "error:no audio_file in calls.csv")
    audio = AUDIO_DIR / row.audio_file if not Path(row.audio_file).is_absolute() \
        else Path(row.audio_file)
    if not audio.exists():
        # try the per-source subdir layout (e.g. AUDIO_DIR/zoom/<id>.m4a)
        alt = AUDIO_DIR / row.source / Path(row.audio_file).name
        if alt.exists():
            audio = alt
        else:
            return ("skip-no-source", f"error:audio missing: {audio}")
    try:
        text, segs = _whisper_transcribe(audio)
    except Exception as e:  # noqa: BLE001
        return ("whisper", f"error:{e}")
    txt_out.write_text(text, encoding="utf-8")
    json_out.write_text(json.dumps(segs, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return ("whisper", "done")


def run(source: str | None = None, limit: int | None = None,
        skip_whisper: bool = False) -> dict:
    ensure_dirs()
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    logfile = OUT_LOGS / f"transcribe_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(logfile, encoding="utf-8"),
                  logging.StreamHandler()],
        force=True,
    )

    counts = {"existing": 0, "whisper": 0, "cached": 0,
              "skip-no-source": 0, "error": 0}
    t0 = time.monotonic()
    audio_sec = 0
    for i, row in enumerate(iter_calls()):
        if source and row.source != source:
            continue
        if limit is not None and i >= limit:
            break
        if skip_whisper and row.source == "genesys":
            counts["skip-no-source"] += 1
            continue
        track, status = transcribe_one(row)
        if status.startswith("error"):
            counts["error"] += 1
            log.warning("%s/%s [%s] %s", row.source, row.call_id, track, status)
            continue
        counts[track] = counts.get(track, 0) + 1
        if status == "done" and row.duration_sec:
            audio_sec += row.duration_sec
        if i and i % 50 == 0:
            log.info("progress %d calls — counts=%s", i, counts)

    wall = time.monotonic() - t0
    log.info("done in %.1fs — counts=%s audio_processed=%.1fh throughput=%.1fx",
             wall, counts, audio_sec / 3600, (audio_sec / wall) if wall else 0)
    return counts


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("genesys", "zoom"),
                    help="Restrict to one source.")
    ap.add_argument("--limit", type=int, help="Max rows to process.")
    ap.add_argument("--skip-whisper", action="store_true",
                    help="Only run the Zoom VTT track; skip anything that needs Whisper.")
    args = ap.parse_args()
    run(source=args.source, limit=args.limit, skip_whisper=args.skip_whisper)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
