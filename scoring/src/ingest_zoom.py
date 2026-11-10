"""Phase 1 data source: pull Zoom Meetings audio + Zoom's own VTT transcript
+ AI Companion summaries. Mirrors ingest_genesys.py.

For each meeting in the date window, persists:

  <AUDIO_DIR>/zoom/<safe_uuid>.m4a              # the audio_only file
  data/transcripts/zoom/<safe_uuid>.vtt          # audio_transcript (if present)
  data/zoom_extras/<safe_uuid>.summary.json      # AI summary (if present)
  data/zoom_extras/<safe_uuid>.next_steps.json   # AI next steps (if present)

CSV outputs:
  data/zoom_metadata.csv   # one row per meeting (idempotent: re-runs skip
                           #   meeting_uuids already present)
  data/labels.csv          # appended with source=zoom rows

`safe_uuid` is URL-safe-base64 ('+' -> '-', '/' -> '_', '=' stripped); the
join key remains the original `meeting_uuid` (column in zoom_metadata.csv).

Usage:
  python -m src.ingest_zoom --start 2026-04-24 --end 2026-05-20
  python -m src.ingest_zoom --start 2026-04-24 --end 2026-05-20 --max-meetings 5
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from . import config
from .zoom_client import ZoomClient, ZoomError, safe_uuid

META_FIELDS = [
    "meeting_uuid", "meeting_id", "topic", "host_id", "host_email",
    "start_time", "duration_min", "audio_file", "audio_bytes",
    "transcript_file", "transcript_bytes",
    "summary_file", "next_steps_file",
    "recording_count", "total_size", "source", "call_type",
]

# Host -> call_type. booker@example.com runs appointment-setting (booking) calls;
# the ZoomItaN accounts run the sales consultations. Kept in sync with
# build_calls_csv._classify_zoom so zoom_metadata.csv and calls.csv agree.
_BOOKER_EMAILS = {"booker@example.com"}


def _call_type(host_email: str) -> str:
    return "booking" if (host_email or "").lower() in _BOOKER_EMAILS else "sales"
LABEL_FIELDS = [
    "call_id", "rep_loi", "rep_reasoning", "conversion_outcome",
    "rep_id", "call_date", "source",
]

# Recording-file types we care about (everything else — timeline, chat — skipped)
WANT_TYPES = {
    "audio_only": ("audio", "m4a"),
    "audio_transcript": ("transcript", "vtt"),
    "summary": ("summary", "json"),
    "summary_next_steps": ("next_steps", "json"),
}


def _log(msg: str, fh) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def _load_done(meta_csv: Path) -> set[str]:
    if not meta_csv.exists():
        return set()
    with meta_csv.open(newline="", encoding="utf-8") as f:
        return {row["meeting_uuid"] for row in csv.DictReader(f)}


def _append_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="window start YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="window end YYYY-MM-DD (inclusive)")
    ap.add_argument("--max-meetings", type=int, default=0,
                    help="stop after N meetings with a downloaded audio (0=no cap)")
    ap.add_argument("--transcripts-only", action="store_true",
                    help="skip audio downloads; register a meeting off its VTT "
                         "(audio reused from disk when present, else audio_bytes=0 "
                         "for a later audio backfill)")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    audio_root = config.AUDIO_DIR / "zoom"
    transcript_root = config.DATA / "transcripts" / "zoom"
    extras_root = config.DATA / "zoom_extras"
    for d in (audio_root, transcript_root, extras_root):
        d.mkdir(parents=True, exist_ok=True)

    zoom_meta_csv = config.DATA / "zoom_metadata.csv"
    done = _load_done(zoom_meta_csv)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = config.OUT_LOGS / f"ingest_zoom_{ts}.log"

    zc = ZoomClient()
    meta_rows: list[dict] = []
    label_rows: list[dict] = []
    scanned = seen = skip_noaudio = skip_ghost = downloaded = 0

    with log_path.open("w", encoding="utf-8") as fh:
        def _flush_buffers(reason: str) -> None:
            """Persist progress so a long audio pull can resume after timeout."""
            if not meta_rows:
                return
            count = len(meta_rows)
            _append_rows(zoom_meta_csv, META_FIELDS, meta_rows)
            _append_rows(config.LABELS_CSV, LABEL_FIELDS, label_rows)
            meta_rows.clear()
            label_rows.clear()
            _log(f"CHECKPOINT {reason}: committed={count}", fh)

        _log(f"window={args.start}..{args.end} already_done={len(done)} "
             f"max={args.max_meetings or 'all'}", fh)
        try:
            for m in zc.iter_account_recordings(args.start, args.end):
                if args.max_meetings and downloaded >= args.max_meetings:
                    break
                scanned += 1
                uuid = m.get("uuid", "")
                if not uuid or uuid in done:
                    continue
                done.add(uuid)
                seen += 1

                files = m.get("recording_files") or []
                files_by_type: dict[str, dict] = {}
                for f in files:
                    t = f.get("recording_type")
                    if t in WANT_TYPES and f.get("file_size", 0) > 0 \
                            and f.get("download_url"):
                        files_by_type.setdefault(t, f)

                audio = files_by_type.get("audio_only")
                if not audio:
                    skip_noaudio += 1
                    continue
                if audio.get("status") and audio["status"] != "completed":
                    skip_ghost += 1
                    _log(f"SKIP {uuid} audio status={audio['status']}", fh)
                    continue

                su = safe_uuid(uuid)

                def _get(url: str, dest: Path, kind: str,
                         required: bool) -> int | None:
                    """Download to dest unless dest already exists with content.
                    Returns size in bytes, or None on failure.
                    """
                    if dest.exists() and dest.stat().st_size > 0:
                        size = dest.stat().st_size
                        _log(f"SKIP-DL {uuid} {kind} already on disk ({size}B)", fh)
                        return size
                    try:
                        return zc.download(url, dest)
                    except Exception as e:  # noqa: BLE001
                        level = "SKIP" if required else "WARN"
                        _log(f"{level} {uuid} {kind} download: "
                             f"{type(e).__name__}: {str(e)[:120]}", fh)
                        return None

                # Audio (required, unless --transcripts-only)
                audio_dest = audio_root / f"{su}.m4a"
                if args.transcripts_only:
                    audio_bytes = (audio_dest.stat().st_size
                                   if audio_dest.exists()
                                   and audio_dest.stat().st_size > 0 else 0)
                else:
                    audio_bytes = _get(audio["download_url"], audio_dest,
                                       "audio", required=True)
                    if audio_bytes is None:
                        skip_ghost += 1
                        continue

                # Transcript / summaries (optional; meeting still counted on audio)
                transcript_bytes = 0
                transcript_rel = ""
                t_file = files_by_type.get("audio_transcript")
                if t_file:
                    t_dest = transcript_root / f"{su}.vtt"
                    n = _get(t_file["download_url"], t_dest,
                             "transcript", required=False)
                    if n is not None:
                        transcript_bytes = n
                        transcript_rel = f"transcripts/zoom/{su}.vtt"

                # In transcripts-only mode a meeting must have SOMETHING scoreable
                # or reusable; otherwise leave it unregistered so a normal (audio)
                # run can still pick it up later.
                if args.transcripts_only and not audio_bytes \
                        and not transcript_bytes:
                    skip_ghost += 1
                    _log(f"SKIP {uuid} transcripts-only: no vtt, no local audio", fh)
                    continue

                summary_rel = ""
                s_file = files_by_type.get("summary")
                if s_file:
                    s_dest = extras_root / f"{su}.summary.json"
                    if _get(s_file["download_url"], s_dest,
                            "summary", required=False) is not None:
                        summary_rel = f"zoom_extras/{su}.summary.json"

                next_rel = ""
                n_file = files_by_type.get("summary_next_steps")
                if n_file:
                    n_dest = extras_root / f"{su}.next_steps.json"
                    if _get(n_file["download_url"], n_dest,
                            "next_steps", required=False) is not None:
                        next_rel = f"zoom_extras/{su}.next_steps.json"

                row = {
                    "meeting_uuid": uuid,
                    "meeting_id": m.get("id", ""),
                    "topic": (m.get("topic") or "").replace("\n", " ").replace("\r", " "),
                    "host_id": m.get("host_id", ""),
                    "host_email": m.get("host_email", ""),
                    "start_time": m.get("start_time", ""),
                    "duration_min": m.get("duration", 0),
                    "audio_file": f"zoom/{su}.m4a" if audio_bytes else "",
                    "audio_bytes": audio_bytes,
                    "transcript_file": transcript_rel,
                    "transcript_bytes": transcript_bytes,
                    "summary_file": summary_rel,
                    "next_steps_file": next_rel,
                    "recording_count": m.get("recording_count", len(files)),
                    "total_size": m.get("total_size", 0),
                    "source": "zoom",
                    "call_type": _call_type(m.get("host_email", "")),
                }
                meta_rows.append(row)
                label_rows.append({
                    "call_id": uuid,
                    "rep_loi": "", "rep_reasoning": "", "conversion_outcome": "",
                    "rep_id": m.get("host_email", ""),
                    "call_date": (m.get("start_time") or "")[:10],
                    "source": "zoom",
                })
                downloaded += 1
                _log(f"OK   {uuid} host={m.get('host_email','?')[:30]} "
                     f"dur={m.get('duration', 0)}min audio={audio_bytes}B "
                     f"vtt={transcript_bytes}B summary={'Y' if summary_rel else 'N'} "
                     f"({downloaded}/{args.max_meetings or '?'})", fh)
                if len(meta_rows) >= 25:
                    _flush_buffers("periodic")
        finally:
            _flush_buffers("final")
            _log(f"DONE scanned={scanned} new_meetings={seen} "
                 f"skip_no_audio={skip_noaudio} skip_ghost={skip_ghost} "
                 f"downloaded={downloaded}", fh)
            _log(f"metadata -> {zoom_meta_csv}", fh)
            _log(f"labels   -> {config.LABELS_CSV} (source=zoom rows appended)", fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
