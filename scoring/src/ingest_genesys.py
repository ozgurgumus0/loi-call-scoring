"""Phase 1 data source: pull voice-call audio + metadata from Genesys Cloud.

The workflow doc assumed a local audio path; the real source is Genesys, so this
module replaces "point at a folder". It produces exactly what the rest of the
pipeline expects:

  data/audio/<conversation_id>.webm        # one mono Opus file per call
  data/genesys_metadata.csv                # call_id + call metadata (no PII)
  data/labels.csv                          # SKELETON: call_id/rep_id/call_date
                                           #   rep_loi / rep_reasoning /
                                           #   conversion_outcome are NOT in
                                           #   Genesys -> fill from survey/CRM.

conversation_id is used as call_id everywhere — the natural join key to labels.

Idempotent: re-running skips conversations already in genesys_metadata.csv.
No PII in logs or metadata: GUIDs only, never phone numbers / names / transcript.

Usage:
  python -m src.ingest_genesys --days 21 --max-calls 25 --min-sec 60
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from . import config
from .genesys_client import GenesysClient

META_FIELDS = [
    "conversation_id", "conversation_start", "conversation_end", "duration_sec",
    "direction", "agent_id", "queue_id", "wrapup_code",
    "recording_id", "recording_file_state", "n_audio_tracks",
    "audio_file", "audio_bytes",
]
LABEL_FIELDS = [
    "call_id", "rep_loi", "rep_reasoning", "conversion_outcome",
    "rep_id", "call_date",
]


def _log(msg: str, fh) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()}  {msg}"
    print(line)
    fh.write(line + "\n")
    fh.flush()


def _parse_iso(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_meta(conv: dict) -> dict:
    start = _parse_iso(conv.get("conversationStart"))
    end = _parse_iso(conv.get("conversationEnd"))
    dur = round((end - start).total_seconds()) if start and end else 0
    agent_id = queue_id = wrapup = ""
    has_agent = False
    for p in conv.get("participants", []):
        if p.get("purpose") == "agent":
            has_agent = True
            agent_id = agent_id or p.get("userId", "")
        for s in p.get("sessions", []):
            for seg in s.get("segments", []):
                queue_id = queue_id or seg.get("queueId", "")
                wrapup = wrapup or seg.get("wrapUpCode", "")
    return {
        "conversation_id": conv.get("conversationId", ""),
        "conversation_start": conv.get("conversationStart", ""),
        "conversation_end": conv.get("conversationEnd", ""),
        "duration_sec": dur,
        "direction": conv.get("originatingDirection", ""),
        "agent_id": agent_id,
        "queue_id": queue_id,
        "wrapup_code": wrapup,
        "_has_agent": has_agent,
    }


def _load_done(meta_csv: Path) -> set[str]:
    if not meta_csv.exists():
        return set()
    with meta_csv.open(newline="", encoding="utf-8") as f:
        return {row["conversation_id"] for row in csv.DictReader(f)}


def _append_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21,
                    help="look back this many days from now")
    ap.add_argument("--max-calls", type=int, default=25,
                    help="stop after N calls with a downloaded recording")
    ap.add_argument("--min-sec", type=int, default=60,
                    help="skip calls shorter than this (no-answer/IVR)")
    ap.add_argument("--start", help="override window start (ISO8601 Z)")
    ap.add_argument("--end", help="override window end (ISO8601 Z)")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    now = dt.datetime.now(dt.timezone.utc)
    end_iso = args.end or now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    start_iso = args.start or (
        (now - dt.timedelta(days=args.days)).strftime("%Y-%m-%dT00:00:00.000Z"))

    ts = now.strftime("%Y%m%d_%H%M%S")
    log_path = config.OUT_LOGS / f"ingest_{ts}.log"
    done = _load_done(config.GENESYS_METADATA_CSV)

    gc = GenesysClient()
    meta_rows: list[dict] = []
    label_rows: list[dict] = []
    seen = scanned = skip_short = skip_norec = skip_archived = downloaded = 0

    with log_path.open("w", encoding="utf-8") as fh:
        _log(f"org={gc.org_name()!r} window={start_iso}..{end_iso} "
             f"already_done={len(done)} target={args.max_calls}", fh)
        try:
            for conv in gc.iter_voice_conversations(start_iso, end_iso):
                if downloaded >= args.max_calls:
                    break
                scanned += 1
                m = _extract_meta(conv)
                cid = m["conversation_id"]
                if not cid or cid in done:
                    continue
                done.add(cid)
                seen += 1
                if not m["_has_agent"] or m["duration_sec"] < args.min_sec:
                    skip_short += 1
                    continue
                try:
                    recs = gc.get_recordings(cid)
                except Exception as e:  # noqa: BLE001
                    _log(f"SKIP {cid} recordings error: {type(e).__name__}", fh)
                    skip_norec += 1
                    continue
                rec = next((r for r in recs if r.get("media") == "audio"), None)
                if rec is None:
                    skip_norec += 1
                    continue
                if rec.get("fileState") not in ("AVAILABLE", "RESTORED"):
                    skip_archived += 1
                    _log(f"SKIP {cid} fileState={rec.get('fileState')}", fh)
                    continue
                url, n_tracks = gc.audio_uri(rec)
                if not url:
                    skip_norec += 1
                    _log(f"SKIP {cid} no mediaUri", fh)
                    continue
                dest = config.AUDIO_DIR / f"{cid}.webm"
                if dest.exists() and dest.stat().st_size > 0:
                    nbytes = dest.stat().st_size
                    _log(f"SKIP-DL {cid} already on disk ({nbytes}B)", fh)
                else:
                    try:
                        nbytes = gc.download(url, dest)
                    except Exception as e:  # noqa: BLE001
                        _log(f"SKIP {cid} download error: {type(e).__name__}", fh)
                        skip_norec += 1
                        continue

                m.update(recording_id=rec.get("id", ""),
                         recording_file_state=rec.get("fileState", ""),
                         n_audio_tracks=n_tracks,
                         audio_file=dest.name, audio_bytes=nbytes)
                m.pop("_has_agent", None)
                meta_rows.append(m)
                label_rows.append({
                    "call_id": cid, "rep_loi": "", "rep_reasoning": "",
                    "conversion_outcome": "", "rep_id": m["agent_id"],
                    "call_date": (m["conversation_start"] or "")[:10],
                })
                downloaded += 1
                _log(f"OK   {cid} dur={m['duration_sec']}s "
                     f"tracks={n_tracks} bytes={nbytes} ({downloaded}/{args.max_calls})",
                     fh)
        finally:
            if meta_rows:
                _append_rows(config.GENESYS_METADATA_CSV, META_FIELDS, meta_rows)
            if label_rows:
                _append_rows(config.LABELS_CSV, LABEL_FIELDS, label_rows)
            _log(f"DONE scanned={scanned} candidates={seen} "
                 f"skip_short_or_noagent={skip_short} skip_no_recording={skip_norec} "
                 f"skip_archived={skip_archived} downloaded={downloaded}", fh)
            _log(f"metadata -> {config.GENESYS_METADATA_CSV}", fh)
            _log(f"labels skeleton -> {config.LABELS_CSV} "
                 f"(rep_loi/rep_reasoning/conversion_outcome MUST come from "
                 f"survey/CRM, joined on call_id)", fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
