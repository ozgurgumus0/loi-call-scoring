"""Merge per-source ingest outputs into the consolidated data/calls.csv.

The scoring pipeline reads `calls.csv` (one row per (source, call_id)) as
its scoreable corpus. The two ingest scripts write to per-source files:

    src.ingest_genesys -> data/genesys_metadata.csv  +  data/labels.csv
    src.ingest_zoom    -> data/zoom_metadata.csv     +  data/labels.csv

This module joins them into the calls.csv schema (32 cols), enriches Genesys
rows with queue_name/agent_name from data/omnibus_metadata.csv when present,
and appends only call_ids not already in calls.csv. Historic rows are never
overwritten — re-runs are safe.

Label fields (rep_loi / rep_reasoning / conversion_outcome) are pulled from
labels.csv when they're non-empty, even for rows that already exist in
calls.csv — so once sales ops delivers labels they flow through cleanly.

Usage:
    python -m src.build_calls_csv               # in-place rebuild/append
    python -m src.build_calls_csv --dry-run     # show counts, write nothing
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from .config import (
    CALLS_CSV, DATA, GENESYS_METADATA_CSV, LABELS_CSV, OMNIBUS_METADATA_CSV,
    ZOOM_METADATA_CSV,
)

# The canonical column order for calls.csv (must match the existing file).
CALLS_COLS = [
    "source", "call_id", "call_date", "start_time", "duration_sec",
    "direction", "rep_id", "agent_name", "call_type",
    "audio_file", "audio_bytes",
    "transcript_file", "summary_file",
    "queue_id", "queue_name", "wrapup_code",
    "recording_id", "recording_file_state", "n_audio_tracks",
    "recording_start", "recording_end",
    "meeting_id", "topic", "host_id", "host_email",
    "transcript_bytes", "next_steps_file", "recording_count", "total_size",
    "rep_loi", "rep_reasoning", "conversion_outcome", "contract_signed",
]

# Zoom host -> call_type heuristic (based on observed data in the existing
# calls.csv: booker@example.com runs booking calls; zoomItaN are sales).
_ZOOM_BOOKER_EMAILS = {"booker@example.com"}


@dataclass
class MergeResult:
    total: int          # rows in calls.csv after merge
    added: int          # new (source, call_id) rows inserted
    label_updates: int  # historic rows whose label cells got filled in
    skipped_seen: int   # ingest rows ignored because already present


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _index_by(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    return {tuple(r.get(k, "") for k in keys): r for r in rows}


def _classify_zoom(host_email: str) -> str:
    return "booking" if (host_email or "").lower() in _ZOOM_BOOKER_EMAILS else "sales"


def _build_genesys_row(meta: dict, label: dict, omnibus: dict | None) -> dict:
    cid = meta.get("conversation_id", "")
    start = meta.get("conversation_start", "")
    # Enrich queue_name / agent_name / call_type from omnibus when we have it
    queue_name = (omnibus or {}).get("queue_name", "") or ""
    agent_name = (omnibus or {}).get("agent_name", "") or ""
    # call_type stays empty unless we can infer; preserves rows added later
    # by the build_omnibus step (out of scope here).
    transcript_file = ""
    tpath = DATA / "transcripts" / "genesys" / f"{cid}.txt"
    if tpath.exists():
        transcript_file = f"transcripts/genesys/{cid}.txt"
    return {
        "source":              "genesys",
        "call_id":             cid,
        "call_date":           start[:10] if start else "",
        "start_time":          start,
        "duration_sec":        meta.get("duration_sec", ""),
        "direction":           meta.get("direction", ""),
        "rep_id":              meta.get("agent_id", ""),
        "agent_name":          agent_name,
        "call_type":           "",  # filled later by the queue-mapping step
        "audio_file":          meta.get("audio_file", ""),
        "audio_bytes":         meta.get("audio_bytes", ""),
        "transcript_file":     transcript_file,
        "summary_file":        "",
        "queue_id":            meta.get("queue_id", ""),
        "queue_name":          queue_name,
        "wrapup_code":         meta.get("wrapup_code", ""),
        "recording_id":        meta.get("recording_id", ""),
        "recording_file_state": meta.get("recording_file_state", ""),
        "n_audio_tracks":      meta.get("n_audio_tracks", ""),
        "recording_start":     "",
        "recording_end":       "",
        "meeting_id":          "",
        "topic":               "",
        "host_id":             "",
        "host_email":          "",
        "transcript_bytes":    "",
        "next_steps_file":     "",
        "recording_count":     "",
        "total_size":          "",
        "rep_loi":             (label or {}).get("rep_loi", ""),
        "rep_reasoning":       (label or {}).get("rep_reasoning", ""),
        "conversion_outcome":  (label or {}).get("conversion_outcome", ""),
    }


def _build_zoom_row(meta: dict, label: dict) -> dict:
    uuid = meta.get("meeting_uuid", "")
    start = meta.get("start_time", "")
    dur_min = meta.get("duration_min") or "0"
    try:
        dur_sec = int(float(dur_min)) * 60
    except ValueError:
        dur_sec = 0
    return {
        "source":              "zoom",
        "call_id":             uuid,
        "call_date":           start[:10] if start else "",
        "start_time":          start,
        "duration_sec":        dur_sec,
        "direction":           "",
        "rep_id":              meta.get("host_email", ""),
        "agent_name":          "",
        "call_type":           _classify_zoom(meta.get("host_email", "")),
        "audio_file":          meta.get("audio_file", ""),
        "audio_bytes":         meta.get("audio_bytes", ""),
        "transcript_file":     meta.get("transcript_file", ""),
        "summary_file":        meta.get("summary_file", ""),
        "queue_id":            "",
        "queue_name":          "",
        "wrapup_code":         "",
        "recording_id":        "",
        "recording_file_state": "",
        "n_audio_tracks":      "",
        "recording_start":     "",
        "recording_end":       "",
        "meeting_id":          meta.get("meeting_id", ""),
        "topic":               meta.get("topic", ""),
        "host_id":             meta.get("host_id", ""),
        "host_email":          meta.get("host_email", ""),
        "transcript_bytes":    meta.get("transcript_bytes", ""),
        "next_steps_file":     meta.get("next_steps_file", ""),
        "recording_count":     meta.get("recording_count", ""),
        "total_size":          meta.get("total_size", ""),
        "rep_loi":             (label or {}).get("rep_loi", ""),
        "rep_reasoning":       (label or {}).get("rep_reasoning", ""),
        "conversion_outcome":  (label or {}).get("conversion_outcome", ""),
    }


def _build_xcally_row(meta: dict, label: dict) -> dict:
    """xCally row. NOTE: transcription is OFF on this instance, so there is no
    transcript_file — these calls are NOT scoreable until transcription is
    solved. Metadata + disposition only.
    """
    uid = meta.get("uniqueid", "")
    created = meta.get("createdAt", "")
    try:
        dur = int(float(meta.get("duration") or 0))
        if dur > 36000:          # some xCally durations are in ms -> seconds
            dur //= 1000
    except (ValueError, TypeError):
        dur = ""
    return {
        "source":              "xcally",
        "call_id":             uid,
        "call_date":           created[:10] if created else "",
        "start_time":          created,
        "duration_sec":        dur,
        "direction":           "outbound",
        "rep_id":              meta.get("UserId", ""),
        "agent_name":          meta.get("agentName", ""),
        # call_type comes from the sweep's seller-UserId classification; the
        # legacy queue-based ingest had no call_type so fall back to "sales".
        "call_type":           meta.get("call_type") or "sales",
        "audio_file":          "",
        "audio_bytes":         "",
        "transcript_file":     "",
        "summary_file":        "",
        "queue_id":            "",
        "queue_name":          meta.get("queue", ""),
        "wrapup_code":         meta.get("userDisposition", ""),
        "recording_id":        meta.get("recording_id", ""),
        "recording_file_state": meta.get("transcribeStatus", ""),
        "n_audio_tracks":      meta.get("numberOfChannels", ""),
        "recording_start":     "",
        "recording_end":       "",
        "meeting_id":          "",
        "topic":               "",
        "host_id":             "",
        "host_email":          "",
        "transcript_bytes":    "",
        "next_steps_file":     "",
        "recording_count":     "",
        "total_size":          "",
        "rep_loi":             (label or {}).get("rep_loi", ""),
        "rep_reasoning":       (label or {}).get("rep_reasoning", ""),
        "conversion_outcome":  (label or {}).get("conversion_outcome", ""),
    }


def merge(dry_run: bool = False) -> MergeResult:
    # Existing calls.csv → (source, call_id) → row
    existing_rows = _read_csv(CALLS_CSV)
    existing_idx = _index_by(existing_rows, "source", "call_id")

    # New per-source ingest outputs
    genesys_meta = _read_csv(GENESYS_METADATA_CSV)
    zoom_meta = _read_csv(ZOOM_METADATA_CSV)
    labels = _read_csv(LABELS_CSV)

    # Labels file from ingest_genesys has no `source` column — assume genesys
    # if source is missing (matches the older ingest_genesys output).
    labels_idx: dict[tuple, dict] = {}
    for lr in labels:
        src = lr.get("source") or "genesys"
        cid = lr.get("call_id", "")
        if cid:
            labels_idx[(src, cid)] = lr

    # Genesys enrichment (queue_name, agent_name)
    omnibus = _read_csv(OMNIBUS_METADATA_CSV)
    omnibus_idx = {r.get("conversation_id", ""): r for r in omnibus}

    added = 0
    label_updates = 0
    skipped_seen = 0
    appended_rows: list[dict] = []

    # --- Genesys ---------------------------------------------------------
    for m in genesys_meta:
        cid = m.get("conversation_id", "")
        if not cid:
            continue
        key = ("genesys", cid)
        if key in existing_idx:
            skipped_seen += 1
            continue
        row = _build_genesys_row(m, labels_idx.get(key, {}),
                                 omnibus_idx.get(cid))
        appended_rows.append(row)
        existing_idx[key] = row
        added += 1

    # --- Zoom ------------------------------------------------------------
    for m in zoom_meta:
        uuid = m.get("meeting_uuid", "")
        if not uuid:
            continue
        key = ("zoom", uuid)
        if key in existing_idx:
            skipped_seen += 1
            continue
        row = _build_zoom_row(m, labels_idx.get(key, {}))
        appended_rows.append(row)
        existing_idx[key] = row
        added += 1

    # --- xCally (metadata only — transcription is off, no transcripts) ---
    for m in _read_csv(DATA / "xcally_metadata.csv"):
        uid = m.get("uniqueid", "")
        if not uid:
            continue
        key = ("xcally", uid)
        if key in existing_idx:
            skipped_seen += 1
            continue
        row = _build_xcally_row(m, labels_idx.get(key, {}))
        appended_rows.append(row)
        existing_idx[key] = row
        added += 1

    # --- Label backfill on EXISTING rows ---------------------------------
    # If labels.csv has a non-empty rep_loi / rep_reasoning / conversion_outcome
    # for a row that's already in calls.csv but missing it, fill it in.
    LABEL_FIELDS = ("rep_loi", "rep_reasoning", "conversion_outcome")
    for (src, cid), lr in labels_idx.items():
        row = existing_idx.get((src, cid))
        if row is None or row in appended_rows:
            continue
        changed = False
        for f in LABEL_FIELDS:
            new_v = (lr.get(f) or "").strip()
            if new_v and not (row.get(f) or "").strip():
                row[f] = new_v
                changed = True
        if changed:
            label_updates += 1

    total = len(existing_idx)

    if not dry_run and (appended_rows or label_updates):
        # Write back the full file (preserves order: existing rows first,
        # then new rows in source-then-encountered order).
        out_rows = existing_rows + appended_rows
        # Apply label-backfill changes (existing_idx mutated in place, but
        # existing_rows shares the same dict objects, so they're updated too).
        with CALLS_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CALLS_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows)

    return MergeResult(total=total, added=added, label_updates=label_updates,
                       skipped_seen=skipped_seen)


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing calls.csv.")
    args = ap.parse_args()
    r = merge(dry_run=args.dry_run)
    tag = "DRY-RUN " if args.dry_run else ""
    print(f"{tag}calls.csv: total={r.total} added={r.added} "
          f"label_updates={r.label_updates} skipped_existing={r.skipped_seen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
