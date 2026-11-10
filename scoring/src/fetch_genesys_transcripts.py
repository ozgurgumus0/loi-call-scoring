"""Pull Genesys Speech-and-Text-Analytics transcripts for conversations not
covered by an OMNIBUS export (the 1,916 audio calls dated 2026-05-06..05-20,
plus within-window gaps on 2026-04-30 / 05-04 / 05-05).

For each missing conversation_id:
  1. GET /api/v2/conversations/{cid}                 -> participants[*].calls[*].id
  2. GET /api/v2/speechandtextanalytics/conversations/{cid}/communications/{cm}/transcripturl
     (only the customer-side communicationId returns 200; agent-side 404s)
  3. Download the returned signed URL -> transcript JSON
  4. Convert phrases to OMNIBUS-style prose: `[Agente]`/`[Cliente]` turns,
     one turn per line, adjacent same-speaker phrases collapsed.
  5. Write to data/transcripts/genesys/<cid>.txt and set transcript_file
     on the matching row in data/calls.csv.

Idempotent: skips cids that already have a .txt file on disk. Safe to resume.

Usage:
  python -m src.fetch_genesys_transcripts                  # all missing
  python -m src.fetch_genesys_transcripts --max-calls 20   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config
from .genesys_client import GenesysClient, GenesysError


def phrases_to_text(phrases: list[dict]) -> str:
    """Collapse a list of Genesys analytics phrases into OMNIBUS prose."""
    lines: list[str] = []
    cur_speaker: str | None = None
    cur_buf: list[str] = []
    for ph in phrases:
        purpose = (ph.get("participantPurpose")
                   or (ph.get("participant") or {}).get("purpose")
                   or "").lower()
        speaker = "Agente" if purpose == "internal" else "Cliente"
        text = (ph.get("text") or "").strip()
        if not text:
            continue
        if speaker != cur_speaker:
            if cur_speaker is not None:
                lines.append(f"[{cur_speaker}] {' '.join(cur_buf)}")
            cur_speaker = speaker
            cur_buf = [text]
        else:
            cur_buf.append(text)
    if cur_speaker is not None:
        lines.append(f"[{cur_speaker}] {' '.join(cur_buf)}")
    return "\n".join(lines)


def fetch_one(gc: GenesysClient, cid: str) -> tuple[str, str]:
    """Returns (text, status). status in {'ok','no_transcript','no_comm','error:<msg>'}."""
    try:
        conv = gc._api_request("GET", f"/api/v2/conversations/{cid}")
    except GenesysError as e:
        return "", f"error:conv:{str(e)[:80]}"

    comm_ids: list[tuple[str, str]] = []  # (purpose, communicationId)
    for p in conv.get("participants", []) or []:
        purpose = p.get("purpose", "?")
        for c in p.get("calls", []) or []:
            if c.get("id"):
                comm_ids.append((purpose, c["id"]))
    if not comm_ids:
        return "", "no_comm"

    # Try customer-side first (transcript lives there); fall back to others.
    comm_ids.sort(key=lambda t: 0 if t[0] == "customer" else 1)
    last_err = ""
    for _purpose, cm in comm_ids:
        path = (f"/api/v2/speechandtextanalytics/conversations/{cid}"
                f"/communications/{cm}/transcripturl")
        try:
            resp = gc._api_request("GET", path)
        except GenesysError as e:
            last_err = str(e)[:120]
            continue
        url = (resp or {}).get("url") or (resp or {}).get("uri") or ""
        if not url:
            last_err = "empty url"
            continue
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read()
            j = json.loads(body)
        except (urllib.error.HTTPError, urllib.error.URLError,
                ConnectionResetError, TimeoutError, json.JSONDecodeError) as e:
            last_err = f"download:{type(e).__name__}"
            continue

        transcripts = j.get("transcripts") or []
        all_phrases: list[dict] = []
        for tr in transcripts:
            all_phrases.extend(tr.get("phrases") or [])
        if not all_phrases:
            return "", "no_transcript"
        text = phrases_to_text(all_phrases)
        return text, "ok" if text else "no_transcript"

    return "", f"error:{last_err}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=0,
                    help="stop after N successful fetches (0=all)")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    out_dir = config.DATA / "transcripts" / "genesys"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify missing conversations from calls.csv (genesys rows with no transcript_file).
    calls = list(csv.DictReader(open(config.DATA / "calls.csv", encoding="utf-8")))
    todo = [r for r in calls
            if r["source"] == "genesys"
            and not r["transcript_file"]
            and not (out_dir / f"{r['call_id']}.txt").exists()]
    print(f"missing transcripts: {len(todo)}")

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = config.OUT_LOGS / f"fetch_genesys_transcripts_{ts}.log"
    gc = GenesysClient()

    ok = empty = err = 0
    fields = list(calls[0].keys())
    cid_to_row = {r["call_id"]: r for r in calls}

    with log_path.open("w", encoding="utf-8") as fh:
        try:
            for i, r in enumerate(todo, 1):
                if args.max_calls and ok >= args.max_calls:
                    break
                cid = r["call_id"]
                text, status = fetch_one(gc, cid)
                if status == "ok":
                    (out_dir / f"{cid}.txt").write_text(text, encoding="utf-8")
                    cid_to_row[cid]["transcript_file"] = f"transcripts/genesys/{cid}.txt"
                    ok += 1
                elif status in ("no_transcript", "no_comm"):
                    empty += 1
                else:
                    err += 1
                fh.write(f"{cid}\t{status}\t{len(text)}\n")
                if i % 50 == 0 or i == len(todo):
                    fh.flush()
                    print(f"  {i}/{len(todo)}  ok={ok}  empty={empty}  err={err}",
                          flush=True)
                # 0.25s = 4 req/sec on the outer puller (each fetch_one is ~3
                # API calls -> ~12 req/sec) — well under Genesys's 300/min cap.
                time.sleep(0.25)
        finally:
            # Rewrite calls.csv with any new transcript_file values
            with (config.DATA / "calls.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(calls)
            print(f"DONE  ok={ok} empty={empty} err={err}  log={log_path}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
