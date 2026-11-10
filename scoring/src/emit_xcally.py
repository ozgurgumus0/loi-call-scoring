"""Emit xCally OMNIBUS transcripts for prompted Qwen scoring on Kaggle.

Input = an OMNIBUS_<date>.csv export (xCally speech-analytics dump: one row per
call with FULL_TRANSCRIPT already diarized as [CLIENTE]/[AGENTE] lines plus
queue/agent/disposition metadata). The export can be truncated mid-write —
incomplete trailing rows are dropped.

Scope filter (substantive calls only — near-empty inputs produce junk cold
scores, see the Whisper-rescue verdict):
  transcript >= 500 chars AND customer words >= 20
  AND USER_DISPOSITION != CHIAMATA_SENZA_RISPOSTA (no-answer dials)

Outputs:
  data/xcally/omnibus_meta_<date>.csv          — ALL rows + in_scope/skip_reason
                                                 (the enrichment sidecar for
                                                 export_xcally_excel)
  data/train/test_prompting_xcally_b<N>.jsonl  — Kaggle input batches
                                                 (--batch-size, default 750 to
                                                 stay under the session cap)

Rows use the same shape as emit_test_prompting (system=RUBRIC_PROMPT, the
kernel appends CALIBRATION itself), no assistant turn (no gold), and
source='xcally' so predictions never collide with the zoom masters.

To run a batch: copy it over data/train/test_prompting_sales.jsonl (the name
kaggle_run stages), then `python -m src.kaggle_run --go`.

Usage:  python -m src.emit_xcally --csv <OMNIBUS.csv>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .config import DATA
from .rubric import RUBRIC_PROMPT, build_messages
from .transcript_prep import head_tail_truncate

MIN_CHARS = 500
MIN_CUST_WORDS = 20
NO_ANSWER = "CHIAMATA_SENZA_RISPOSTA"

META_COLS = [
    "call_id", "created_at", "agent", "queue", "type", "direction",
    "caller_num", "user_disposition", "duration_min", "customer_words",
    "agent_words", "transcript_chars", "in_scope", "skip_reason",
]


def load_omnibus(csv_path: Path) -> list[dict]:
    """Read the export row-by-row, dropping truncated/malformed rows."""
    csv.field_size_limit(10**8)
    rows, bad = [], 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for raw in r:
            if len(raw) == len(header):
                rows.append(dict(zip(header, raw)))
            else:
                bad += 1
    if bad:
        print(f"  WARNING: dropped {bad} malformed/truncated row(s)")
    return rows


def _skip_reason(row: dict) -> str | None:
    if row.get("TRANSCRIBE_STATUS") != "completed":
        return "not_transcribed"
    if row.get("USER_DISPOSITION") == NO_ANSWER:
        return "no_answer"
    if len(row.get("FULL_TRANSCRIPT") or "") < MIN_CHARS:
        return "transcript_too_short"
    try:
        cw = float(row.get("CUSTOMER_WORD_COUNT") or 0)
    except ValueError:
        cw = 0
    if cw < MIN_CUST_WORDS:
        return "customer_silent"
    return None


def main(csv_path: Path, batch_size: int) -> int:
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", csv_path.name)
    tag = date_m.group(1) if date_m else "latest"

    rows = load_omnibus(csv_path)
    print(f"OMNIBUS rows: {len(rows)}  ({csv_path.name})")

    meta, emit = [], []
    for row in sorted(rows, key=lambda r: r.get("CREATEDAT", "")):
        reason = _skip_reason(row)
        try:
            dur_min = round(float(row.get("DURATION_MS") or 0) / 60000, 2)
        except ValueError:
            dur_min = None
        meta.append({
            "call_id": row["UNIQUEID"],
            "created_at": row.get("CREATEDAT", ""),
            "agent": row.get("AGENT", ""),
            "queue": row.get("QUEUE", ""),
            "type": row.get("TYPE", ""),
            "direction": row.get("DIRECTION", ""),
            "caller_num": row.get("CALLERIDNUM", ""),
            "user_disposition": row.get("USER_DISPOSITION", ""),
            "duration_min": dur_min,
            "customer_words": row.get("CUSTOMER_WORD_COUNT", ""),
            "agent_words": row.get("AGENT_WORD_COUNT", ""),
            "transcript_chars": len(row.get("FULL_TRANSCRIPT") or ""),
            "in_scope": reason is None,
            "skip_reason": reason or "",
        })
        if reason is not None:
            continue
        transcript = head_tail_truncate(row["FULL_TRANSCRIPT"])
        emit.append({
            "call_id": row["UNIQUEID"],
            "source": "xcally",
            "messages": [
                {"role": "system", "content": RUBRIC_PROMPT},
                {"role": "user", "content": build_messages(transcript)[0]["content"]},
            ],
        })

    xdir = DATA / "xcally"
    xdir.mkdir(parents=True, exist_ok=True)
    meta_path = xdir / f"omnibus_meta_{tag}.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_COLS)
        w.writeheader()
        w.writerows(meta)
    print(f"[saved] {meta_path}  ({len(meta)} rows, {len(emit)} in scope)")

    skips: dict[str, int] = {}
    for m in meta:
        if m["skip_reason"]:
            skips[m["skip_reason"]] = skips.get(m["skip_reason"], 0) + 1
    for k, v in sorted(skips.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {k}: {v}")

    batches = [emit[i:i + batch_size] for i in range(0, len(emit), batch_size)] \
        if batch_size else [emit]
    for i, batch in enumerate(batches, 1):
        out = DATA / "train" / f"test_prompting_xcally_b{i}.jsonl"
        out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch),
            encoding="utf-8")
        print(f"[saved] {out}  ({len(batch)} rows)")
    print("Copy a batch over data/train/test_prompting_sales.jsonl, then "
          "`python -m src.kaggle_run --go`.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="OMNIBUS export csv")
    ap.add_argument("--batch-size", type=int, default=750,
                    help="rows per Kaggle batch jsonl (0 = single file)")
    a = ap.parse_args()
    raise SystemExit(main(Path(a.csv), a.batch_size))
