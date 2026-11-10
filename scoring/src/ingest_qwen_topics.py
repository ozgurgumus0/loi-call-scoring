"""Ingest the Kaggle Qwen topic-extraction output into per-call JSONs.

Reads the downloaded `topics_qwen.jsonl` (from kaggle/extract_topics_qwen.py)
and writes one file per call to `outputs/topics_qwen/zoom__<id>.json`, schema-
compatible with the Claude files in `outputs/topics/` (which stay untouched as
the audit set). Re-running is idempotent: existing files are overwritten with
the same content; multiple downloads (split runs via START/END) just append
their slice — pass each file in turn.

Usage:
    python -m src.ingest_qwen_topics "C:/Users/.../Downloads/topics_qwen.jsonl"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import OUTPUTS

QWEN_TOPICS_DIR = OUTPUTS / "topics_qwen"


def _output_path(call_id: str) -> Path:
    stem = call_id.replace("/", "_").replace("+", "-").replace("=", "")
    return QWEN_TOPICS_DIR / f"zoom__{stem}.json"


def main(in_path: Path) -> int:
    QWEN_TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    written = parse_fails = bad = 0
    dropped_tokens = 0
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            cid = r.get("call_id")
            if not cid:
                bad += 1
                continue
            if r.get("parse_fail"):
                parse_fails += 1     # still write it — _load_topics skips parse_fail rows
            dropped_tokens += int(r.get("vocab_dropped") or 0)
            payload = {
                "call_id": cid,
                "source": r.get("source", "zoom"),
                "model": r.get("model", "qwen3-8b-topics"),
                "topic_schema_version": "v1-2026-06-02",
                "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "extraction": r.get("extraction") or {},
                "parse_fail": bool(r.get("parse_fail")),
                "raw": r.get("raw", ""),
            }
            _output_path(cid).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8")
            written += 1
    total_on_disk = len(list(QWEN_TOPICS_DIR.glob("zoom__*.json")))
    print(f"[ingested] {written} calls from {in_path.name} "
          f"({parse_fails} parse-fails, {dropped_tokens} invented tokens were "
          f"dropped on Kaggle, {bad} bad lines)")
    print(f"outputs/topics_qwen/ now holds {total_on_disk} calls")
    print("Next: python -m src.analyze_topics --extractor qwen")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, help="downloaded topics_qwen.jsonl")
    a = ap.parse_args()
    if not a.jsonl.exists():
        raise SystemExit(f"not found: {a.jsonl}")
    raise SystemExit(main(a.jsonl))
