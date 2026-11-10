"""Emit ALL Sonnet-scored SALES calls for Qwen topic extraction on Kaggle.

Builds the input for `kaggle/extract_topics_qwen.py` (the $0 replacement for
the frozen Claude topic extraction — see the no-Claude-spend constraint).

Design:
  * Same TOPIC_PROMPT Claude used (verbatim, from src.extract_topics) so the
    Qwen-vs-Claude agreement on the overlap is a clean extractor comparison.
  * Every Sonnet-scored sales call is included — INCLUDING the ~282 that
    already have Claude topics. Those carry their Claude labels in a `gold`
    field and come FIRST in the file, so the Kaggle script's validation phase
    (VALIDATE_ONLY=True → first ~110 gold rows) needs no extra plumbing, and
    the full run re-extracts them for a single-extractor analysis.
  * Transcripts are head/tail-capped at MAX_CHARS so prompt+transcript+output
    fit Qwen3-8B's 8192 window (TOPIC_PROMPT ~1.2k tok; Italian ~3.08 ch/tok).
    Claude saw the full transcript — the validation phase therefore measures
    extractor difference INCLUDING this truncation, which is what we ship.

Each row: {call_id, source, gold|null,
           messages:[system(TOPIC_PROMPT), user(transcript)]}

Output: data/train/topics_prompting_sales.jsonl  (upload to the Kaggle
dataset `<KAGGLE_USERNAME>/loi-qwen-training` as a new version).

Usage:  python -m src.emit_topics_prompting
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import CALLS_CSV, DATA, OUT_SCORES, OUTPUTS
from .extract_topics import TOPIC_PROMPT
from .transcribe import CallRow, load_existing_transcript

OUT_PATH = DATA / "train" / "topics_prompting_sales.jsonl"
CLAUDE_TOPICS_DIR = OUTPUTS / "topics"

# Window budget (v2, after the round-1 validation CHECK): MAX_SEQ is now 12288
# in the Kaggle script. TOPIC_PROMPT ~1.2k tok + calibration ~0.6k + transcript
# + 1024 gen. 26k chars Italian ~= 8.4k tok -> input ~10.3k tok, ~1k headroom
# even at a pessimistic 2.9 ch/tok. Round-1 diagnosis: the old 16k cap cut 67%
# of calls and halved recall on mid-call labels (technical_questions full-R
# 0.60 vs truncated-R 0.29); 26k cuts only ~41%.
MAX_CHARS = 26_000
HEAD_CHARS = 15_000  # opening: intro, needs assessment, most topics surface
TAIL_CHARS = 10_500  # close: objections, next steps, buyer signals
OMISSION_MARK = "\n[... parte centrale della trascrizione omessa ...]\n"


def _head_tail(text: str) -> str:
    """Cap the transcript, keeping the opening and the close (string-level —
    topic extraction reads the diarized text, not segments). Cut at line
    boundaries so we never split a speaker turn mid-sentence."""
    if len(text) <= MAX_CHARS:
        return text
    head = text[:HEAD_CHARS]
    nl = head.rfind("\n")
    if nl > HEAD_CHARS // 2:
        head = head[:nl]
    tail = text[-TAIL_CHARS:]
    nl = tail.find("\n")
    if 0 <= nl < TAIL_CHARS // 2:
        tail = tail[nl + 1:]
    return head + OMISSION_MARK + tail


def _scored_sales_call_ids() -> list[str]:
    """Sonnet-scored zoom call_ids (deduped) restricted to call_type=='sales'."""
    scored: set[str] = set()
    for p in OUT_SCORES.glob("zoom__*.json"):
        n = p.name
        if "__ollama_" in n or n.startswith("upload__") or n.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (d.get("model") or "").lower().startswith("claude-sonnet"):
            continue
        sc = d.get("score") or {}
        if d.get("call_id") and sc.get("outcome"):
            scored.add(d["call_id"])

    sales: list[str] = []
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("source") == "zoom" and r.get("call_id") in scored
                    and r.get("call_type") == "sales"):
                sales.append(r["call_id"])
    return sorted(set(sales))


def _claude_gold() -> dict[str, dict]:
    """{call_id -> {topics_discussed, objections_raised, buyer_signals}} from
    the existing Claude topic extractions (the validation overlap)."""
    gold: dict[str, dict] = {}
    for p in CLAUDE_TOPICS_DIR.glob("zoom__*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (d.get("model") or "").lower().startswith("claude-sonnet"):
            continue
        ex = d.get("extraction") or {}
        cid = d.get("call_id")
        if cid and cid not in gold:
            gold[cid] = {
                "topics_discussed": sorted(ex.get("topics_discussed") or []),
                "objections_raised": sorted(ex.get("objections_raised") or []),
                "buyer_signals": sorted(ex.get("buyer_signals") or []),
            }
    return gold


def _transcript_map() -> dict[str, str]:
    out = {}
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source") == "zoom" and r.get("transcript_file"):
                out[r["call_id"]] = r["transcript_file"]
    return out


def main() -> int:
    ids = _scored_sales_call_ids()
    gold = _claude_gold()
    tmap = _transcript_map()
    print(f"Sonnet-scored sales calls: {len(ids)}  "
          f"(with Claude-topic gold: {sum(1 for c in ids if c in gold)})")

    rows, missing, truncated, lens = [], [], 0, []
    for cid in ids:
        tf = tmap.get(cid)
        if not tf:
            missing.append(cid)
            continue
        loaded = load_existing_transcript(
            CallRow(source="zoom", call_id=cid, audio_file=None,
                    transcript_file=tf, duration_sec=None))
        if loaded is None:
            missing.append(cid)
            continue
        text, _segments = loaded
        capped = _head_tail(text)
        if len(capped) < len(text):
            truncated += 1
        lens.append(len(capped))
        rows.append({
            "call_id": cid,
            "source": "zoom",
            "gold": gold.get(cid),
            "messages": [
                {"role": "system", "content": TOPIC_PROMPT},
                {"role": "user", "content": (
                    "Trascrizione della call (italiano, già diarizzata):\n\n"
                    f"<transcript>\n{capped}\n</transcript>\n\n"
                    "Estrai le etichette strutturate secondo il vocabolario."
                )},
            ],
        })

    # gold-bearing rows first -> the Kaggle validation phase is just "the
    # first N rows that have gold", and a capped full run still re-extracts
    # the audit set before anything else.
    rows.sort(key=lambda r: (r["gold"] is None, r["call_id"]))

    OUT_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")

    n_gold = sum(1 for r in rows if r["gold"])
    lens.sort()
    print(f"[saved] {OUT_PATH}")
    print(f"  rows={len(rows)} (gold-first: {n_gold} with gold, "
          f"{len(rows) - n_gold} without)")
    print(f"  transcripts capped at {MAX_CHARS} chars: {truncated} truncated; "
          f"median={lens[len(lens)//2]} max={lens[-1]} chars")
    if missing:
        print(f"  WARNING {len(missing)} calls without transcript, skipped: "
              f"{missing[:5]}")
    print("Upload as a new version of the Kaggle dataset, then run "
          "kaggle/extract_topics_qwen.py (VALIDATE_ONLY=True first).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
