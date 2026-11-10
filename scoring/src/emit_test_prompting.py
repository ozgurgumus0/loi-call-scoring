"""Emit the frozen test set for PROMPTING a big open model (no fine-tuning).

Same 200 frozen-test calls as emit_test_with_ids, but the system message is the
FULL `RUBRIC_PROMPT` (the one Claude gets) instead of the condensed
`QWEN_RUBRIC_PROMPT` we had to use to fit a 7B's 4096 window. A prompted 9B has
a huge context, so it gets the full rubric — exactly how Claude is prompted.

Each row: {call_id, source, messages:[system(full rubric), user(transcript),
assistant(gold)]}. The assistant turn is the Sonnet gold label — the Kaggle
notebook reads it for scoring and DROPS it before generation (msgs[:-1]).

Output: data/train/test_prompting.jsonl  (upload to the Kaggle dataset).

Usage:  python -m src.emit_test_prompting
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import DATA
from .rubric import RUBRIC_PROMPT, build_messages
from . import build_training_jsonl as B


def main(ids_path: Path | None = None, out_path: Path | None = None) -> int:
    ids_path = ids_path or (DATA / "train" / "test_ids.json")
    out_path = out_path or (DATA / "train" / "test_prompting.jsonl")

    meta = json.loads(ids_path.read_text(encoding="utf-8"))
    test_ids = meta["call_ids"]
    source = meta.get("source", "zoom")
    print(f"Frozen test ids: {len(test_ids)} (source={source})")

    B._TR_MAP = B._load_transcript_map()
    score_by_id = {r["call_id"]: r["score"] for r in B._load_sonnet_scores()}

    rows, missing, n_unlabeled = [], [], 0
    # Preserve the ids-file order (the full-corpus file is labeled-first so a
    # split run's session 1 covers every gold row).
    for cid in test_ids:
        # Sonnet label optional (2026-06-12: the full-corpus run includes the
        # ~351 unlabeled sales calls — they get a production score but no
        # gold/agreement row). _build_example only uses the score for the
        # assistant turn, so a dummy works for unlabeled calls.
        sc = score_by_id.get(cid)
        ex = B._build_example(cid, sc if sc is not None else {})
        if ex is None:
            missing.append(cid); continue
        # swap the condensed system prompt for the FULL rubric; keep user
        # (+assistant gold only when the call has a Sonnet label)
        user_msg = ex["messages"][1]["content"]
        msgs = [
            {"role": "system", "content": RUBRIC_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        if sc is not None:
            msgs.append({"role": "assistant",
                         "content": ex["messages"][2]["content"]})
        else:
            n_unlabeled += 1
        rows.append({"call_id": cid, "source": source, "messages": msgs})

    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    print(f"[saved] {out_path}  ({len(rows)} rows, of which {n_unlabeled} "
          f"unlabeled/no-gold; {len(missing)} missing)")
    if missing:
        print(f"  WARNING missing: {missing[:10]}")
    print("Upload this file to the Kaggle dataset, then run kaggle/prompt_qwen.py.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="JSON ids file (default data/train/test_ids.json)")
    ap.add_argument("--out", help="output jsonl (default data/train/test_prompting.jsonl)")
    a = ap.parse_args()
    raise SystemExit(main(Path(a.ids) if a.ids else None,
                          Path(a.out) if a.out else None))
