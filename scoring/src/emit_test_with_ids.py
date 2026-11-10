"""Emit the frozen test set WITH call_id+source on every row, for the contract
verdict.

WHY: test.jsonl rows carry only {"messages": [...]} — no call_id. So when
kaggle/eval_only.py produces predictions there is no robust way to join them
back to conversion_outcome (positional zipping is unsafe — test.jsonl is written
in score-FILENAME sort order, while test_ids.json is sorted by raw call_id, and
the two differ because call_ids contain '/','+','=' that filenames sanitize).

This writes data/train/test_with_ids.jsonl: each row is
    {"call_id": ..., "source": ..., "messages": [sys, user, assistant]}
with messages IDENTICAL to the existing test.jsonl (same prompt, same
transcript, same gold). The extra top-level keys are ignored by training/eval
(both read only ex["messages"]).

It then VERIFIES the regenerated rows match the existing test.jsonl as a set of
(system,user,assistant) tuples, so we know the uploaded eval inputs are
unchanged. Upload this file as the test file for the v3 eval; eval_only.py
(patched) dumps predictions.csv keyed by call_id.

Usage:  python -m src.emit_test_with_ids
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import DATA
from . import build_training_jsonl as B


def _existing_keyset(path: Path) -> set[tuple[str, str, str]]:
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)["messages"]
            out.add((m[0]["content"], m[1]["content"], m[2]["content"]))
    return out


def main() -> int:
    ids_path = DATA / "train" / "test_ids.json"
    out_path = DATA / "train" / "test_with_ids.jsonl"
    existing = DATA / "train" / "test.jsonl"

    meta = json.loads(ids_path.read_text(encoding="utf-8"))
    test_ids = meta["call_ids"]
    source = meta.get("source", "zoom")
    print(f"Frozen test ids: {len(test_ids)} (source={source})")

    # Reuse the builder's transcript map + Sonnet scores so content is identical.
    B._TR_MAP = B._load_transcript_map()
    score_by_id = {r["call_id"]: r["score"] for r in B._load_sonnet_scores()}

    rows, missing = [], []
    for cid in sorted(test_ids):
        sc = score_by_id.get(cid)
        if sc is None:
            missing.append(cid); continue
        ex = B._build_example(cid, sc)
        if ex is None:
            missing.append(cid); continue
        ex = {"call_id": cid, "source": source, **ex}
        rows.append(ex)

    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    print(f"[saved] {out_path}  ({len(rows)} rows, {len(missing)} missing)")
    if missing:
        print(f"  WARNING missing (no score/transcript): {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")

    # Verify content parity with the existing test.jsonl. Check BOTH the set of
    # msg-triples AND the row count — a set-only check would miss a row-count
    # mismatch if duplicate transcripts existed.
    if existing.exists():
        old_rows = sum(1 for _ in existing.open(encoding="utf-8"))
        old = _existing_keyset(existing)
        new = {(r["messages"][0]["content"], r["messages"][1]["content"],
                r["messages"][2]["content"]) for r in rows}
        same = old & new
        print(f"\nContent parity vs existing test.jsonl: "
              f"{len(same)}/{len(old)} existing rows reproduced "
              f"({len(new - old)} new-only, {len(old - new)} old-only); "
              f"row count {len(rows)} new vs {old_rows} existing.")
        if old == new and len(rows) == old_rows:
            print("  OK — identical eval inputs; safe to upload as the test file.")
        elif old == new and len(rows) != old_rows:
            print(f"  WARNING — same unique content but row counts differ "
                  f"({len(rows)} vs {old_rows}); duplicate transcript(s) — "
                  f"investigate before trusting per-call joins.")
        else:
            print("  WARNING — content differs from the uploaded test.jsonl. "
                  "The v3 model was aligned to the uploaded version; investigate "
                  "before trusting an eval on this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
