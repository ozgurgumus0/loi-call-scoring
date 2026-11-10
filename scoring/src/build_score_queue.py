"""Build the SCORE QUEUE for the periodic Kaggle LOI run.

Finds the zoom SALES calls in data/calls.csv that are NOT yet in
predictions_qwen3_8b_sales_full.csv, and writes an ids file that
`emit_test_prompting` turns into the jsonl uploaded to Kaggle.

"sales" = consultations (SC1 + follow-ups). The SC1-vs-follow-up split is still
provisional, so we score every call_type=='sales' call. emit_test_prompting
drops any id without a usable transcript, so the queue may safely list ids whose
VTT is missing — they just won't appear in the jsonl.

Usage:
  python -m src.build_score_queue                       # all unscored sales calls
  python -m src.build_score_queue --since 2026-06-12    # only recent
  python -m src.build_score_queue --dry-run             # counts only, write nothing
  python -m src.build_score_queue --out data/train/ids_sales_new.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DATA, ROOT

PRED_CSV = ROOT / "predictions_qwen3_8b_sales_full.csv"
DEFAULT_OUT = DATA / "train" / "ids_sales_new.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="only calls on/after this date YYYY-MM-DD")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="ids json output path")
    ap.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    a = ap.parse_args(argv)

    calls = pd.read_csv(DATA / "calls.csv", low_memory=False)
    z = calls[(calls["source"] == "zoom") & (calls["call_type"] == "sales")].copy()
    n_all = len(z)

    if a.since:
        d = pd.to_datetime(z["call_date"], errors="coerce")
        z = z[d >= pd.Timestamp(a.since)]

    scored = set()
    if PRED_CSV.exists():
        scored = set(pd.read_csv(PRED_CSV, usecols=["call_id"])["call_id"].astype(str))

    z["call_id"] = z["call_id"].astype(str)
    queue = [c for c in z["call_id"].tolist() if c not in scored]

    print(f"zoom sales calls (calls.csv) : {n_all}")
    if a.since:
        print(f"  after --since {a.since}        : {len(z)}")
    print(f"already scored (predictions)  : {len(scored)}")
    print(f"=> SCORE QUEUE (unscored)     : {len(queue)}")

    if a.dry_run:
        print("(dry-run: nothing written)")
        return 0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"call_ids": queue, "source": "zoom",
         "note": f"unscored zoom sales calls; since={a.since or 'all'}"},
        ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[saved] {out}  ({len(queue)} ids)")
    # emit to the filename prompt_qwen.py looks for FIRST so the Kaggle run
    # picks up exactly this batch (the full-corpus jsonl is regenerable from
    # ids_sales_full.json anytime).
    print(f"NEXT: python -m src.emit_test_prompting --ids {out} "
          f"--out data/train/test_prompting_sales.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
