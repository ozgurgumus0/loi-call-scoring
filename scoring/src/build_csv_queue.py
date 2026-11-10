"""Build a scoring queue from a standalone scheduling CSV.

Unlike build_score_queue, CSV membership includes both sales and booking calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DATA
from .csv_handoff import enrich_csv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument(
        "--out",
        default=str(DATA / "train" / "ids_csv_batch.json"),
        help="output IDs JSON",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    source, block = enrich_csv(args.csv_path)
    counts = block["loi_match_status"].value_counts()
    print(f"CSV rows: {len(source)}")
    print(counts.to_string())

    queue = (
        block.loc[
            block["loi_match_status"] == "recorded_awaiting_score",
            "matched_call_id",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    calls = pd.read_csv(DATA / "calls.csv", low_memory=False, dtype={"call_id": str})
    types = (
        calls.drop_duplicates("call_id", keep="last")
        .set_index("call_id")["call_type"]
    )
    type_counts = pd.Series([types.get(call_id, "unknown") for call_id in queue])
    if not type_counts.empty:
        print("scoreable call types:")
        print(type_counts.value_counts().to_string())
    print(f"=> CSV SCORE QUEUE: {len(queue)}")

    if args.dry_run:
        print("(dry-run: nothing written)")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "call_ids": queue,
                "source": "zoom",
                "note": f"standalone CSV queue from {Path(args.csv_path).name}",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[saved] {out} ({len(queue)} ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

