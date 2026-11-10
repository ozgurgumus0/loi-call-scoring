"""Refresh zoom contract outcomes in data/calls.csv from an updated CRM merge.

The Zoom CRM re-merge (e.g. Downloads/calls2_zoom_merged.xlsx) carries a fresh
`CONTRACT_SIGNED` column (0/1) that supersedes the older `conversion_outcome`
(more signers + far better coverage). This script:

  * backs up data/calls.csv -> data/calls.csv.bak_contracts
  * for source==zoom rows, sets conversion_outcome from CONTRACT_SIGNED
    (1 -> 'signed', 0 -> 'not_signed', blank -> '') and adds an explicit
    `contract_signed` column (1/0/'')
  * leaves all non-zoom rows untouched
  * prints an old->new diff (signer counts, flips, gains)

It NEVER touches data/train/*.jsonl or test_ids.json — conversion is read only
for the post-hoc contract join, never into a training message, and the frozen
test membership stays pinned.

Usage:
    python -m src.refresh_contracts --xlsx "C:/.../calls2_zoom_merged.xlsx"
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd

from .config import CALLS_CSV


def load_contract_map(xlsx: str, sheet: str = "merged") -> dict[str, str]:
    """{zoom call_id -> '1'|'0'|''} from the merge's CONTRACT_SIGNED column."""
    df = pd.read_excel(xlsx, sheet_name=sheet)
    out: dict[str, str] = {}
    for r in df.itertuples():
        cs = getattr(r, "CONTRACT_SIGNED")
        if pd.isna(cs):
            out[str(r.call_id)] = ""
        else:
            out[str(r.call_id)] = "1" if int(cs) == 1 else "0"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="path to the zoom CRM merge xlsx")
    ap.add_argument("--sheet", default="merged")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the diff only; do not write")
    a = ap.parse_args()

    cs_map = load_contract_map(a.xlsx, a.sheet)

    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "contract_signed" not in fields:
        fields.append("contract_signed")

    old_signed, new_signed, flips, gains, unmatched = set(), set(), [], [], 0
    for row in rows:
        if row.get("source") != "zoom":
            row.setdefault("contract_signed", "")
            continue
        cid = row["call_id"]
        old = (row.get("conversion_outcome") or "").strip()
        if old == "signed":
            old_signed.add(cid)
        cs = cs_map.get(cid)
        if cs is None:                       # zoom call not in the new merge
            unmatched += 1
            row.setdefault("contract_signed", "")
            continue
        row["contract_signed"] = cs
        new_co = {"1": "signed", "0": "not_signed", "": ""}[cs]
        row["conversion_outcome"] = new_co
        if cs == "1":
            new_signed.add(cid)
            if old != "signed":
                gains.append(cid)
        elif old == "signed":
            flips.append(cid)

    print(f"zoom call_ids in new merge: {len(cs_map)}   unmatched in calls.csv: {unmatched}")
    print(f"OLD signed: {len(old_signed)}   NEW signed: {len(new_signed)}")
    print(f"  gained (newly signed): {len(gains)}")
    print(f"  flipped off (was signed, now not): {len(flips)} {flips}")

    if a.dry_run:
        print("\n[dry-run] no files written.")
        return 0

    backup = CALLS_CSV.with_suffix(".csv.bak_contracts")
    shutil.copy2(CALLS_CSV, backup)
    print(f"\n[backup] {backup}")

    with CALLS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            row.setdefault("contract_signed", "")
            w.writerow(row)
    print(f"[saved]  {CALLS_CSV}  ({len(rows)} rows, +contract_signed column)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
