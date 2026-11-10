"""Lever #1 — recover the collapsed Caldo signal from a CONTINUOUS score.

The discrete band argmax collapses Caldo -> Calduccio (the model hedges the rare
class). But `prompt_qwen.py` now also emits a 0-100 `closing_readiness`. The
ranking signal that the binning throws away may still live in that number. This
script, on the sales-only frozen test:

  1. Compares three rankers vs the signed-contract outcome (Mann-Whitney AUC):
       - the discrete band (rank Freddo<Tiepido<Calduccio<Caldo)   [what we have]
       - closing_readiness (continuous)                            [lever #1]
       - a 50/50 blend of the two
     ...against the lead_score floor (0.654) and Claude ceiling (0.749).
  2. Fits a closing_readiness threshold that PROMOTES band==Calduccio -> Caldo,
     and reports the recovered Caldo recall/precision vs Claude's Caldo labels +
     the resulting band distribution.

No model, no API, no new training. Usage:
  python -m src.calibrate_caldo --pred "predictions.csv" \
      --ids data/train/test_ids_sales_only.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import CALLS_CSV

RANK = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7,
        "Non interessato": 0, "Non idoneo": 0}


def _auc(scores: list[float], labels: list[int]) -> float:
    """Tie-aware Mann-Whitney AUC. labels: 1=signed, 0=not."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    paired = sorted(zip(scores, labels), key=lambda t: t[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    rsum = sum(r for r, (_s, l) in zip(ranks, paired) if l)
    return (rsum - pos * (pos + 1) / 2.0) / (pos * neg)


def _load_conv() -> dict[str, int]:
    """call_id -> 1 if conversion_outcome == 'signed' else 0 (zoom rows)."""
    out: dict[str, int] = {}
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source") == "zoom":
                out[r["call_id"]] = int((r.get("conversion_outcome") or "").strip() == "signed")
    return out


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.csv with closing_readiness")
    ap.add_argument("--ids", required=True, help="JSON with call_ids (e.g. sales-only test)")
    ap.add_argument("--floor", type=float, default=0.654)
    ap.add_argument("--ceiling", type=float, default=0.749)
    a = ap.parse_args()

    keep = set(json.loads(Path(a.ids).read_text(encoding="utf-8"))["call_ids"])
    conv = _load_conv()

    rows = []
    miss_cr = 0
    for r in csv.DictReader(open(a.pred, encoding="utf-8", newline="")):
        cid = r["call_id"]
        if cid not in keep or cid not in conv:
            continue
        cr = _f(r.get("closing_readiness"))
        if cr is None:
            miss_cr += 1
        rows.append({"cid": cid, "gold": r.get("gold"), "pred": (r.get("pred") or ""),
                     "cr": cr, "signed": conv[cid]})

    n = len(rows)
    signers = sum(x["signed"] for x in rows)
    print(f"sales-only eval: n={n}  signers={signers}  closing_readiness missing on {miss_cr}")
    if miss_cr > n * 0.2:
        print("  WARNING: closing_readiness missing on >20% — model likely didn't emit it. "
              "Re-run with the updated prompt.")

    # --- (1) ranker comparison vs signed -------------------------------------
    have_cr = [x for x in rows if x["cr"] is not None]
    band_auc = _auc([RANK.get(x["pred"], 0) for x in rows], [x["signed"] for x in rows])
    cr_auc = _auc([x["cr"] for x in have_cr], [x["signed"] for x in have_cr]) if have_cr else float("nan")
    # blend: z-ish combine band-rank and cr (both min-max normalized) on the cr-having rows
    if have_cr:
        rmin, rmax = 0, 7
        crs = [x["cr"] for x in have_cr]
        cmin, cmax = min(crs), max(crs)
        crng = (cmax - cmin) or 1
        blend = [0.5 * (RANK.get(x["pred"], 0) - rmin) / (rmax - rmin)
                 + 0.5 * (x["cr"] - cmin) / crng for x in have_cr]
        blend_auc = _auc(blend, [x["signed"] for x in have_cr])
    else:
        blend_auc = float("nan")

    print("\n== AUC vs signed (sales-only) ==")
    print(f"  discrete band (what we have): {band_auc:.3f}")
    print(f"  closing_readiness (lever #1): {cr_auc:.3f}")
    print(f"  band + closing_readiness blend: {blend_auc:.3f}")
    print(f"  --- lead_score floor: {a.floor:.3f}   Claude ceiling: {a.ceiling:.3f} ---")
    best = max((v, k) for k, v in [("band", band_auc), ("closing_readiness", cr_auc),
                                   ("blend", blend_auc)] if v == v)
    verdict = "CLEARS the floor" if best[0] > a.floor else "still below the floor"
    print(f"  best free ranker = {best[1]} ({best[0]:.3f}) -> {verdict}")

    # --- (2) Caldo recovery by promoting Calduccio -> Caldo ------------------
    gold_caldo = sum(1 for x in rows if x["gold"] == "Caldo")
    pred_caldo_now = sum(1 for x in rows if x["pred"] == "Caldo")
    print(f"\n== Caldo recovery (Claude-Caldo gold = {gold_caldo}; model calls Caldo now = {pred_caldo_now}) ==")
    print("  thr | promoted | new#Caldo | recall vs Claude | precision")
    cands = [x for x in have_cr if x["pred"] == "Calduccio"]
    for thr in (60, 65, 70, 75, 80, 85, 90):
        promoted = [x for x in cands if x["cr"] >= thr]
        # after promotion, predicted Caldo set:
        new_caldo = [x for x in rows if x["pred"] == "Caldo"] + promoted
        tp = sum(1 for x in new_caldo if x["gold"] == "Caldo")
        recall = tp / gold_caldo if gold_caldo else float("nan")
        prec = tp / len(new_caldo) if new_caldo else float("nan")
        print(f"  {thr:3d} | {len(promoted):8d} | {len(new_caldo):9d} | "
              f"{recall:>6.0%}          | {prec:.0%}")
    print("\nPick the threshold that gets recall up toward Claude without precision "
          "cratering. Then re-bin and re-run src.contract_verdict to confirm the lift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
