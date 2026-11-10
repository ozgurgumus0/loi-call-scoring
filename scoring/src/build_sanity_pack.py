"""Manual sanity-check pack for the Qwen LOI scores (user request 2026-06-11:
"read and look at LOI scores ... I don't want outliers that are too obviously
wrong").

Builds an xlsx of the calls most worth eyeballing, each with a transcript
head/tail snippet, the Qwen band + closing_readiness, Claude's band (where it
exists), and the contract outcome. Sheets target the failure modes, not random
reading:

  parse_fails        - Qwen emitted no parsable band (production retry would
                       recover most; here so none hide)
  caldo              - every call Qwen calls Caldo (the hot band that drives
                       action - must be right)
  excluded_signed    - Qwen said Non idoneo/Non interessato but the deal SIGNED
                       (the most damaging error class)
  big_disagreement   - Qwen vs Claude >= 2 bands apart (where one of them is
                       clearly wrong)
  cold_but_warm_cr   - band is cold yet closing_readiness is high (internal
                       inconsistency)
  sample_per_band    - a few random calls per Qwen band (calibration spot-check)

Usage:
  python -m src.build_sanity_pack --pred predictions_qwen3_8b_sales_full.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd

from .config import DATA, OUTPUTS
from .transcribe import CallRow, load_existing_transcript

RANK = {"Non idoneo": 0, "Non interessato": 0, "Freddo": 1, "Tiepido": 4,
        "Calduccio": 6, "Caldo": 7}
# Ordinal band POSITION (adjacent bands differ by 1) — used for "how many
# bands apart", so the warm-tilt Tiepido<->Calduccio counts as 1, not 2.
ORD = {"Non idoneo": 0, "Non interessato": 0, "Freddo": 1, "Tiepido": 2,
       "Calduccio": 3, "Caldo": 4}
EXCLUSIONS = {"Non idoneo", "Non interessato"}
SNIPPET_HEAD = 1800
SNIPPET_TAIL = 1200


def _snippet(transcript: str) -> str:
    t = transcript.strip()
    if len(t) <= SNIPPET_HEAD + SNIPPET_TAIL + 40:
        return t
    return (t[:SNIPPET_HEAD] + "\n\n[... middle omitted ...]\n\n"
            + t[-SNIPPET_TAIL:])


def _transcript_map() -> dict[str, str]:
    out = {}
    with (DATA / "calls.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source") == "zoom" and r.get("transcript_file"):
                out[r["call_id"]] = r["transcript_file"]
    return out


def _conv() -> dict[str, str]:
    out = {}
    with (DATA / "calls.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source") == "zoom":
                out[r["call_id"]] = (r.get("conversion_outcome") or "").strip()
    return out


def _claude_labels() -> dict[str, str]:
    import src.contract_verdict as CV
    return CV.load_claude_outcomes("zoom")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=str(OUTPUTS / "loi_sanity_pack.xlsx"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-band", type=int, default=6)
    a = ap.parse_args()
    random.seed(a.seed)

    preds = {r["call_id"]: r for r in csv.DictReader(open(a.pred, encoding="utf-8"))}
    tmap = _transcript_map()
    conv = _conv()
    claude = _claude_labels()

    # assemble one record per scored call
    recs = []
    for cid, p in preds.items():
        band = (p.get("pred") or "").strip()
        cr = p.get("closing_readiness")
        cr = float(cr) if cr not in (None, "", "NA") else None
        recs.append({
            "call_id": cid,
            "qwen_band": band or "(parse-fail)",
            "closing_readiness": cr,
            "claude_band": claude.get(cid, ""),
            "outcome": conv.get(cid, ""),
            "signed": conv.get(cid, "") == "signed",
        })
    df = pd.DataFrame(recs)

    def with_snippet(sub: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in sub.iterrows():
            tf = tmap.get(r["call_id"])
            snip = ""
            if tf:
                loaded = load_existing_transcript(CallRow(
                    source="zoom", call_id=r["call_id"], audio_file=None,
                    transcript_file=tf, duration_sec=None))
                if loaded:
                    snip = _snippet(loaded[0])
            d = r.to_dict()
            d["transcript_snippet"] = snip
            rows.append(d)
        cols = ["call_id", "qwen_band", "closing_readiness", "claude_band",
                "outcome", "signed", "transcript_snippet"]
        return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)

    sheets: dict[str, pd.DataFrame] = {}

    sheets["parse_fails"] = df[df["qwen_band"] == "(parse-fail)"]

    sheets["caldo"] = df[df["qwen_band"] == "Caldo"]

    sheets["excluded_signed"] = df[df["qwen_band"].isin(EXCLUSIONS) & df["signed"]]

    # Qwen vs Claude >= 2 bands apart (both present, in-band)
    both = df[(df["claude_band"] != "") & (df["qwen_band"] != "(parse-fail)")].copy()
    both["gap"] = both.apply(
        lambda r: abs(ORD.get(r["qwen_band"], 0) - ORD.get(r["claude_band"], 0)),
        axis=1)
    sheets["big_disagreement"] = both[both["gap"] >= 2].sort_values("gap", ascending=False).drop(columns="gap")

    # cold band but high closing_readiness (internal inconsistency)
    cold = df[df["qwen_band"].isin(EXCLUSIONS | {"Freddo", "Tiepido"})]
    sheets["cold_but_warm_cr"] = cold[cold["closing_readiness"].fillna(0) >= 65]

    # random sample per band
    samp = []
    for band, g in df[df["qwen_band"] != "(parse-fail)"].groupby("qwen_band"):
        samp.append(g.sample(min(a.per_band, len(g)), random_state=a.seed))
    sheets["sample_per_band"] = pd.concat(samp) if samp else df.head(0)

    out = Path(a.out)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary = pd.DataFrame(
            [{"sheet": k, "n_calls": len(v)} for k, v in sheets.items()])
        summary.to_excel(w, sheet_name="_summary", index=False)
        for name, sub in sheets.items():
            with_snippet(sub).to_excel(w, sheet_name=name[:31], index=False)

    print(f"[saved] {out}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
