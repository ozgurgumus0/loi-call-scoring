"""Build outputs/loi_compare.xlsx — a side-by-side view of every call we
have an LLM LOI score for, with the CRM LEAD_SCORE as the third comparator.

Per-call sheet columns: call_id, meeting_id, call_date, sonnet_outcome,
sonnet_score, qwen_outcome, qwen_score, agreement (sonnet vs qwen),
lead_score, lead_segmentation, lead_region, lead_producttype,
lead_evaluationtime, transcript_file, summary_file.

LEAD_SCORE caveat: it is a pre-call CRM model score (predicts conversion
from lead-form features). It is NOT a sales rep's post-call LOI judgment.
A column noting this is added to the `notes` sheet.

Reads:
    data/calls_enriched.csv (from src.join_crm)
    outputs/scores/zoom__*.json (Sonnet + Qwen)

Writes:
    outputs/loi_compare.xlsx (3 sheets: per_call, summary, notes)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA, OUT_SCORES, OUTPUTS

ENRICHED_CSV = DATA / "calls_enriched.csv"
OUT_XLSX = OUTPUTS / "loi_compare.xlsx"

OUTCOME_TO_SCORE = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7,
                    "Non interessato": 0, "Non idoneo": -1}


def _load_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sonnet_df, qwen_df) each indexed by call_id."""
    sonnet, qwen = [], []
    for p in sorted(OUT_SCORES.glob("zoom__*.json")):
        if p.name.startswith("upload__"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sc = d.get("score") or {}
        row = {
            "call_id":   d.get("call_id"),
            "outcome":   sc.get("outcome"),
            "score":     sc.get("score"),
            "confidence": sc.get("confidence"),
        }
        model = (d.get("model") or "").lower()
        if "qwen" in model or "ollama" in p.name:
            qwen.append(row)
        elif "claude-sonnet" in model:
            sonnet.append(row)
    s = pd.DataFrame(sonnet).rename(
        columns={"outcome": "sonnet_outcome", "score": "sonnet_score",
                 "confidence": "sonnet_confidence"})
    q = pd.DataFrame(qwen).rename(
        columns={"outcome": "qwen_outcome", "score": "qwen_score",
                 "confidence": "qwen_confidence"})
    return s, q


def _classify_agreement(row: pd.Series) -> str:
    s_out, q_out = row.get("sonnet_outcome"), row.get("qwen_outcome")
    if pd.isna(s_out) or pd.isna(q_out):
        return "single-model"
    if s_out == q_out:
        return "exact"
    excl = {"Non idoneo", "Non interessato"}
    if (s_out in excl) ^ (q_out in excl):
        return "excl-disagree"
    s_n = OUTCOME_TO_SCORE.get(s_out)
    q_n = OUTCOME_TO_SCORE.get(q_out)
    if s_n is not None and q_n is not None and abs(s_n - q_n) <= 2:
        return "in-band ±1"
    return "disagree"


def _summary_means(per_call: pd.DataFrame) -> pd.DataFrame:
    """For each scorer, mean LEAD_SCORE per outcome band."""
    rows = []
    for scorer in ["sonnet", "qwen"]:
        oc = f"{scorer}_outcome"
        sub = per_call[per_call[oc].notna() & per_call["lead_score"].notna()].copy()
        sub["lead_score"] = pd.to_numeric(sub["lead_score"], errors="coerce")
        g = sub.groupby(oc)["lead_score"].agg(["count", "mean"]).round(0)
        order = ["Freddo", "Tiepido", "Calduccio", "Caldo",
                 "Non interessato", "Non idoneo"]
        g = g.reindex([o for o in order if o in g.index])
        for outcome, r in g.iterrows():
            rows.append({
                "scorer": scorer,
                "outcome": outcome,
                "n": int(r["count"]),
                "mean_lead_score": r["mean"],
            })
    return pd.DataFrame(rows)


def _agreement_table(per_call: pd.DataFrame) -> pd.DataFrame:
    both = per_call[per_call["sonnet_outcome"].notna() & per_call["qwen_outcome"].notna()]
    if both.empty:
        return pd.DataFrame()
    ct = pd.crosstab(both["sonnet_outcome"], both["qwen_outcome"], dropna=False)
    ct.index.name = "sonnet_outcome"
    ct.columns.name = "qwen_outcome"
    return ct


def build() -> None:
    enriched = pd.read_csv(ENRICHED_CSV)
    enriched = enriched[enriched["source"] == "zoom"].copy()
    s, q = _load_scores()

    # Outer-merge sonnet + qwen so any call with at least one score is included
    scored = s.merge(q, on="call_id", how="outer")

    per_call = scored.merge(
        enriched[["call_id", "meeting_id", "call_date", "call_type",
                  "transcript_file", "summary_file",
                  "lead_score", "lead_segmentation", "lead_region",
                  "lead_province", "lead_producttype",
                  "lead_evaluationtime", "crm_netsc1"]],
        on="call_id", how="left",
    )
    per_call["agreement"] = per_call.apply(_classify_agreement, axis=1)

    # Column order tuned for side-by-side reading
    cols = [
        "call_id", "meeting_id", "call_date", "call_type",
        "sonnet_outcome", "sonnet_score", "sonnet_confidence",
        "qwen_outcome", "qwen_score", "qwen_confidence",
        "agreement",
        "lead_score", "lead_segmentation",
        "lead_region", "lead_province",
        "lead_producttype", "lead_evaluationtime",
        "crm_netsc1",
        "transcript_file", "summary_file",
    ]
    per_call = per_call[[c for c in cols if c in per_call.columns]]
    per_call = per_call.sort_values(
        ["sonnet_score", "qwen_score"], ascending=[False, False],
        na_position="last",
    )

    summary = _summary_means(per_call)
    agree_ct = _agreement_table(per_call)

    notes = pd.DataFrame({
        "note": [
            "LEAD_SCORE is a PRE-CALL CRM model score (predicts conversion "
            "from lead-form features). It is NOT a sales rep's post-call "
            "LOI judgment. Treat as a weak external comparator, not ground "
            "truth.",
            "Sonnet scores from 2026-05-25 dry-run (51 zoom calls). Qwen "
            "scores from 2026-05-28 dry-run (15 zoom calls, subset of Sonnet).",
            "agreement column: exact = same outcome; in-band ±1 = both "
            "in {Freddo, Tiepido, Calduccio, Caldo} with |score diff|≤2; "
            "excl-disagree = exactly one model said Non idoneo/Non "
            "interessato; disagree = bigger band gap; single-model = only "
            "one of Sonnet/Qwen scored this call.",
            "crm_netsc1 semantics still TBD. On the recorded subset it is "
            "may represent 'appointment held' rather than 'sale "
            "closed'. Do not use as a target.",
            "rep_loi (sales-people LOI) is NOT in this file. The sales-ops owner's "
            "FINAL SURVEY export is the source we are still waiting on.",
        ],
    })

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        per_call.to_excel(w, sheet_name="per_call", index=False)
        summary.to_excel(w, sheet_name="mean_lead_score_per_outcome", index=False)
        if not agree_ct.empty:
            agree_ct.to_excel(w, sheet_name="sonnet_x_qwen_crosstab")
        notes.to_excel(w, sheet_name="notes", index=False)

    print(f"per_call rows: {len(per_call)}")
    print(f"  with sonnet:        {per_call['sonnet_outcome'].notna().sum()}")
    print(f"  with qwen:          {per_call['qwen_outcome'].notna().sum()}")
    print(f"  with both:          {(per_call['sonnet_outcome'].notna() & per_call['qwen_outcome'].notna()).sum()}")
    print(f"  with lead_score:    {per_call['lead_score'].notna().sum()}")
    print()
    print("Agreement breakdown (sonnet vs qwen):")
    print(per_call['agreement'].value_counts().to_string())
    print()
    print("Mean LEAD_SCORE per outcome:")
    print(summary.to_string(index=False))
    print()
    print(f"[saved] {OUT_XLSX}")


if __name__ == "__main__":
    build()
