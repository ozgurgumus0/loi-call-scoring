"""Build `data/master.csv` — one row per call with every label / feature we
currently have, joined together. This is the single source of truth for
"what labels does each call have right now".

Joins:
    data/calls_enriched.csv   (calls.csv + CRM xlsx via src.join_crm)
    outputs/scores/*.json     (Sonnet + Qwen + any backend, both genesys & zoom)

Columns (in order):
    source, call_id, call_date, call_type, duration_sec, has_transcript,
    has_audio, agent_name, host_email,
    sonnet_outcome, sonnet_score, sonnet_confidence, sonnet_model,
    qwen_outcome,   qwen_score,   qwen_confidence,   qwen_model,
    agreement,
    crm_netsc1, lead_score, lead_segmentation, lead_region, lead_province,
    lead_producttype, lead_evaluationtime, lead_status,
    transcript_file, summary_file, audio_file
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA, OUT_SCORES

ENRICHED_CSV = DATA / "calls_enriched.csv"
MASTER_CSV = DATA / "master.csv"

OUTCOME_TO_NUM = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7,
                  "Non interessato": 0, "Non idoneo": -1}
EXCL = {"Non idoneo", "Non interessato"}


def _scan_scores() -> dict[str, list[dict]]:
    """Return {'sonnet': [...], 'qwen': [...]} keyed by backend family."""
    out: dict[str, list[dict]] = {"sonnet": [], "qwen": []}
    for p in sorted(OUT_SCORES.glob("*.json")):
        name = p.name
        if name.startswith("upload__") or name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sc = d.get("score") or {}
        model = (d.get("model") or "").lower()
        family: str | None = None
        if "qwen" in model or "__ollama_" in name:
            family = "qwen"
        elif "claude-sonnet" in model:
            family = "sonnet"
        if family is None:
            continue
        out[family].append({
            "call_id":    d.get("call_id"),
            "outcome":    sc.get("outcome"),
            "score":      sc.get("score"),
            "confidence": sc.get("confidence"),
            "model":      d.get("model"),
        })
    return out


def _to_df(rows: list[dict], prefix: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["call_id", f"{prefix}_outcome",
                                     f"{prefix}_score", f"{prefix}_confidence",
                                     f"{prefix}_model"])
    df = pd.DataFrame(rows).rename(columns={
        "outcome":    f"{prefix}_outcome",
        "score":      f"{prefix}_score",
        "confidence": f"{prefix}_confidence",
        "model":      f"{prefix}_model",
    })
    # If multiple score files exist for the same call_id, keep the latest
    # (lexicographic on model name is a reasonable proxy here).
    df = df.sort_values("call_id").drop_duplicates(subset="call_id", keep="last")
    return df


def _classify_agreement(row: pd.Series) -> str:
    s_out, q_out = row.get("sonnet_outcome"), row.get("qwen_outcome")
    if pd.isna(s_out) and pd.isna(q_out):
        return "unscored"
    if pd.isna(s_out) or pd.isna(q_out):
        return "single-model"
    if s_out == q_out:
        return "exact"
    if (s_out in EXCL) ^ (q_out in EXCL):
        return "excl-disagree"
    s_n, q_n = OUTCOME_TO_NUM.get(s_out), OUTCOME_TO_NUM.get(q_out)
    if s_n is not None and q_n is not None and abs(s_n - q_n) <= 2:
        return "in-band +/-1"
    return "disagree"


def build() -> dict:
    enriched = pd.read_csv(ENRICHED_CSV)
    scores = _scan_scores()
    s = _to_df(scores["sonnet"], "sonnet")
    q = _to_df(scores["qwen"],   "qwen")

    df = enriched.merge(s, on="call_id", how="left") \
                 .merge(q, on="call_id", how="left")
    df["agreement"] = df.apply(_classify_agreement, axis=1)
    df["has_transcript"] = df["transcript_file"].notna() & (df["transcript_file"] != "")
    df["has_audio"] = df["audio_file"].notna() & (df["audio_file"] != "")

    # Fixed front-of-table columns (most important for inspection),
    # then EVERY remaining column from calls_enriched (so the full xlsx
    # payload — all lead_*, crm_*, calendar event fields — flows through).
    front = [
        "source", "call_id", "call_date", "call_type", "duration_sec",
        "has_transcript", "has_audio", "agent_name", "host_email",
        "sonnet_outcome", "sonnet_score", "sonnet_confidence", "sonnet_model",
        "qwen_outcome",   "qwen_score",   "qwen_confidence",   "qwen_model",
        "agreement",
        "crm_netsc1", "crm_grosssc1", "crm_appt_date", "crm_lead_id",
        "lead_score", "lead_segmentation", "lead_status",
        "lead_region", "lead_province",
        "lead_producttype", "lead_evaluationtime",
    ]
    tail = ["transcript_file", "summary_file", "audio_file"]
    seen = set(front) | set(tail)
    middle = [c for c in df.columns if c not in seen]
    cols = [c for c in front if c in df.columns] + middle + \
           [c for c in tail if c in df.columns]
    df = df[cols]

    # Sort: scored calls first (sonnet then qwen), then unscored
    df["_sort_key"] = df["sonnet_score"].fillna(-99)
    df = df.sort_values(["_sort_key", "qwen_score"],
                        ascending=[False, False],
                        na_position="last").drop(columns="_sort_key")

    df.to_csv(MASTER_CSV, index=False)

    return {
        "rows":                   len(df),
        "zoom":                   int((df["source"] == "zoom").sum()),
        "genesys":                int((df["source"] == "genesys").sum()),
        "with_sonnet_label":      int(df["sonnet_outcome"].notna().sum()),
        "with_qwen_label":        int(df["qwen_outcome"].notna().sum()),
        "with_both_labels":       int((df["sonnet_outcome"].notna()
                                       & df["qwen_outcome"].notna()).sum()),
        "with_crm_features":      int(df["lead_score"].notna().sum()),
        "with_transcript":        int(df["has_transcript"].sum()),
        "with_audio":             int(df["has_audio"].sum()),
        "out_file":               str(MASTER_CSV),
    }


def _main() -> int:
    r = build()
    width = max(len(k) for k in r)
    for k, v in r.items():
        print(f"{k:<{width}}  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
