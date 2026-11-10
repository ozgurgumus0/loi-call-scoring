"""Phase 5 — evaluation report (v3 metrics).

Reads outputs/scores/<source>__<call_id>.json + data/calls.csv and produces
outputs/reports/03_eval.md.

Metrics (v3):
- Quadratic weighted Cohen's κ (model vs rep, when rep labels exist).
- Exact and ±1-band accuracy.
- Distribution KL divergence vs the authored 30/50/15/5.
- AUC(model -> conversion), AUC(rep -> conversion), AUC(zoom_ai -> conversion).
- Confidence calibration (model confidence vs empirical agreement).
- Disagreement analysis (|model_score - rep_score| >= 3).

When rep labels / conversion outcomes are absent (current state), the script
degrades gracefully: it still emits the distribution check + per-source
outcome counts + Sonnet-vs-Opus comparison if both exist.

Spearman ρ is computed but reported as a footnote — weak under heavy ties.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

from .config import (
    CALLS_CSV, EXPECTED_DIST, OUT_REPORTS, OUT_SCORES, OUTCOME_SCORES,
    RUBRIC_VERSION, ensure_dirs,
)

log = logging.getLogger("evaluate")


# --------------------------------------------------------------- loading ---
def _load_scores() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(OUT_SCORES.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.warning("skip unparseable score file: %s", p.name)
    return rows


def _load_labels() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not CALLS_CSV.exists():
        return out
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["source"], row["call_id"])] = row
    return out


# --------------------------------------------------------------- metrics ---
def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    eps = 1e-9
    keys = set(p) | set(q)
    return sum(p.get(k, eps) * math.log((p.get(k, eps)) / (q.get(k, eps)))
               for k in keys if p.get(k, 0) > 0)


def empirical_dist(outcomes: list[str], keys=("Freddo", "Tiepido",
                                              "Calduccio", "Caldo")) -> dict[str, float]:
    counted = Counter(o for o in outcomes if o in keys)
    n = sum(counted.values()) or 1
    return {k: counted.get(k, 0) / n for k in keys}


def weighted_kappa(y_rep: list[int], y_model: list[int]) -> float | None:
    if not y_rep:
        return None
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        log.warning("scikit-learn not installed; skipping weighted kappa")
        return None
    return cohen_kappa_score(y_rep, y_model, weights="quadratic")


def exact_and_one_off(y_rep: list[int], y_model: list[int]) -> tuple[float, float]:
    if not y_rep:
        return (float("nan"), float("nan"))
    n = len(y_rep)
    exact = sum(a == b for a, b in zip(y_rep, y_model)) / n
    # Treat ±1-band as "within one tree step" — 1→4 is one band away, etc.
    bands = [1, 4, 6, 7]
    idx = {s: i for i, s in enumerate(bands)}
    one_off = sum(
        abs(idx.get(a, -99) - idx.get(b, -99)) <= 1
        for a, b in zip(y_rep, y_model)
        if a in idx and b in idx
    ) / n
    return (exact, one_off)


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney U based AUC. labels: 1=positive, 0=negative."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1
            elif sp == sn:
                wins += 0.5
    return wins / (len(pos) * len(neg))


# --------------------------------------------------------------- report ----
def build_report() -> Path:
    ensure_dirs()
    scores = _load_scores()
    labels = _load_labels()
    if not scores:
        report = ("# 03 — Evaluation\n\n"
                  "**No scores found** in `outputs/scores/`. "
                  "Run `python -m src.score_llm ...` or `python -m src.dry_run_scoring` "
                  "first.\n")
        out = OUT_REPORTS / "03_eval.md"
        out.write_text(report, encoding="utf-8")
        return out

    # group by model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        by_model[s["model"]].append(s)

    lines: list[str] = []
    lines.append("# 03 — Evaluation\n")
    lines.append(f"_Rubric version: `{RUBRIC_VERSION}`_  ")
    lines.append(f"_N scored: {len(scores)} across {len(by_model)} model(s)_\n")

    # ---- distribution per model
    lines.append("## Distribution vs authored (30 / 50 / 15 / 5)\n")
    lines.append("| Model | N | Freddo | Tiepido | Calduccio | Caldo | KL(model‖authored) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model, ss in by_model.items():
        outs = [s["score"]["outcome"] for s in ss]
        emp = empirical_dist(outs)
        kl = kl_divergence(emp, EXPECTED_DIST)
        lines.append(f"| {model} | {len(ss)} | "
                     f"{emp['Freddo']:.0%} | {emp['Tiepido']:.0%} | "
                     f"{emp['Calduccio']:.0%} | {emp['Caldo']:.0%} | {kl:.3f} |")
    lines.append("")

    # ---- per-source breakdown
    lines.append("## Outcome counts by source\n")
    lines.append("| Model | Source | Non idoneo | Non int. | Freddo | Tiepido | Calduccio | Caldo |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for model, ss in by_model.items():
        by_src: dict[str, Counter] = defaultdict(Counter)
        for s in ss:
            by_src[s["source"]][s["score"]["outcome"]] += 1
        for src, ctr in by_src.items():
            lines.append(f"| {model} | {src} | "
                         f"{ctr.get('Non idoneo',0)} | {ctr.get('Non interessato',0)} | "
                         f"{ctr.get('Freddo',0)} | {ctr.get('Tiepido',0)} | "
                         f"{ctr.get('Calduccio',0)} | {ctr.get('Caldo',0)} |")
    lines.append("")

    # ---- agreement vs rep (only if any rep labels exist)
    have_rep = any(
        labels.get((s["source"], s["call_id"]), {}).get("rep_loi", "")
        for s in scores
    )
    if have_rep:
        lines.append("## Agreement vs rep LOI\n")
        for model, ss in by_model.items():
            y_rep, y_model = [], []
            for s in ss:
                lab = labels.get((s["source"], s["call_id"]), {})
                rl = str(lab.get("rep_loi") or "").strip()
                if not rl:
                    continue
                # rep_loi may be a band name ("Tiepido") or a number ("4").
                if rl in OUTCOME_SCORES:
                    rep_score = OUTCOME_SCORES[rl]
                else:
                    try:
                        rep_score = int(float(rl))
                    except ValueError:
                        continue
                y_rep.append(rep_score)
                y_model.append(int(s["score"].get("score") or 0))
            if y_rep:
                kappa = weighted_kappa(y_rep, y_model)
                exact, one_off = exact_and_one_off(y_rep, y_model)
                k_str = f"{kappa:.3f}" if kappa is not None else "n/a"
                lines.append(f"- **{model}** (n={len(y_rep)}): "
                             f"weighted κ = {k_str}, exact = {exact:.1%}, "
                             f"±1-band = {one_off:.1%}")
            else:
                lines.append(f"- **{model}** (n=0): no usable rep labels.")
        lines.append("")
    else:
        lines.append("## Agreement vs rep LOI\n\n"
                     "_No rep labels in `data/calls.csv` yet — skipped._\n")

    # ---- AUC vs conversion
    have_conv = any(
        labels.get((s["source"], s["call_id"]), {}).get("conversion_outcome", "")
        for s in scores
    )
    if have_conv:
        lines.append("## AUC vs conversion outcome\n")
        for model, ss in by_model.items():
            yhat, y = [], []
            conv_map = {"signed": 1, "not_signed": 0, "1": 1, "0": 0,
                        "true": 1, "false": 0}
            for s in ss:
                lab = labels.get((s["source"], s["call_id"]), {})
                co = str(lab.get("conversion_outcome") or "").strip().lower()
                if not co:
                    continue
                if co in conv_map:
                    y.append(conv_map[co])
                else:
                    try:
                        y.append(int(co))
                    except ValueError:
                        continue
                yhat.append(float(s["score"].get("score") or 0))
            a = auc(yhat, y)
            lines.append(f"- **{model}** AUC(model_score → conversion) "
                         f"(n={len(y)}): {a if a is None else f'{a:.3f}'}")
        lines.append("")
    else:
        lines.append("## AUC vs conversion outcome\n\n"
                     "_No conversion_outcome in `data/calls.csv` yet — skipped. "
                     "See Phase 5b (CRM-conversion track) in the v3 workflow._\n")

    # ---- cost
    total_cost = sum(s.get("cost_usd", 0) for s in scores)
    lines.append(f"## Cost\n\nTotal Anthropic spend across scored calls: "
                 f"**${total_cost:.4f}**\n")

    out = OUT_REPORTS / "03_eval.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = build_report()
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
