"""Contract verdict — does an LOI model's band actually predict who signs?

Agreement with teacher labels and performance against held-out outcomes are separate measures.

So we measure the model's bands against the OBJECTIVE outcome
(`conversion_outcome == 'signed'` in calls.csv):

    * band -> conversion table (signed / total, %), ordered cold -> hot
    * base rate
    * ordinal-score AUC (Mann-Whitney) of the score vs the signed flag
    * signer concentration: what share of all signers land in Calduccio+Caldo
      and in Caldo alone (the follow-up-the-warm-leads recall)
    * when a model CSV is scored, a SIDE-BY-SIDE vs Claude on the IDENTICAL call
      set so the comparison is apples-to-apples.

Two scores sources:
    --model claude            read Sonnet outcomes from outputs/scores/zoom__*.json
    --model <name> --pred X   read a predictions CSV with columns
                              call_id, source, pred  (gold/pred_score/confidence
                              optional). Produced by kaggle/eval_only.py.

Restrict the call universe with --ids data/train/test_ids.json (the frozen
held-out test set — leakage-clean and signer-dense) or omit for the full
contract-bearing corpus.

Usage:
    python -m src.contract_verdict --model claude --ids data/train/test_ids.json
    python -m src.contract_verdict --model claude            # full corpus
    python -m src.contract_verdict --model qwen-7b-v3 --pred preds_v3.csv \
        --ids data/train/test_ids.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from .config import CALLS_CSV, OUT_SCORES, OUT_REPORTS, OUTCOME_SCORES, DATA

# Cold -> hot display order. Exclusions sit at the cold floor for ranking.
BAND_ORDER = ["Non idoneo", "Non interessato", "Freddo", "Tiepido",
              "Calduccio", "Caldo"]
# Numeric rank for AUC. In-band uses the authored OUTCOME_SCORES; exclusions
# map to 0 (below Freddo=1) — "no LOI" floor. Ties handled by mid-rank.
RANK = {"Non idoneo": 0, "Non interessato": 0, **OUTCOME_SCORES}


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_conversion() -> dict[tuple[str, str], str]:
    """{(source, call_id) -> conversion_outcome} for rows that have one."""
    out: dict[tuple[str, str], str] = {}
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            cv = (r.get("conversion_outcome") or "").strip()
            if cv:
                out[(r["source"], r["call_id"])] = cv
    return out


def load_claude_outcomes(source: str = "zoom") -> dict[str, str]:
    """{call_id -> Sonnet outcome} from outputs/scores/<source>__*.json.

    Deduplicated by call_id (first in sorted order wins), matching the training
    builder's dedup so the call universe is identical.
    """
    out: dict[str, str] = {}
    for p in sorted(glob.glob(str(OUT_SCORES / f"{source}__*.json"))):
        name = os.path.basename(p)
        if "__ollama_" in name or name.startswith("_"):
            continue
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (d.get("model") or "").lower().startswith("claude-sonnet"):
            continue
        sc = d.get("score") or {}
        cid = d.get("call_id")
        if cid and cid not in out and isinstance(sc, dict) and sc.get("outcome"):
            out[cid] = sc["outcome"]
    return out


def load_pred_outcomes(pred_csv: str) -> tuple[dict[str, str], str, list[str]]:
    """Read a model predictions CSV -> ({call_id -> pred outcome}, source, fails).

    Required columns: call_id, pred (or 'outcome'). Optional: source (default
    'zoom'). `fails` = call_ids that HAVE a row but no parseable prediction
    (the model emitted unparseable JSON). These are returned, NOT silently
    dropped: a model that fails to parse its hottest calls would otherwise look
    artificially good (those rows just vanish from the verdict). The caller
    surfaces the count and treats the metrics as an optimistic bound.
    """
    out: dict[str, str] = {}
    fails: list[str] = []
    source = "zoom"
    with open(pred_csv, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        cols = {c.lower(): c for c in (r.fieldnames or [])}
        pred_col = cols.get("pred") or cols.get("outcome") or cols.get("pred_outcome")
        cid_col = cols.get("call_id") or cols.get("id")
        src_col = cols.get("source")
        if not pred_col or not cid_col:
            raise SystemExit(f"{pred_csv}: need columns call_id and pred/outcome "
                             f"(found {r.fieldnames})")
        for row in r:
            cid = (row[cid_col] or "").strip()
            pred = (row[pred_col] or "").strip()
            if src_col:
                source = (row[src_col] or source).strip() or source
            if not cid:
                continue
            if pred and pred.lower() not in ("none", "null", "nan"):
                out[cid] = pred
            else:
                fails.append(cid)
    return out, source, fails


def load_lead_scores(source: str = "zoom") -> dict[str, float]:
    """{call_id -> CRM lead_score} from master.csv (the pre-call grade).

    Used as a FLOOR baseline: the LOI model only earns its keep if its score
    predicts signing better than this free, pre-call CRM number. Best-effort —
    returns {} if master.csv is absent or has no lead_score column.
    """
    out: dict[str, float] = {}
    path = DATA / "master.csv"
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "lead_score" not in (r.fieldnames or []):
            return out
        for row in r:
            if row.get("source") != source:
                continue
            ls = (row.get("lead_score") or "").strip()
            if ls:
                try:
                    out[row["call_id"]] = float(ls)
                except ValueError:
                    pass
    return out


def baseline_lead_auc(conv: dict[tuple[str, str], str], source: str,
                      ids: set[str] | None) -> tuple[float | None, int, int]:
    """AUC of CRM lead_score vs signed on the same universe -> (auc, n, signers)."""
    lead = load_lead_scores(source)
    universe = ids if ids is not None else {c for (s, c) in conv if s == source}
    s, y = [], []
    for cid in universe:
        cv = conv.get((source, cid))
        if cv is None or cid not in lead:
            continue
        s.append(lead[cid]); y.append(1 if cv == "signed" else 0)
    return auc_mann_whitney(s, y), len(s), sum(y)


def load_id_filter(ids_json: str | None) -> tuple[set[str] | None, str | None]:
    """Frozen-test (or any) call-id allowlist -> (set, source) or (None, None)."""
    if not ids_json:
        return None, None
    d = json.loads(Path(ids_json).read_text(encoding="utf-8"))
    return set(d.get("call_ids", [])), d.get("source")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auc_mann_whitney(scores: list[float], labels: list[int]) -> float | None:
    """Probabilistic AUC = P(score(signer) > score(non-signer)), ties=0.5.

    Computed via the rank-sum identity with mid-ranks for ties — no sklearn
    dependency. Returns None if a class is empty.
    """
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                  # average rank, 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n. Returns (lo, hi) in [0,1].

    The band rates here rest on tiny n (Caldo n~18), so a point estimate alone
    invites over-reading "Caldo 72%". Wilson is well-behaved at small n / extreme p.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_delta(score_a: list[float], score_b: list[float],
                           labels: list[int], n_boot: int = 5000,
                           seed: int = 0) -> dict | None:
    """Paired bootstrap on AUC_b - AUC_a over the SAME calls (a=Claude, b=model).

    Claude and the model score the identical calls/signers, so the two AUCs are
    correlated — comparing them as independent samples understates power. We
    resample CALLS (rows) with replacement, recompute both AUCs on each resample,
    and summarise the delta distribution. Returns observed delta, 95% percentile
    CI, and P(model worse than Claude). None if a class is empty.
    """
    base = auc_mann_whitney
    obs_a, obs_b = base(score_a, labels), base(score_b, labels)
    if obs_a is None or obs_b is None:
        return None
    rng = random.Random(seed)
    m = len(labels)
    deltas = []
    idx = list(range(m))
    for _ in range(n_boot):
        pick = [rng.choice(idx) for _ in range(m)]
        la = [labels[i] for i in pick]
        if sum(la) == 0 or sum(la) == len(la):
            continue
        a = base([score_a[i] for i in pick], la)
        b = base([score_b[i] for i in pick], la)
        if a is not None and b is not None:
            deltas.append(b - a)
    if not deltas:
        return None
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    p_worse = sum(1 for d in deltas if d < 0) / len(deltas)
    return {"auc_a": obs_a, "auc_b": obs_b, "delta": obs_b - obs_a,
            "ci_lo": lo, "ci_hi": hi, "p_model_worse": p_worse,
            "n_boot": len(deltas)}


def verdict_table(outcomes: dict[str, str], conv: dict[tuple[str, str], str],
                  source: str, ids: set[str] | None):
    """Build the band->conversion table + AUC + concentration for one model.

    Restricts to calls that (a) are in `ids` (if given), (b) have a predicted
    outcome, and (c) have a conversion outcome. Returns a dict bundle.
    """
    universe = ids if ids is not None else {c for (s, c) in conv if s == source}
    rows = []   # (outcome, signed_bool)
    for cid in universe:
        o = outcomes.get(cid)
        cv = conv.get((source, cid))
        if o is None or cv is None:
            continue
        rows.append((o, 1 if cv == "signed" else 0))

    by_band: dict[str, list[int]] = defaultdict(list)
    for o, s in rows:
        by_band[o].append(s)

    n = len(rows)
    total_signed = sum(s for _, s in rows)
    table = []
    for band in BAND_ORDER:
        flags = by_band.get(band, [])
        if not flags:
            continue
        table.append((band, sum(flags), len(flags)))
    # any unexpected outcome label not in BAND_ORDER
    for band in sorted(set(by_band) - set(BAND_ORDER)):
        flags = by_band[band]
        table.append((band, sum(flags), len(flags)))

    scores = [RANK.get(o, 0) for o, _ in rows]
    labels = [s for _, s in rows]
    auc = auc_mann_whitney(scores, labels)

    warm = sum(s for o, s in rows if o in ("Calduccio", "Caldo"))
    hot = sum(s for o, s in rows if o == "Caldo")
    return {
        "n": n, "signed": total_signed,
        "base_rate": (total_signed / n) if n else 0.0,
        "table": table, "auc": auc,
        "warm_recall": (warm / total_signed) if total_signed else 0.0,
        "caldo_recall": (hot / total_signed) if total_signed else 0.0,
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_table(b: dict, label: str) -> str:
    auc_line = (f"- ordinal-score AUC vs signed: **{b['auc']:.3f}**"
                if b['auc'] is not None
                else "- ordinal-score AUC vs signed: n/a (one class empty)")
    lines = [f"### {label}",
             "",
             f"- n with conversion + score: **{b['n']}**  "
             f"signers: **{b['signed']}**  base rate: **{b['base_rate']:.1%}**",
             auc_line,
             f"- signer concentration: **{b['warm_recall']:.0%}** of signers are "
             f"Calduccio+Caldo; **{b['caldo_recall']:.0%}** are Caldo",
             "",
             "| Band | Signed / total | Conversion (95% CI) |",
             "|---|---|---|"]
    for band, s, tot in b["table"]:
        lo, hi = wilson_ci(s, tot)
        lines.append(f"| {band} | {s}/{tot} | {100*s/tot:.0f}% "
                     f"[{100*lo:.0f}–{100*hi:.0f}] |")
    lines.append("")
    return "\n".join(lines)


def _decision(model_auc, floor_auc, claude_auc, model_caldo, claude_caldo,
              p_worse) -> str:
    """Heuristic ship gate. Floor = free CRM lead_score; ceiling = Claude.

    The LOI model only earns deployment if its SCORE beats the free pre-call
    baseline AND it preserves the Caldo concentration (the band that actually
    pays). Thresholds are deliberately stated so the user can override.
    """
    if model_auc is None or floor_auc is None or claude_auc is None:
        return "Decision gate: n/a (missing AUC — need both floor and a model)."
    span = max(1e-9, claude_auc - floor_auc)
    retained = (model_auc - floor_auc) / span          # frac of teacher's lift kept
    caldo_ret = (model_caldo / claude_caldo) if claude_caldo else 0.0
    bits = [f"floor(lead_score)=**{floor_auc:.3f}**, model=**{model_auc:.3f}**, "
            f"ceiling(Claude)=**{claude_auc:.3f}** → model keeps "
            f"**{retained:.0%}** of the teacher's marginal lift over the floor; "
            f"Caldo concentration retained **{caldo_ret:.0%}**"
            + (f"; P(model worse than Claude)=**{p_worse:.0%}**" if p_worse is not None else "")
            + "."]
    if model_auc <= floor_auc + 0.005 or caldo_ret < 0.5:
        verdict = ("🔴 **NO-SHIP.** The score adds ~nothing over the free CRM "
                   "lead grade and/or collapses the Caldo band where signers "
                   "concentrate. Use Claude in production; pursue the real "
                   "ceiling lever (audio / a cleanly-fitting newer base).")
    elif retained >= 0.70 and caldo_ret >= 0.80 and (p_worse is None or p_worse <= 0.90):
        verdict = ("🟢 **SHIP-candidate.** Keeps most of the teacher's "
                   "contract-discrimination AND preserves the Caldo band. "
                   "Validate on a fresh time-bounded slice before full cutover.")
    else:
        verdict = ("🟡 **YELLOW.** Partial value: beats the floor but loses "
                   "Caldo concentration or trails Claude meaningfully. Options: "
                   "deploy with a Claude fallback ONLY on Caldo/grey-zone calls "
                   "(note: conflicts with the no-Claude-spend constraint), or "
                   "invest in the audio lever before shipping standalone.")
    return verdict + "\n\n" + bits[0]


def main() -> int:
    try:                                  # emojis in the gate -> utf-8 console
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="'claude' (read outputs/scores) or a model name "
                         "(needs --pred).")
    ap.add_argument("--pred", help="predictions CSV for a non-claude model "
                                   "(cols: call_id, source, pred[, gold]).")
    ap.add_argument("--ids", help="JSON with {'call_ids': [...], 'source': ...} "
                                  "to restrict the universe (e.g. the frozen "
                                  "test set). Omit for the full corpus.")
    ap.add_argument("--source", default="zoom")
    ap.add_argument("--out", help="markdown report path "
                                  "(default outputs/reports/10_contract_verdict.md)")
    a = ap.parse_args()

    conv = load_conversion()
    ids, ids_source = load_id_filter(a.ids)
    source = ids_source or a.source

    is_claude = a.model.lower() == "claude"
    fails: list[str] = []
    if is_claude:
        outcomes = load_claude_outcomes(source)
        model_label = "Claude (Sonnet teacher)"
    else:
        if not a.pred:
            raise SystemExit("--pred is required for a non-claude model")
        outcomes, psrc, fails = load_pred_outcomes(a.pred)
        source = ids_source or psrc
        model_label = a.model

    scope = (f"frozen test set ({a.ids})" if a.ids
             else f"full {source} contract-bearing corpus")
    bundle = verdict_table(outcomes, conv, source, ids)

    # FLOOR baseline: does the free pre-call CRM lead_score predict signing as
    # well? The LOI model only earns deployment if it beats this.
    base_auc, base_n, base_sign = baseline_lead_auc(conv, source, ids)
    base_line = (f"CRM lead_score AUC (floor): **{base_auc:.3f}** "
                 f"(n={base_n}, signers={base_sign})"
                 if base_auc is not None else
                 "CRM lead_score AUC (floor): n/a (master.csv missing lead_score)")

    md = [f"# Contract verdict — {model_label}", "",
          f"Scope: **{scope}**, source=`{source}`.", "",
          "What this answers: do the model's LOI bands actually rank who "
          "signs? The signal that matters for production lives in the warm/hot "
          "end (esp. **Caldo**); a model that under-calls Caldo flattens the "
          "gradient even if its label-agreement looks fine.", "",
          f"_{base_line}_  — the LOI score must beat this free pre-call number "
          "to justify itself.", "",
          render_table(bundle, model_label)]

    # Parse-failure accounting — NEVER silently dropped (a model that fails on
    # its hottest calls would otherwise look artificially good).
    if not is_claude and fails:
        failed_signers = [c for c in fails if conv.get((source, c)) == "signed"]
        warn = (f"⚠️ **{len(fails)} prediction(s) unparseable** and excluded "
                f"({len(failed_signers)} of them are signers). Metrics are an "
                f"OPTIMISTIC bound — in production the retry/format:json path "
                f"recovers most; rerun the eval if the count is high.")
        md += ["", warn, ""]
        print("\n" + warn.replace("**", ""))

    print(f"\n===== CONTRACT VERDICT — {model_label} =====")
    print(f"scope: {scope}  source={source}")
    print(base_line.replace("**", ""))
    print(render_table(bundle, model_label))

    # apples-to-apples vs Claude on the IDENTICAL call set
    if not is_claude:
        claude = load_claude_outcomes(source)
        # restrict Claude to exactly the calls the model produced a pred for
        common = {c for c in (ids if ids is not None else set(outcomes))
                  if c in outcomes and c in claude}
        cb = verdict_table(claude, conv, source, common)
        mb = verdict_table(outcomes, conv, source, common)

        # paired bootstrap on AUC delta (same calls scored by both models)
        pair = sorted(c for c in common if conv.get((source, c)) is not None)
        sa = [RANK.get(claude.get(c), 0) for c in pair]
        sb = [RANK.get(outcomes.get(c), 0) for c in pair]
        ly = [1 if conv.get((source, c)) == "signed" else 0 for c in pair]
        bs = paired_bootstrap_delta(sa, sb, ly)
        if bs:
            delta_line = (
                f"- **Paired AUC test** (bootstrap, same {len(pair)} calls): "
                f"Claude **{bs['auc_a']:.3f}** vs {model_label} "
                f"**{bs['auc_b']:.3f}**, Δ **{bs['delta']:+.3f}** "
                f"(95% CI [{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}]); "
                f"P({model_label} worse) = **{bs['p_model_worse']:.0%}**.")
        else:
            delta_line = (f"- AUC: Claude **{cb['auc']:.3f}** vs {model_label} "
                          f"**{mb['auc']:.3f}** (paired test n/a).")

        # decision gate — floor is LEAD_SCORE, ceiling is Claude
        gate = _decision(mb['auc'], base_auc, cb['auc'],
                         mb['caldo_recall'], cb['caldo_recall'],
                         bs['p_model_worse'] if bs else None)

        md += ["---", "",
               "## Side-by-side on the identical call set", "",
               f"Both restricted to the {len(common)} calls scored by BOTH "
               f"{model_label} and Claude (with a conversion outcome).", "",
               render_table(cb, "Claude (Sonnet teacher)"),
               render_table(mb, model_label),
               "## Read", "",
               delta_line,
               f"- Caldo recall (share of signers called Caldo): Claude "
               f"**{cb['caldo_recall']:.0%}** vs {model_label} "
               f"**{mb['caldo_recall']:.0%}**.",
               f"- Warm+hot recall: Claude **{cb['warm_recall']:.0%}** vs "
               f"{model_label} **{mb['warm_recall']:.0%}**.", "",
               "## Decision gate", "", gate, ""]
        print(render_table(cb, "Claude (same calls)"))
        print(render_table(mb, f"{model_label} (same calls)"))
        print(delta_line.replace("**", ""))
        print("\n" + gate.replace("**", ""))

    out_path = Path(a.out) if a.out else OUT_REPORTS / "10_contract_verdict.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
