"""Phase 0 dry-run: score N calls and check the outcome distribution.

Uses existing Phase-1 transcripts (Genesys .txt or Zoom VTT). No Whisper.
This is the v3 §6 "do this right now" action.

If the model's distribution lands wildly off the authored 30/50/15/5, fix
the rubric prompt BEFORE scaling up.

    python -m src.dry_run_scoring --n 50              # mixed sources
    python -m src.dry_run_scoring --n 50 --source zoom
    python -m src.dry_run_scoring --n 50 --source genesys
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from collections import Counter
from pathlib import Path

from .config import (
    BACKEND_ANTHROPIC, BACKEND_OLLAMA, CALLS_CSV, DEFAULT_BACKEND,
    EXPECTED_DIST, OLLAMA_MODEL, OUT_LOGS, OUT_REPORTS, SCORING_MODEL,
    ensure_dirs,
)
from .evaluate import empirical_dist, kl_divergence
from .score_llm import score_call, write_score
from .transcribe import iter_calls, load_existing_transcript
from .transcript_prep import prepare_transcript

log = logging.getLogger("dry_run")

# "Fit for LOI scoring" thresholds — drop short / procedural calls.
MIN_TRANSCRIPT_CHARS_FIT = 800     # ~200 tokens; below this, scoring is noise
MIN_DURATION_SEC_FIT = 180         # 3 min, only enforced when duration > 0


def _row_meta(call_id: str, source: str, _cache=[]) -> tuple[str, str, str]:
    """Return (call_type, call_date, start_time) from the raw CSV."""
    if not _cache:
        m = {}
        with CALLS_CSV.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                m[(r["source"], r["call_id"])] = (
                    r.get("call_type") or "",
                    r.get("call_date") or "",
                    r.get("start_time") or "",
                )
        _cache.append(m)
    return _cache[0].get((source, call_id), ("", "", ""))


def _pick(n: int, source: str | None, seed: int = 42) -> list:
    """Random sample of any call with a loadable transcript."""
    candidates = []
    for row in iter_calls():
        if source and row.source != source:
            continue
        if load_existing_transcript(row) is not None:
            candidates.append(row)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def _pick_recent_fit(n: int, source: str | None) -> list:
    """Most-recent N calls that are 'fit' for LOI scoring.

    Fitness filters:
    - call_type == 'sales' (excludes booking-confirmation calls, admin, etc.)
    - transcript chars >= MIN_TRANSCRIPT_CHARS_FIT (real conversation, not "hi, callback")
    - duration_sec >= MIN_DURATION_SEC_FIT when populated (Zoom often has 0)
    Sort: call_date desc, then transcript chars desc as tiebreaker.
    """
    candidates: list[tuple[str, str, object]] = []
    for row in iter_calls():
        if source and row.source != source:
            continue
        ct, cdate, stime = _row_meta(row.call_id, row.source)
        if ct != "sales":
            continue
        if row.duration_sec and row.duration_sec < MIN_DURATION_SEC_FIT:
            continue
        loaded = load_existing_transcript(row)
        if loaded is None:
            continue
        text, _ = loaded
        if len(text) < MIN_TRANSCRIPT_CHARS_FIT:
            continue
        # Sort key: ISO-8601 start_time (descending). Fallback to call_date.
        candidates.append((stime or cdate, cdate, row))
    candidates.sort(key=lambda t: t[0], reverse=True)
    log.info("fit pool: %d candidates (filtered to sales + substantive)",
             len(candidates))
    return [c[2] for c in candidates[:n]]


def run(n: int = 50, source: str | None = None, seed: int = 42,
        model: str = SCORING_MODEL, recent: bool = False,
        backend: str = DEFAULT_BACKEND) -> Path:
    ensure_dirs()
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    logfile = OUT_LOGS / f"dry_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile, encoding="utf-8"),
                  logging.StreamHandler()],
        force=True,
    )

    if recent:
        picks = _pick_recent_fit(n, source)
        mode = f"most-recent fit (sales, >={MIN_TRANSCRIPT_CHARS_FIT}c)"
    else:
        picks = _pick(n, source, seed=seed)
        mode = f"random seed={seed}"
    if not picks:
        raise SystemExit("No scoreable calls found.")
    log.info("dry-run [%s, backend=%s]: scoring %d calls (source=%s) with %s",
             mode, backend, len(picks), source or "any", model)

    # Only instantiate the Anthropic client when needed; Ollama needs nothing.
    client = None
    if backend == BACKEND_ANTHROPIC:
        import anthropic  # type: ignore
        from .config import env
        client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY",
                                                 required=True))

    outcomes: list[str] = []
    by_source: dict[str, list[str]] = {"genesys": [], "zoom": []}
    total_cost = 0.0
    errors: list[str] = []

    for i, row in enumerate(picks, 1):
        loaded = load_existing_transcript(row)
        if loaded is None:
            errors.append(f"{row.source}/{row.call_id}: no transcript")
            continue
        transcript, segments = loaded
        # Match Qwen-LoRA training preprocessing for the ollama backend (see
        # score_all.py); Sonnet keeps the full transcript.
        if backend == BACKEND_OLLAMA:
            transcript = prepare_transcript(segments)
        try:
            res = score_call(transcript, row.call_id, row.source,
                             model=model, backend=backend, client=client)
        except Exception as e:  # noqa: BLE001
            log.warning("score failed for %s/%s: %s", row.source, row.call_id, e)
            errors.append(f"{row.source}/{row.call_id}: {e}")
            continue
        write_score(res)
        outcomes.append(res.parsed["outcome"])
        by_source[row.source].append(res.parsed["outcome"])
        total_cost += res.cost_usd
        if i % 10 == 0:
            log.info("progress %d/%d cost=$%.4f", i, len(picks), total_cost)

    emp = empirical_dist(outcomes)
    kl = kl_divergence(emp, EXPECTED_DIST)
    excluded = Counter(o for o in outcomes
                       if o in ("Non interessato", "Non idoneo"))

    report = OUT_REPORTS / "02_dry_run.md"
    report.write_text(_render(outcomes, by_source, emp, kl, excluded,
                              total_cost, errors,
                              n=len(picks), model=model, source=source),
                      encoding="utf-8")
    log.info("dry-run done — report: %s", report)
    return report


def _render(outcomes, by_source, emp, kl, excluded, total_cost, errors,
            *, n, model, source) -> str:
    L = [f"# 02 — Phase 0 Dry-Run\n",
         f"_Model: `{model}` · source: `{source or 'mixed'}` · "
         f"N attempted: {n} · scored: {len(outcomes)} · "
         f"errors: {len(errors)} · cost: ${total_cost:.4f}_\n",
         "## Distribution (over Freddo/Tiepido/Calduccio/Caldo)\n",
         "| Band | Observed | Authored target |",
         "|---|---:|---:|"]
    for k in ("Freddo", "Tiepido", "Calduccio", "Caldo"):
        L.append(f"| {k} | {emp.get(k, 0):.0%} | {EXPECTED_DIST[k]:.0%} |")
    L.append(f"\n**KL(model‖authored) = {kl:.3f}** "
             f"(0 = perfect fit; >0.3 = meaningfully off)\n")
    if any(by_source.values()):
        L.append("## By source\n")
        L.append("| Source | N | Freddo | Tiepido | Calduccio | Caldo | Non int. | Non idoneo |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for src, outs in by_source.items():
            if not outs:
                continue
            ctr = Counter(outs)
            L.append(f"| {src} | {len(outs)} | "
                     f"{ctr.get('Freddo',0)} | {ctr.get('Tiepido',0)} | "
                     f"{ctr.get('Calduccio',0)} | {ctr.get('Caldo',0)} | "
                     f"{ctr.get('Non interessato',0)} | {ctr.get('Non idoneo',0)} |")
        L.append("")
    if excluded:
        L.append(f"## Excluded outcomes\n\nNon idoneo: {excluded.get('Non idoneo', 0)} · "
                 f"Non interessato: {excluded.get('Non interessato', 0)}\n")
    L.append("## Next step\n")
    if kl > 0.3:
        L.append("- Distribution is **meaningfully off**. Iterate on `src/rubric.py` "
                 "before running `src.score_all`.")
    else:
        L.append("- Distribution is within tolerance. Safe to proceed: "
                 "`python -m src.score_all`.")
    if errors:
        L.append(f"\n## Errors ({len(errors)})\n")
        for e in errors[:20]:
            L.append(f"- {e}")
        if len(errors) > 20:
            L.append(f"- ... and {len(errors) - 20} more (see logs).")
    return "\n".join(L)


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--source", choices=("genesys", "zoom"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=(BACKEND_ANTHROPIC, BACKEND_OLLAMA),
                    default=DEFAULT_BACKEND,
                    help=f"Scoring backend (default: {DEFAULT_BACKEND}).")
    ap.add_argument("--model",
                    help="Override the model. Defaults per backend.")
    ap.add_argument("--recent", action="store_true",
                    help="Most-recent fit calls (sales, substantive) instead of random.")
    a = ap.parse_args()
    default_model = SCORING_MODEL if a.backend == BACKEND_ANTHROPIC else OLLAMA_MODEL
    run(n=a.n, source=a.source, seed=a.seed,
        model=a.model or default_model, recent=a.recent, backend=a.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
