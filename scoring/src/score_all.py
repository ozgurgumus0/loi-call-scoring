"""Phase 4 batch scoring — score every call with a Phase-1 transcript.

Idempotent: skips calls whose score file already exists. Safe to re-run; safe
to interrupt with Ctrl+C (current call finishes, then exits).

    python -m src.score_all                      # full corpus, Sonnet
    python -m src.score_all --source zoom        # Zoom only
    python -m src.score_all --limit 100          # smoke test
    python -m src.score_all --max-cost 5.0       # stop at $5
    python -m src.score_all --ceiling --limit 30 # Opus on a subset
"""
from __future__ import annotations

import argparse
import logging
import signal
import time
from collections import Counter

import csv

from .config import (
    BACKEND_ANTHROPIC, BACKEND_OLLAMA, CALLS_CSV, CEILING_MODEL, DATA,
    DEFAULT_BACKEND, OLLAMA_MODEL, OUT_LOGS, OUT_SCORES, SCORING_MODEL,
    ensure_dirs, env,
)
from .score_llm import safe_filename, score_call, score_output_path, write_score
from .transcribe import iter_calls, load_existing_transcript
from .transcript_prep import prepare_transcript


def _load_date_map() -> dict[tuple[str, str], str]:
    """{(source, call_id) -> call_date 'YYYY-MM-DD'} from calls.csv."""
    out: dict[tuple[str, str], str] = {}
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["source"], r["call_id"])] = r.get("call_date", "")
    return out


def _load_xlsx_set() -> set[tuple[str, str]]:
    """Set of (source, call_id) that have a CRM (xlsx) match.

    Reads from calls_enriched.csv (produced by src.join_crm). A row counts
    as in-xlsx if its crm_netsc1 cell is non-empty.
    """
    path = DATA / "calls_enriched.csv"
    out: set[tuple[str, str]] = set()
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("crm_netsc1") or "").strip():
                out.add((r["source"], r["call_id"]))
    return out

log = logging.getLogger("score_all")

_INTERRUPTED = False


def _handle_sigint(_sig, _frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    log.warning("SIGINT received — will exit after current call.")


def _stored_model(model: str, backend: str) -> str:
    """The model identifier as stored in ScoreResult.model (mirrors score_llm)."""
    return f"ollama:{model}" if backend == BACKEND_OLLAMA else model


def _already_scored(source: str, call_id: str, model: str, backend: str) -> bool:
    return score_output_path(source, call_id,
                             _stored_model(model, backend)).exists()


def run(source: str | None, limit: int | None, model: str,
        max_cost: float | None, shuffle_seed: int | None,
        backend: str = DEFAULT_BACKEND,
        since: str | None = None, until: str | None = None,
        in_xlsx: bool = False, sort_recent_first: bool = False) -> dict:
    ensure_dirs()
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    logfile = OUT_LOGS / f"score_all_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile, encoding="utf-8"),
                  logging.StreamHandler()],
        force=True,
    )
    signal.signal(signal.SIGINT, _handle_sigint)

    # Build the worklist first so we can report up-front size and stay
    # deterministic across resumes.
    date_map = _load_date_map() if (since or until or sort_recent_first) else {}
    xlsx_set = _load_xlsx_set() if in_xlsx else set()

    work = []
    skipped_no_transcript = 0
    skipped_filter = 0
    for row in iter_calls():
        if source and row.source != source:
            continue
        if since or until:
            d = date_map.get((row.source, row.call_id), "")
            if since and (not d or d < since):
                skipped_filter += 1
                continue
            if until and (not d or d > until):
                skipped_filter += 1
                continue
        if in_xlsx and (row.source, row.call_id) not in xlsx_set:
            skipped_filter += 1
            continue
        if load_existing_transcript(row) is None:
            skipped_no_transcript += 1
            continue
        if _already_scored(row.source, row.call_id, model, backend):
            continue
        work.append(row)

    if sort_recent_first:
        work.sort(key=lambda r: date_map.get((r.source, r.call_id), ""),
                  reverse=True)
    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(work)
    if limit is not None:
        work = work[:limit]

    log.info("worklist: %d calls (skipped %d without Phase-1 transcript; "
             "already-scored not in list)", len(work), skipped_no_transcript)
    if not work:
        log.info("nothing to do.")
        return {"scored": 0}

    client = None
    if backend == BACKEND_ANTHROPIC:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY",
                                                 required=True))

    scored = 0
    cost = 0.0
    errors = 0
    outcome_ctr: Counter = Counter()
    t0 = time.monotonic()

    for i, row in enumerate(work, 1):
        if _INTERRUPTED:
            log.info("stopping at user request after %d calls", scored)
            break
        if max_cost is not None and cost >= max_cost:
            log.info("hit max-cost cap ($%.2f) after %d calls", max_cost, scored)
            break

        loaded = load_existing_transcript(row)
        if loaded is None:
            continue  # race / disappearance — skip
        transcript, segments = loaded
        # Match the Qwen-LoRA training preprocessing (collapse turns + head/tail
        # truncate to fit 4096) so the deployed model sees the same shape it was
        # trained on. Sonnet (200k ctx) gets the full transcript untouched.
        if backend == BACKEND_OLLAMA:
            transcript = prepare_transcript(segments)
        try:
            res = score_call(transcript, row.call_id, row.source,
                             model=model, backend=backend, client=client)
        except Exception as e:  # noqa: BLE001
            log.warning("%s/%s failed: %s", row.source, row.call_id, e)
            errors += 1
            continue
        # write_score handles the suffixed path automatically based on
        # res.model (which already encodes ollama: / canonical / opus).
        write_score(res)
        scored += 1
        cost += res.cost_usd
        outcome_ctr[res.parsed["outcome"]] += 1
        if i % 25 == 0 or i == len(work):
            rate = scored / max(time.monotonic() - t0, 1e-9)
            log.info("progress %d/%d  cost=$%.4f  rate=%.2f calls/s  "
                     "outcomes=%s", i, len(work), cost, rate, dict(outcome_ctr))

    wall = time.monotonic() - t0
    log.info("done — scored=%d errors=%d cost=$%.4f wall=%.1fs "
             "(outcomes=%s)", scored, errors, cost, wall, dict(outcome_ctr))
    return {"scored": scored, "errors": errors, "cost_usd": cost,
            "outcomes": dict(outcome_ctr)}


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("genesys", "zoom"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-cost", type=float,
                    help="Stop when accumulated Anthropic cost reaches USD this much.")
    ap.add_argument("--backend", choices=(BACKEND_ANTHROPIC, BACKEND_OLLAMA),
                    default=DEFAULT_BACKEND,
                    help=f"Scoring backend (default: {DEFAULT_BACKEND}).")
    ap.add_argument("--ceiling", action="store_true",
                    help="Use Opus 4.7 (anthropic backend only).")
    ap.add_argument("--model",
                    help="Override the model; defaults are Sonnet for "
                         "anthropic, qwen2.5:7b for ollama.")
    ap.add_argument("--shuffle-seed", type=int,
                    help="Shuffle worklist with this seed (for unbiased --limit subsets).")
    ap.add_argument("--since", help="Only score calls with call_date >= YYYY-MM-DD.")
    ap.add_argument("--until", help="Only score calls with call_date <= YYYY-MM-DD.")
    ap.add_argument("--in-xlsx", action="store_true",
                    help="Only score calls that have a CRM match in calls_enriched.csv.")
    ap.add_argument("--recent-first", action="store_true",
                    help="Process calls newest-first (sorts by call_date desc).")
    a = ap.parse_args()
    if a.backend == BACKEND_ANTHROPIC:
        model = a.model or (CEILING_MODEL if a.ceiling else SCORING_MODEL)
    else:
        model = a.model or OLLAMA_MODEL
    run(source=a.source, limit=a.limit, model=model,
        max_cost=a.max_cost, shuffle_seed=a.shuffle_seed, backend=a.backend,
        since=a.since, until=a.until, in_xlsx=a.in_xlsx,
        sort_recent_first=a.recent_first)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
