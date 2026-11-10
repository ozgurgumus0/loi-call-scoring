"""Honest, production-realistic eval of the deployed Qwen-LOI model via Ollama.

The Kaggle eval was a LOWER BOUND: single-shot generation, strict parse, NO
retry. So every time the model fumbled the JSON it scored as wrong — even though
the real pipeline retries and recovers those. This script runs the held-out test
set through the SAME path production uses (`src.score_llm._ollama_chat`):
`format: json`, num_ctx 4096, and one retry on a bad parse. The result is the
number that actually matters — how often the deployed model agrees with Sonnet
when used for real.

Gold labels are the Sonnet outcomes baked into data/train/test.jsonl (the same
66 the Kaggle cell scored). The system + user messages are sent verbatim, so the
prompt + prepped transcript exactly match training.

    ollama serve                                   # in another terminal
    python -m src.eval_qwen_ollama --model qwen-loi          # full 66 (slow on CPU)
    python -m src.eval_qwen_ollama --model qwen-loi --limit 10   # quick sanity
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .config import DATA
from .score_llm import ParseError, _ollama_chat, parse_score

TEST = DATA / "train" / "test.jsonl"

# LOI scores for the 4 in-band outcomes; ORD = ordinal rank for a clean ±1.
BAND = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}
ORD = {"Freddo": 0, "Tiepido": 1, "Calduccio": 2, "Caldo": 3}

_NUDGE = ("\n\n# RICORDA\nDevi restituire SOLO il JSON, senza testo aggiuntivo. "
          "Tutti i campi sono obbligatori.")


def _predict(model: str, system: str, user: str) -> tuple[dict | None, str]:
    """One scoring call with one retry — mirrors score_llm._score_ollama.

    Returns (parsed_or_None, status) where status is "ok" (first try),
    "recovered" (failed once, retry parsed), or "fail" (unparseable twice).
    """
    nudge = ""
    for attempt in (1, 2):
        resp = _ollama_chat(model, system + nudge, user)
        raw = (resp.get("message") or {}).get("content", "")
        try:
            parsed = parse_score(raw)
            return parsed, ("ok" if attempt == 1 else "recovered")
        except ParseError:
            if attempt == 2:
                return None, "fail"
            nudge = _NUDGE
    return None, "fail"


def main(model: str, limit: int | None) -> None:
    rows = [json.loads(ln) for ln in TEST.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    if limit:
        rows = rows[:limit]

    n = exact = legacy = adj = recovered = failed = 0
    conf: Counter = Counter()
    t0 = time.monotonic()

    for i, ex in enumerate(rows, 1):
        msgs = ex["messages"]
        system, user = msgs[0]["content"], msgs[1]["content"]
        gold = json.loads(msgs[-1]["content"]).get("outcome")

        pred_obj, status = _predict(model, system, user)
        pred = (pred_obj or {}).get("outcome")
        if status == "recovered":
            recovered += 1
        if status == "fail":
            failed += 1

        n += 1
        if pred == gold:
            exact += 1
            legacy += 1
            adj += 1
        else:
            if pred in BAND and gold in BAND:
                if abs(BAND[pred] - BAND[gold]) <= 2:
                    legacy += 1
                if abs(ORD[pred] - ORD[gold]) <= 1:
                    adj += 1
            conf[(gold, pred)] += 1

        flag = "" if status == "ok" else f"  [{status}]"
        print(f"  [{i:>2}/{len(rows)}] gold={gold!s:>14}  pred={pred!s:>14}{flag}")

    wall = time.monotonic() - t0
    print(f"\n=== Qwen-LOI via Ollama (production path: retry + format:json + num_ctx 4096) ===")
    print(f"model: {model}    n = {n}    wall = {wall/60:.1f} min "
          f"({wall/max(n,1):.0f} s/call)")
    print(f"Exact outcome:    {exact}/{n} = {exact/n:.1%}   (Kaggle single-shot = 37.9%)")
    print(f"Band, legacy <=2: {legacy}/{n} = {legacy/n:.1%}   (Kaggle = 68.2%)")
    print(f"Band, true +-1:   {adj}/{n} = {adj/n:.1%}")
    print(f"Parse recovered by retry: {recovered}/{n}    still unparseable: {failed}/{n}")
    print("\nTop confusions (gold -> pred):")
    for (g, p), c in conf.most_common(15):
        print(f"  {c:2d}x  {g} -> {p}")


def _cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-loi",
                    help="Ollama model name (default: qwen-loi).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N test calls (quick sanity check).")
    a = ap.parse_args()
    main(a.model, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
