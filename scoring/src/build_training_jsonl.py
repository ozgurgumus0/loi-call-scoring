"""Build training data for Qwen LoRA fine-tuning from existing Sonnet scores.

Reads:
    outputs/scores/zoom__*.json  (Sonnet outputs — the teacher)
    data/transcripts/zoom/*.vtt  (input transcripts)

Writes:
    data/train/train.jsonl   (~330 examples)
    data/train/val.jsonl     (~35  examples)
    data/train/test.jsonl    (~65  examples — includes the 15 calls we have
                              existing Qwen scores for, so the eval is
                              directly comparable to the 2026-05-28 baseline)

JSONL schema (HuggingFace-standard "messages" format, Unsloth-compatible):
    {"messages": [
        {"role": "system", "content": "<RUBRIC_PROMPT>"},
        {"role": "user",   "content": "<transcript wrapped + instruction>"},
        {"role": "assistant", "content": "<JSON score>"}
    ]}

Split is stratified by sonnet_outcome so minority bands (Caldo, Freddo) are
proportionally represented in train/val/test.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import csv

from .config import CALLS_CSV, DATA, OUT_SCORES
from .rubric import QWEN_RUBRIC_PROMPT, build_messages
from .transcribe import CallRow, load_existing_transcript
from .transcript_prep import prepare_transcript


# v3 recipe — reason-first target key order (CoT scaffold): emit the decision
# tree + the (boolean) behavior-modifier decision BEFORE the label so outcome is
# conditioned on the generated reasoning (the structural fix for the v2
# warm-tilt). Keep the verbose free-text behavior_modifier_reason AFTER the
# label (variable-length; would otherwise push outcome deep into generation) and
# the long extractive evidence_quotes LAST. rubric.QWEN_RUBRIC_PROMPT's schema
# line is reordered to match (train/serve consistency). Parsing is
# order-independent (parse_score reads keys by name), so this is safe.
_ASSISTANT_KEY_ORDER = (
    "tree_path",
    "behavior_modifier_applied",
    "outcome",
    "score",
    "behavior_modifier_reason",
    "time_signal",
    "confidence",
    "evidence_quotes",
)


def _reorder_assistant(score: dict) -> dict:
    """Re-serialise a Sonnet score dict reason-first (see _ASSISTANT_KEY_ORDER).

    Missing keys are skipped (older records won't KeyError); any unexpected
    extra keys are appended unobtrusively at the end.
    """
    out = {k: score[k] for k in _ASSISTANT_KEY_ORDER if k in score}
    for k, v in score.items():
        if k not in out:
            out[k] = v
    return out


# v3 recipe — TRAIN-ONLY duplication oversampling to counter the measured
# warm-tilt (student warmer than Sonnet: gold=Tiepido, pred=Calduccio). Cap is
# 2.0x (at most one extra copy per row) to bound overfitting on duplicated
# transcripts. Calduccio is the anchor (1.0). VAL/TEST are NEVER reweighted, so
# eval stays at the true Sonnet distribution. Re-tune ONLY this dict after the
# corpus grows; treat Tiepido 1.5 as the ceiling (2.0 over-corrects to cold).
TRAIN_OVERSAMPLE = {
    "Tiepido": 1.5, "Freddo": 2.0, "Caldo": 2.0,
    "Calduccio": 1.0, "Non interessato": 1.0, "Non idoneo": 1.0,
}


def _reweight(rows: list[dict], seed: int) -> list[dict]:
    """TRAIN-ONLY: duplicate rows per-outcome per TRAIN_OVERSAMPLE.

    Integer part = whole extra copies; fractional remainder = a seeded random
    subset of the band. Multiplier clamped to <=2.0 (one extra copy max). Uses
    random.Random(seed + 1) so it does NOT disturb the split RNG. NEVER call on
    val/test.
    """
    rng = random.Random(seed + 1)
    by_outcome: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_outcome[r["outcome"]].append(r)
    out: list[dict] = list(rows)            # every original kept exactly once
    for outcome, bucket in by_outcome.items():
        mult = min(2.0, TRAIN_OVERSAMPLE.get(outcome, 1.0))
        if mult <= 1.0:
            continue
        whole = int(mult) - 1               # extra whole copies (mult<=2 -> 0)
        frac = mult - int(mult)
        for _ in range(whole):
            out.extend(bucket)
        k = int(round(len(bucket) * frac))
        if k:
            pool = list(bucket)
            rng.shuffle(pool)
            out.extend(pool[:k])
    rng.shuffle(out)
    return out


def _load_transcript_map() -> dict[str, str]:
    """{zoom call_id -> transcript_file relative path} from calls.csv."""
    out: dict[str, str] = {}
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source") == "zoom":
                tf = (r.get("transcript_file") or "").strip()
                if tf:
                    out[r["call_id"]] = tf
    return out


_TR_MAP: dict[str, str] = {}

TRAIN_DIR = DATA / "train"
TRAIN_JSONL = TRAIN_DIR / "train.jsonl"
VAL_JSONL = TRAIN_DIR / "val.jsonl"
TEST_JSONL = TRAIN_DIR / "test.jsonl"
TEST_IDS_JSON = TRAIN_DIR / "test_ids.json"                      # frozen test membership (persisted, reused across builds)
TEST_IDS_V2_BASELINE = TRAIN_DIR / "test_ids_v2_baseline.json"   # v2's 65 calls — bootstrap seed for the freeze

# The 15 call_ids that have existing Qwen scores from the 2026-05-28 baseline.
# These MUST land in the test set for the LoRA-vs-baseline comparison to be
# valid. Read directly from the Qwen score files at runtime.
def _qwen_baseline_ids() -> set[str]:
    out = set()
    for p in OUT_SCORES.glob("zoom__*__ollama_qwen*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("call_id"):
                out.add(d["call_id"])
        except Exception:
            continue
    return out


def _signed_call_ids() -> set[str]:
    """Zoom call_ids with a real signed-contract outcome (conversion_outcome ==
    'signed'). Used ONLY to route contract-bearing calls into the frozen test
    set so the contract-discrimination eval has positives — NEVER fed into any
    training message (the assistant target is the Sonnet score only).
    """
    out: set[str] = set()
    if not CALLS_CSV.exists():
        return out
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("source") == "zoom"
                    and (r.get("conversion_outcome") or "").strip() == "signed"):
                out.add(r["call_id"])
    return out


def _load_frozen_test_ids() -> set[str]:
    """Frozen test membership for cross-version comparability. Prefer the
    persisted test_ids.json; on the first build fall back to the reconstructed
    v2 baseline. Empty if neither file exists.
    """
    for path in (TEST_IDS_JSON, TEST_IDS_V2_BASELINE):
        if path.exists():
            try:
                return set(json.loads(path.read_text(encoding="utf-8")).get("call_ids", []))
            except Exception:
                continue
    return set()


def _load_sonnet_scores() -> list[dict]:
    """Return list of {call_id, outcome, score} for every Sonnet score.

    Deduplicated by call_id (first file wins in sorted order) so a call with
    more than one Sonnet score file can't land twice in the split.
    """
    rows = []
    seen: set[str] = set()
    for p in sorted(OUT_SCORES.glob("zoom__*.json")):
        name = p.name
        if "__ollama_" in name or name.startswith("upload__") or name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (d.get("model") or "").lower().startswith("claude-sonnet"):
            continue
        sc = d.get("score")
        if not isinstance(sc, dict) or not sc.get("outcome"):
            continue
        cid = d["call_id"]
        if cid in seen:
            continue
        seen.add(cid)
        rows.append({"call_id": cid, "outcome": sc["outcome"], "score": sc})
    return rows


def _build_example(call_id: str, sonnet_score: dict) -> dict | None:
    """Load transcript for call_id and assemble the (sys, user, assistant) record."""
    tf = _TR_MAP.get(call_id)
    if not tf:
        return None
    row = CallRow(
        source="zoom", call_id=call_id, audio_file=None,
        transcript_file=tf, duration_sec=None,
    )
    loaded = load_existing_transcript(row)
    if loaded is None:
        return None
    transcript, segments = loaded
    # v1: collapse same-speaker turns + head/tail truncate so the rubric, the
    # decision-relevant opening/close, AND the assistant label all fit in the
    # 4096-token window. (v0 truncated blindly and lost the label / the close.)
    transcript = prepare_transcript(segments)
    user_msg = build_messages(transcript)[0]["content"]
    assistant_msg = json.dumps(_reorder_assistant(sonnet_score), ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": QWEN_RUBRIC_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
    }


def _stratified_split(rows: list[dict], must_be_test: set[str],
                      target_test: int, target_val: int,
                      seed: int) -> tuple[list, list, list]:
    """Split rows into (train, val, test), stratified by outcome.

    must_be_test call_ids are PINNED into test first and never dropped or
    re-rolled across builds (the frozen-test guarantee). The remainder of
    target_test is filled by stratified sampling and trimmed to the EXACT target
    (fixes the old round()-quota drift, e.g. 65 -> 66). If the pinned set already
    exceeds target_test, all pinned rows are kept. Val is sampled stratified from
    what's left; train is everything else.
    """
    rng = random.Random(seed)

    def _fill(pool: list[dict], target: int) -> list[dict]:
        """Stratified pick of EXACTLY `target` rows from pool (or all if fewer)."""
        if target <= 0 or not pool:
            return []
        if len(pool) <= target:
            return list(pool)
        by_outcome: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_outcome[r["outcome"]].append(r)
        for v in by_outcome.values():
            rng.shuffle(v)
        total = len(pool)
        picked: list[dict] = []
        for bucket in by_outcome.values():
            quota = round(target * len(bucket) / total)
            picked.extend(bucket[:quota])
        rng.shuffle(picked)                       # de-bias the trim across bands
        if len(picked) > target:
            return picked[:target]
        if len(picked) < target:                  # rounding undershoot -> backfill
            have = {r["call_id"] for r in picked}
            extra = [r for r in pool if r["call_id"] not in have]
            rng.shuffle(extra)
            picked.extend(extra[:target - len(picked)])
        return picked

    # Phase 1: pin must_be_test (only those actually present + labeled in rows)
    test_ids = {r["call_id"] for r in rows if r["call_id"] in must_be_test}
    # Phase 2: fill test up to the exact target from the non-pinned pool
    pool = [r for r in rows if r["call_id"] not in test_ids]
    test_ids |= {r["call_id"] for r in _fill(pool, target_test - len(test_ids))}
    # Phase 3: val stratified from what's left
    rest = [r for r in rows if r["call_id"] not in test_ids]
    val_ids = {r["call_id"] for r in _fill(rest, target_val)}
    # Phase 4: rest is train
    test = [r for r in rows if r["call_id"] in test_ids]
    val = [r for r in rows if r["call_id"] in val_ids]
    train = [r for r in rows
             if r["call_id"] not in test_ids and r["call_id"] not in val_ids]
    return train, val, test


def _dump(rows: list[dict], path: Path) -> int:
    examples = []
    skipped = 0
    for r in rows:
        ex = _build_example(r["call_id"], r["score"])
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(examples)


def _dist(rows: list[dict]) -> str:
    c = Counter(r["outcome"] for r in rows)
    return ", ".join(f"{o}={c[o]}" for o in sorted(c, key=lambda x: -c[x]))


def main(test_size: int, val_size: int, seed: int, reweight: bool = True) -> None:
    global _TR_MAP
    _TR_MAP = _load_transcript_map()
    print(f"Transcript paths indexed for {len(_TR_MAP)} Zoom call_ids")
    sonnet = _load_sonnet_scores()
    print(f"Sonnet scores found: {len(sonnet)}")
    print(f"  distribution: {_dist(sonnet)}")

    qwen_ids = _qwen_baseline_ids()
    signed_ids = _signed_call_ids()
    frozen_ids = _load_frozen_test_ids()
    must_be_test = frozen_ids | qwen_ids | signed_ids
    labeled = {r["call_id"] for r in sonnet}
    print(f"Forced into test: frozen={len(frozen_ids)} qwen-anchor={len(qwen_ids)} "
          f"signed-contract={len(signed_ids)}  ->  labeled-in-union="
          f"{len(must_be_test & labeled)} (of {len(must_be_test)} requested)")

    train_rows, val_rows, test_rows = _stratified_split(
        sonnet, must_be_test, test_size, val_size, seed,
    )
    print(f"Stratified split: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"  train dist: {_dist(train_rows)}")
    print(f"  val   dist: {_dist(val_rows)}")
    print(f"  test  dist: {_dist(test_rows)}")

    if reweight:
        train_out = _reweight(train_rows, seed)
        n_train = _dump(train_out, TRAIN_JSONL)
        print(f"  train dist (after reweight, PRE-skip): {_dist(train_out)}  "
              f"[rows={len(train_out)}, built->file={n_train}, was {len(train_rows)}]")
    else:
        n_train = _dump(train_rows, TRAIN_JSONL)
        print(f"  train dist (x1.0 — NO reweight): {_dist(train_rows)}  "
              f"[rows={len(train_rows)}, built->file={n_train}]")
    n_val = _dump(val_rows, VAL_JSONL)
    n_test = _dump(test_rows, TEST_JSONL)
    test_ids_final = sorted(r["call_id"] for r in test_rows)
    TEST_IDS_JSON.write_text(
        json.dumps({"note": "Frozen test membership for cross-version LOI eval "
                            "(pinned: v2 baseline + qwen anchors + signed-contract "
                            "calls; filled stratified to --test-size; reused, never "
                            "re-rolled, on later builds).",
                    "source": "zoom", "call_ids": test_ids_final},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {TEST_IDS_JSON}  ({len(test_ids_final)} frozen test call_ids)")
    print()
    print(f"[saved] {TRAIN_JSONL}  ({n_train} examples)")
    print(f"[saved] {VAL_JSONL}    ({n_val} examples)")
    print(f"[saved] {TEST_JSONL}   ({n_test} examples)")


def _cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=int, default=200,
                    help="Target size of the frozen test set (default 200; "
                         "pinned ids — v2 baseline + qwen anchors + signed — are "
                         "always kept even if they exceed this).")
    ap.add_argument("--val-size", type=int, default=80,
                    help="Target size of validation set (default 80).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-reweight", action="store_true",
                    help="disable train oversampling (x1.0 true distribution); use "
                         "for the 14B run after the v3 x1.5 over-correction.")
    a = ap.parse_args()
    main(test_size=a.test_size, val_size=a.val_size, seed=a.seed,
         reweight=not a.no_reweight)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
