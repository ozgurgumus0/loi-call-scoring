"""Phase 4 — score one transcript with an LLM backend + parse + write.

Two backends are supported:

- `anthropic` (default): claude-sonnet-4-6 (~$0.02/call) or claude-opus-4-7
  (~$0.10/call). Rubric is sent as a cache-control system block.
- `ollama` (local): supports a prompted model or a locally configured adapter.
  Requires Ollama running at OLLAMA_BASE_URL with the model pulled.

The same parse_score + write_score pipeline serves both. Cost is reported
as 0 for Ollama runs.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import rubric
from .config import (
    BACKEND_ANTHROPIC, BACKEND_OLLAMA, CEILING_MODEL, DEFAULT_BACKEND,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OUT_SCORES, OUTCOME_SCORES, RUBRIC_VERSION,
    SCORING_MODEL, ensure_dirs, env,
)


# ---------------------------------------------------------------- parsing ---
class ParseError(ValueError):
    pass


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Only fields that drive scoring + the tree are mandatory. Smaller models
# (e.g. qwen2.5:3b) sometimes omit evidence_quotes / behavior_modifier_*; we
# fill them with sane defaults rather than reject the whole response.
_REQUIRED_KEYS = {"outcome", "score", "tree_path", "confidence"}
_OPTIONAL_DEFAULTS = {
    "time_signal": "assente",
    "behavior_modifier_applied": False,
    "behavior_modifier_reason": None,
    "evidence_quotes": [],
}
_VALID_OUTCOMES = set(OUTCOME_SCORES) | {"Non interessato", "Non idoneo"}


def parse_score(raw: str) -> dict:
    """Strip code fences, parse JSON, validate the required shape.

    Raises ParseError on any failure so the caller can retry once.
    Optional fields get filled with defaults if the model omitted them.
    """
    if not raw or not raw.strip():
        raise ParseError("empty response")
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f"not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ParseError(f"top level must be object, got {type(obj).__name__}")
    missing = _REQUIRED_KEYS - set(obj)
    if missing:
        raise ParseError(f"missing required keys: {sorted(missing)}")
    if obj["outcome"] not in _VALID_OUTCOMES:
        raise ParseError(f"invalid outcome: {obj['outcome']!r}")
    for k, default in _OPTIONAL_DEFAULTS.items():
        obj.setdefault(k, default)
    # score consistency: if outcome maps to a canonical score, enforce it.
    expected = OUTCOME_SCORES.get(obj["outcome"])
    if expected is not None and obj["score"] != expected:
        # Don't reject — score the way the model said, but note the mismatch.
        obj["_score_mismatch"] = {"declared": obj["score"], "canonical": expected}
    return obj


def _normalize_closing_readiness(obj: dict) -> None:
    """Coerce `closing_readiness` to an int in [0, 100], in place.

    The prompted production scorer emits this continuous 0-100 field (see
    rubric.CALIBRATION); it is what production RANKS on, so make sure it lands as
    a clean int. Absent/garbage -> drop the key so downstream treats it as
    missing (the fine-tune path never emits it). parse_score already preserves
    any extra key the model returns, so this only tidies the value.
    """
    if "closing_readiness" not in obj:
        return
    val = obj["closing_readiness"]
    try:
        obj["closing_readiness"] = max(0, min(100, int(round(float(val)))))
    except (TypeError, ValueError):
        obj.pop("closing_readiness", None)


# --------------------------------------------------------------- scoring ---
@dataclass
class ScoreResult:
    call_id: str
    source: str
    model: str
    rubric_version: str
    parsed: dict
    raw: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_sec: float


# Anthropic public list-price as of 2026-05 (USD per 1M tokens).
# Override via env if these drift.
_PRICES = {
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "claude-opus-4-7":   {"in": 15.0, "out": 75.0},
}


def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = _PRICES.get(model, {"in": 0.0, "out": 0.0})
    return (in_tok * p["in"] + out_tok * p["out"]) / 1_000_000.0


def score_call(
    transcript: str,
    call_id: str,
    source: str,
    model: str | None = None,
    backend: str = DEFAULT_BACKEND,
    client=None,
) -> ScoreResult:
    """Score one call via the chosen backend. Returns a ScoreResult; writes nothing.

    backend = "anthropic": uses claude-sonnet-4-6 (default) or claude-opus-4-7.
              `client` may be a reusable `anthropic.Anthropic()` instance.
    backend = "ollama":    uses OLLAMA_MODEL (default qwen2.5:7b). `client`
              is ignored. Requires Ollama running at OLLAMA_BASE_URL.
    """
    if backend == BACKEND_ANTHROPIC:
        return _score_anthropic(transcript, call_id, source,
                                model or SCORING_MODEL, client)
    if backend == BACKEND_OLLAMA:
        return _score_ollama(transcript, call_id, source, model or OLLAMA_MODEL)
    raise ValueError(f"unknown backend: {backend!r}")


# ────────────────────────────────────────────────────────── Anthropic ─────
def _score_anthropic(transcript: str, call_id: str, source: str,
                     model: str, client) -> ScoreResult:
    if client is None:
        import anthropic  # noqa: WPS433
        client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY", required=True))

    messages = rubric.build_messages(transcript)
    t0 = time.monotonic()
    raw_text = ""
    for attempt in (1, 2):  # one retry
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": rubric.RUBRIC_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        )
        raw_text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        try:
            parsed = parse_score(raw_text)
            break
        except ParseError as e:
            if attempt == 2:
                raise ParseError(f"parse failed twice: {e}; raw={raw_text[:300]}") from e
            messages = messages + [
                {"role": "assistant", "content": raw_text},
                {"role": "user", "content":
                    "Il JSON precedente non è valido o manca di campi richiesti. "
                    "Restituisci SOLO il JSON corretto secondo lo schema."},
            ]
    latency = time.monotonic() - t0

    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) if usage else 0
    out_tok = getattr(usage, "output_tokens", 0) if usage else 0
    cost = _estimate_cost(model, in_tok, out_tok)

    return ScoreResult(
        call_id=call_id, source=source, model=model,
        rubric_version=RUBRIC_VERSION, parsed=parsed, raw=raw_text,
        input_tokens=in_tok, output_tokens=out_tok,
        cost_usd=cost, latency_sec=latency,
    )


# ───────────────────────────────────────────────────────────── Ollama ─────
class OllamaUnreachable(RuntimeError):
    """Ollama daemon isn't responding at OLLAMA_BASE_URL."""


def _is_finetuned_loi(model: str) -> bool:
    """True if `model` is our fine-tuned LOI adapter served via Ollama.

    The fine-tune was trained on the CONDENSED `QWEN_RUBRIC_PROMPT` (~487 tok),
    so it MUST be served with that same system prompt — otherwise the model sees
    a 2.7x longer prompt at inference than it ever saw in training (train/serve
    skew). A *base* model (e.g. qwen2.5:7b) has not internalised the decision
    tree and still needs the FULL `RUBRIC_PROMPT`, so we switch only when the
    Ollama model name marks it as the fine-tune (name it e.g. `qwen-loi`).
    """
    return "loi" in model.lower()


def _ollama_system_prompt(model: str) -> str:
    """Pick the rubric variant that matches how `model` was trained/served.

    - Fine-tuned `qwen-loi` adapter -> condensed QWEN_RUBRIC_PROMPT (train/serve
      match; it learned the tree, only needs the schema reminder).
    - Anything else (the PROMPTED production model, e.g. `qwen3:8b`, or a base
      model) -> PROMPTED_RUBRIC_PROMPT = full rubric + CALIBRATION, byte-identical
      to the Kaggle system prompt that produced the validated verdict. The
      calibration is what makes the model (a) keep a real Caldo band and (b) emit
      the integer `closing_readiness` that production ranks on.
    """
    return (rubric.QWEN_RUBRIC_PROMPT if _is_finetuned_loi(model)
            else rubric.PROMPTED_RUBRIC_PROMPT)


# Context window per path. The fine-tune was trained at 4096 (keep train/serve
# match). The prompted path sends full rubric (~1324) + CALIBRATION (~290) +
# prepped transcript (~2770) + JSON, which overflows 4096 -> serve at 8192 to
# match the Kaggle MAX_SEQ and avoid silently re-truncating the call.
_OLLAMA_NUM_CTX_FINETUNE = 4096
_OLLAMA_NUM_CTX_PROMPTED = 8192


def _ollama_num_ctx(model: str) -> int:
    return _OLLAMA_NUM_CTX_FINETUNE if _is_finetuned_loi(model) else _OLLAMA_NUM_CTX_PROMPTED


# CPU inference on a laptop is bandwidth-bound; a 20 KB Zoom VTT can take
# 5–10 min on an i7-U class chip. 30 min ceiling covers worst case.
_OLLAMA_TIMEOUT_SEC = 1800.0
# Truncate long transcripts for Ollama — every input token has to be
# encoded through all 7B weights, so input length dominates wall time.
# 8000 chars ≈ 2000 tokens. Head-heavy split: Q1 (product type),
# Q2 (blockers, ownership), and early Q3 (info-gathering) signals
# live in the opening discovery; price/next-steps (Q4) live in the
# closing. Middle clip dropped key Q1 evidence on long Zoom calls.
_OLLAMA_MAX_INPUT_CHARS = 8000
_OLLAMA_HEAD_FRACTION = 0.75


def _truncate_for_ollama(text: str, limit: int = _OLLAMA_MAX_INPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head_n = int(limit * _OLLAMA_HEAD_FRACTION)
    tail_n = limit - head_n
    head = text[:head_n]
    tail = text[-tail_n:]
    return f"{head}\n\n[...truncated {len(text) - limit:,} chars...]\n\n{tail}"


def _ollama_chat(model: str, system_text: str, user_text: str,
                 timeout: float = _OLLAMA_TIMEOUT_SEC,
                 num_ctx: int = _OLLAMA_NUM_CTX_FINETUNE) -> dict:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "format": "json",
        "stream": False,
        # num_ctx is REQUIRED: Ollama's default context is 2048, which would
        # silently truncate our prepped inputs at serve time and reintroduce the
        # very truncation bug v1/v2 fixed in training. The prompted production
        # path needs 8192 (full rubric + calibration + transcript); the fine-tune
        # matches its 4096 training window. See _ollama_num_ctx.
        # num_predict 1024 matches the Kaggle MAX_NEW (thinking off -> short JSON).
        "options": {"temperature": 0, "num_predict": 1024, "num_ctx": num_ctx},
    }
    # Qwen 3.x (and other thinking-capable models) emit chain-of-thought into
    # a separate `thinking` field by default, leaving `content` empty until the
    # think budget is exhausted. Disable thinking so we get straight JSON.
    if model.lower().startswith(("qwen3", "deepseek-r1")):
        payload["think"] = False
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaUnreachable(
            f"could not reach Ollama at {OLLAMA_BASE_URL} ({e}). "
            f"Start the daemon (`ollama serve`) and `ollama pull {model}`."
        ) from e


def _score_ollama(transcript: str, call_id: str, source: str,
                  model: str) -> ScoreResult:
    bounded = _truncate_for_ollama(transcript)
    user_msg = (
        "Trascrizione della call (italiano, già diarizzata dove possibile):\n\n"
        f"<transcript>\n{bounded}\n</transcript>\n\n"
        "Applica l'albero decisionale e restituisci il JSON."
    )
    # _ollama_chat passes `think: false` for Qwen 3.x to suppress
    # chain-of-thought; nothing extra needed in the user message.
    # Match the system prompt to how the model was trained: the fine-tuned LOI
    # adapter gets the condensed QWEN_RUBRIC_PROMPT (train/serve match); a base
    # model gets the full rubric. The transcript itself is already preprocessed
    # upstream (score_all / dry_run_scoring call prepare_transcript for the
    # ollama backend), so _truncate_for_ollama below is an inert safety net for
    # the rare unprepared input rather than a second, divergent truncation.
    system_prompt = _ollama_system_prompt(model)
    num_ctx = _ollama_num_ctx(model)
    t0 = time.monotonic()
    raw_text = ""
    parsed: dict | None = None
    nudge = ""
    for attempt in (1, 2):
        resp = _ollama_chat(model, system_prompt + nudge, user_msg, num_ctx=num_ctx)
        raw_text = (resp.get("message") or {}).get("content", "")
        try:
            parsed = parse_score(raw_text)
            _normalize_closing_readiness(parsed)
            break
        except ParseError as e:
            if attempt == 2:
                raise ParseError(
                    f"parse failed twice ({model}): {e}; raw={raw_text[:300]}"
                ) from e
            nudge = ("\n\n# RICORDA\nDevi restituire SOLO il JSON, senza testo "
                     "aggiuntivo. Tutti i campi sono obbligatori.")
    latency = time.monotonic() - t0

    in_tok = int(resp.get("prompt_eval_count") or 0)
    out_tok = int(resp.get("eval_count") or 0)
    return ScoreResult(
        call_id=call_id, source=source,
        model=f"ollama:{model}",
        rubric_version=RUBRIC_VERSION, parsed=parsed or {}, raw=raw_text,
        input_tokens=in_tok, output_tokens=out_tok,
        cost_usd=0.0, latency_sec=latency,
    )


def ollama_reachable(timeout: float = 2.0) -> bool:
    """Quick capability check used by the Flask app's status panel."""
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=timeout
        ) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


_FILENAME_SAFE = str.maketrans({"/": "_", "+": "-", "=": "", "\\": "_",
                                ":": "_", "*": "_", "?": "_", '"': "_",
                                "<": "_", ">": "_", "|": "_"})


def safe_filename(source: str, call_id: str) -> str:
    """Windows-safe per-call filename. Zoom call_ids contain /, +, =."""
    return f"{source}__{call_id.translate(_FILENAME_SAFE)}.json"


def score_output_path(source: str, call_id: str, model: str) -> Path:
    """Where this (source, call_id, model) tuple's score JSON should live.

    Anthropic Sonnet (the canonical default) writes to the base filename.
    Any other model gets a suffix so Claude + Qwen + Opus runs of the same
    call coexist on disk instead of overwriting each other.
    """
    base = safe_filename(source, call_id)
    if model == SCORING_MODEL:                        # claude-sonnet-4-6
        return OUT_SCORES / base
    safe_model = model.replace("/", "_").replace(":", "_")
    return OUT_SCORES / base.replace(".json", f"__{safe_model}.json")


def write_score(result: ScoreResult) -> Path:
    ensure_dirs()
    out_path = score_output_path(result.source, result.call_id, result.model)
    payload = {
        "call_id": result.call_id,
        "source": result.source,
        "model": result.model,
        "rubric_version": result.rubric_version,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": round(result.cost_usd, 6),
        "latency_sec": round(result.latency_sec, 2),
        "score": result.parsed,
        "raw": result.raw,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return out_path


# ----------------------------------------------------------------- CLI -----
def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", type=Path, required=True,
                    help="Path to a .txt or .vtt transcript file.")
    ap.add_argument("--call-id", required=True)
    ap.add_argument("--source", required=True, choices=("genesys", "zoom"))
    ap.add_argument("--backend", choices=(BACKEND_ANTHROPIC, BACKEND_OLLAMA),
                    default=DEFAULT_BACKEND,
                    help=f"Scoring backend (default: {DEFAULT_BACKEND}).")
    ap.add_argument("--ceiling", action="store_true",
                    help="Use the Anthropic Opus model instead of Sonnet "
                         "(anthropic backend only).")
    ap.add_argument("--model",
                    help="Override the model. Defaults: Sonnet for anthropic, "
                         "qwen2.5:7b for ollama.")
    args = ap.parse_args()

    if args.backend == BACKEND_ANTHROPIC:
        model = args.model or (CEILING_MODEL if args.ceiling else SCORING_MODEL)
    else:
        model = args.model or OLLAMA_MODEL

    text = args.transcript.read_text(encoding="utf-8")
    res = score_call(text, args.call_id, args.source,
                     model=model, backend=args.backend)
    out = write_score(res)
    cost = f"${res.cost_usd:.4f}" if res.cost_usd > 0 else "free"
    cr = res.parsed.get("closing_readiness")
    cr_str = f"  cr={cr}" if cr is not None else ""
    print(f"{res.parsed['outcome']:>16}  score={res.parsed.get('score')}{cr_str}  "
          f"backend={args.backend}  cost={cost}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
