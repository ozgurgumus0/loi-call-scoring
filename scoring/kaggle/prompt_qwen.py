# %% Prompt Qwen3-8B for LOI — NO training. Scores test_prompting_sales.jsonl
# (now the FULL 892-call sales corpus; earlier versions held the frozen 200/300).
#
# This is the "prompt a big open model" path: we load Qwen3.5-9B (instruct) in
# 4-bit on a single T4 and prompt it with the FULL rubric per call — exactly how
# Claude is prompted. No adapter, no fine-tuning. A prompted model reasons from
# the rubric instead of learning from 40 Caldo examples, so (unlike the v3
# fine-tune) it CAN output Caldo.
#
# INPUTS (attach as a Kaggle dataset): test_prompting.jsonl
#   (produced locally by `python -m src.emit_test_prompting`)
# OUTPUT: /kaggle/working/predictions.csv  — download it, then locally run
#   python -m src.contract_verdict --model qwen3.5-9b-prompt \
#       --pred predictions.csv --ids data/train/test_ids.json
#
# Notes:
#  * Qwen3.5 THINKS by default — we pass enable_thinking=False so it emits JSON
#    directly. We also strip any </think> preamble defensively before parsing.
#  * Inference (not training) — a 9B at 4-bit fits 16 GB comfortably.

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "unsloth", "unsloth_zoo"], check=True)

import os, json, csv, re
from pathlib import Path
from collections import Counter
from unsloth import FastLanguageModel

# Qwen3.5-9B is MULTIMODAL → loads a processor that wants list-of-parts content
# and falls back to slow float32 on a T4. For a text-only LOI task, Qwen3-8B is
# the robust pick: text-only, has thinking mode, nearly as capable, normal
# tokenizer path. (Swap back to "unsloth/Qwen3.5-9B" only if you want to fight
# the multimodal path — the chat_ids() below tolerates both.)
# PRODUCTION PICK (8B/300 verdict, 2026-06-10): Qwen3-8B ranked by continuous
# closing_readiness = AUC 0.671 — statistically indistinguishable from Claude
# (0.696) on the representative 300. The 14B added nothing (its continuous
# score restates the band) — retired. This run = the FULL sales corpus (892).
# 2026-07-14: prefer the KAGGLE-MOUNTED Qwen3-8B (kernel-metadata model_sources
# -> local sharded safetensors under /kaggle/input, zero HF-hub calls). HF-side
# loading broke twice today: unsloth's pre-quantized mirror repo lost its weight
# files, and sharded-index resolution failed even on the official repo. Falls
# back to the official HF repo only if the mount is absent.
import os as _os
MODEL = "Qwen/Qwen3-8B"
for _root, _dirs, _files in _os.walk("/kaggle/input"):
    if "model.safetensors.index.json" in _files and "config.json" in _files:
        MODEL = _root
        break
THINKING = False
MAX_SEQ = 8192                       # full rubric + transcript + JSON (no trace now)
MAX_NEW = 3072 if THINKING else 1024 # thinking off -> short JSON, 1024 is plenty

# Slice for splitting a long run across sessions (row indices into the input).
START, END = 0, None

CALIBRATION = (
    "Include closing_readiness as an integer from 0 to 100 based only on "
    "explicit statements about next steps."
)

# --- locate test_prompting.jsonl (fallback to test_with_ids / test) ---------
TEST = None
for want in ("test_prompting_sales.jsonl", "test_prompting.jsonl",
             "test_with_ids.jsonl", "test.jsonl"):
    for root, _d, files in os.walk("/kaggle/input"):
        if want in files:
            TEST = str(Path(root, want)); break
    if TEST:
        break
assert TEST, "test_prompting.jsonl not found under /kaggle/input (attach the dataset)"
print(f"Model: {MODEL}\nTest:  {TEST}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    # 2026-07-14: unsloth silently remaps ANY Qwen3-8B name to its pre-quantized
    # mirror repo (unsloth/qwen3-8b-unsloth-bnb-4bit), which is currently missing
    # its weight files and 404s. This flag disables the remap so the official
    # Qwen/Qwen3-8B repo is loaded and quantized on the fly.
    use_exact_model_name=True,
)
FastLanguageModel.for_inference(model)

BAND = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}
ORD  = {"Freddo": 0, "Tiepido": 1, "Calduccio": 2, "Caldo": 3}
SCORE_BY_OUTCOME = {
    **BAND,
    "Non interessato": 0,
    "Non idoneo": 0,
}


def _as_parts(msgs):
    """Multimodal processors want content as a list of typed parts."""
    return [{"role": m["role"],
             "content": ([{"type": "text", "text": m["content"]}]
                         if isinstance(m["content"], str) else m["content"])}
            for m in msgs]


def _norm(enc):
    """apply_chat_template may return a tensor, a dict, or a BatchFeature ->
    return an input_ids tensor on cuda."""
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    elif isinstance(enc, dict):
        enc = enc["input_ids"]
    return enc.to("cuda")


def chat_ids(msgs):
    """Render the prompt robustly across plain tokenizers AND multimodal
    processors, with thinking per THINKING."""
    def apply(m, **kw):
        return tokenizer.apply_chat_template(
            m, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", **kw)
    # 1) plain string content + enable_thinking (works for text models)
    try:
        return _norm(apply(msgs, enable_thinking=THINKING))
    except TypeError:
        pass
    # 2) multimodal: convert to list-of-parts, keep enable_thinking
    try:
        return _norm(apply(_as_parts(msgs), enable_thinking=THINKING))
    except TypeError:
        pass
    # 3) template rejects enable_thinking -> drop it
    try:
        return _norm(apply(msgs))
    except TypeError:
        return _norm(apply(_as_parts(msgs)))


def parse_outcome(text):
    """Pull the JSON object out of the model output (after any </think>).
    Returns (outcome, score, confidence, closing_readiness)."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            d = json.loads(text[s:e + 1])
            return (d.get("outcome"), d.get("score"), d.get("confidence"),
                    d.get("closing_readiness"))
        except Exception:
            pass

    # Recover slightly malformed JSON-like answers (single quotes, missing
    # braces, or prose around key/value pairs) without accepting an ambiguous
    # response that mentions multiple LOI bands.
    outcome_match = re.search(
        r"""["']?outcome["']?\s*:\s*["']([^"']+)["']""",
        text,
        flags=re.IGNORECASE,
    )
    outcome = outcome_match.group(1).strip() if outcome_match else None
    if outcome not in SCORE_BY_OUTCOME:
        mentioned = [band for band in SCORE_BY_OUTCOME if band.lower() in text.lower()]
        outcome = mentioned[0] if len(mentioned) == 1 else None
    if outcome is None:
        return None, None, None, None

    score_match = re.search(
        r"""["']?score["']?\s*:\s*(-?\d+(?:\.\d+)?)""",
        text,
        flags=re.IGNORECASE,
    )
    confidence_match = re.search(
        r"""["']?confidence["']?\s*:\s*["']([^"']+)["']""",
        text,
        flags=re.IGNORECASE,
    )
    readiness_match = re.search(
        r"""["']?closing_readiness["']?\s*:\s*(\d{1,3})""",
        text,
        flags=re.IGNORECASE,
    )
    score = (
        float(score_match.group(1))
        if score_match
        else SCORE_BY_OUTCOME[outcome]
    )
    confidence = confidence_match.group(1).strip() if confidence_match else "medium"
    readiness = int(readiness_match.group(1)) if readiness_match else None
    return outcome, score, confidence, readiness


with open(TEST, encoding="utf-8") as f:
    rows_in = [json.loads(l) for l in f]
rows_in = rows_in[START:END]
print(f"scoring rows {START}..{END if END is not None else 'end'} -> {len(rows_in)} calls...")

# Write each prediction as it lands (+ flush every 25) so a dying session
# loses minutes, not hours. With Persistence ON the partial CSV survives.
PRED = "/kaggle/working/predictions.csv"
pred_f = open(PRED, "w", newline="", encoding="utf-8")
w = csv.DictWriter(pred_f, fieldnames=["call_id", "source", "gold", "pred",
                                       "pred_score", "confidence", "closing_readiness"])
w.writeheader()

import time
t0 = time.time()
records, mism, correct, band, n, n_gold, fails, thought, retries = (
    [], [], 0, 0, 0, 0, 0, 0, 0
)
for i, ex in enumerate(rows_in):
    msgs = ex["messages"]
    # Unlabeled calls (full-corpus run) have no assistant turn -> no gold;
    # they still get scored, they just don't enter the agreement stats.
    has_gold = msgs[-1]["role"] == "assistant"
    gold = json.loads(msgs[-1]["content"]).get("outcome") if has_gold else None
    # drop the gold (assistant) turn for generation, and append the anti-warm-tilt
    # calibration to the system prompt (rubric) without mutating the source rows.
    prompt_msgs = [dict(m) for m in (msgs[:-1] if has_gold else msgs)]
    prompt_msgs[0] = {**prompt_msgs[0],
                      "content": prompt_msgs[0]["content"] + "\n\n" + CALIBRATION}
    ids = chat_ids(prompt_msgs)
    out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW, use_cache=True,
                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]
    # Track whether the model actually produced a <think> block, so we can
    # confirm the THINKING flag took effect (and catch JSON cut off by a long
    # trace hitting MAX_NEW).
    if "<think>" in text or "</think>" in text:
        thought += 1
    pred, psc, pconf, pcr = parse_outcome(text)
    if pred is None:
        retries += 1
        retry_msgs = [dict(m) for m in prompt_msgs]
        retry_msgs[-1] = {
            **retry_msgs[-1],
            "content": str(retry_msgs[-1]["content"])
            + "\n\nISTRUZIONE FINALE: rispondi esclusivamente con un singolo "
              "oggetto JSON valido contenente outcome, score, confidence e "
              "closing_readiness. Non aggiungere spiegazioni o markdown.",
        }
        retry_ids = chat_ids(retry_msgs)
        retry_out = model.generate(
            input_ids=retry_ids,
            max_new_tokens=MAX_NEW,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        retry_text = tokenizer.batch_decode(
            retry_out[:, retry_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]
        if "<think>" in retry_text or "</think>" in retry_text:
            thought += 1
        pred, psc, pconf, pcr = parse_outcome(retry_text)
    n += 1
    if pred is None:
        fails += 1
    if has_gold:
        n_gold += 1
        if pred == gold:
            correct += 1; band += 1
        else:
            mism.append((gold, pred))
            if pred in ORD and gold in ORD and abs(ORD[pred] - ORD[gold]) <= 1:
                band += 1
    rec = {"call_id": ex.get("call_id", ""), "source": ex.get("source", "zoom"),
           "gold": gold, "pred": pred, "pred_score": psc, "confidence": pconf,
           "closing_readiness": pcr}
    records.append(rec)
    w.writerow(rec)
    if (i + 1) % 25 == 0 or (i + 1) == len(rows_in):
        pred_f.flush()
        pace = (time.time() - t0) / (i + 1)
        eta = pace * (len(rows_in) - i - 1) / 3600
        print(f"  {i+1}/{len(rows_in)}  {pace:.1f}s/call  ETA {eta:.1f}h  "
              f"(exact so far {correct}/{n_gold} gold, parse-fails {fails})")

pred_f.close()
have_ids = sum(1 for r in records if r["call_id"])
print(f"\n[saved] {PRED}  ({len(records)} rows, {have_ids} with call_id, {fails} parse-fails)")
print(f"THINKING flag = {THINKING}; <think> block seen on {thought}/{n} calls")
print(f"automatic JSON-only retries: {retries}")
if THINKING and thought < n * 0.5:
    print("  WARNING: thinking was requested but rarely happened — the template "
          "may ignore enable_thinking on this build (still fine, just not reasoning).")
if fails > n * 0.1:
    print(f"  WARNING: {fails} parse-fails — if THINKING=True, a long trace may be "
          f"hitting MAX_NEW ({MAX_NEW}); raise it or set THINKING=False.")
if n_gold:
    print(f"Exact vs Sonnet (on {n_gold} gold rows): {correct}/{n_gold} = {correct/n_gold:.1%}")
    print(f"±1 band:                                 {band}/{n_gold} = {band/n_gold:.1%}")
else:
    print("(no gold rows in this slice — agreement skipped)")
print("\nTop mismatches (gold -> pred):")
for (g, p), c in Counter(mism).most_common(12):
    print(f"  {c:2d}x  {g} -> {p}")
print("\nNow download predictions.csv and locally run:")
print("  python -m src.contract_verdict --model qwen3.5-9b-prompt --pred predictions.csv --ids data/train/test_ids.json")
