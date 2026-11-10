# %% Qwen3-8B topic extraction — BOOKING-CALL edition (xCally OMNIBUS corpus).
#
# Same mechanics as extract_topics_qwen.py (T4, 4-bit, no thinking, incremental
# jsonl write) but a BOOKING-specific controlled vocabulary: these are short
# phone qualification / appointment-setting / reminder calls, not consultations.
# There are NO Claude gold labels for these calls (no-Claude-spend constraint),
# so there is no in-kernel validation phase — validation happens locally after
# ingest by cross-checking tags against the agents' own dispositions
# (not_interested tag vs NON_INTERESSATO, etc.).
#
# INPUT  (Kaggle dataset): topics_booking.jsonl
#        (built locally by `python -m src.emit_topics_booking`; the system
#        message already carries the full prompt+guidance — nothing appended)
# OUTPUT: /kaggle/working/topics_booking_qwen.jsonl
#        download, then locally: python -m src.analyze_xcally_booking

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "unsloth", "unsloth_zoo"], check=True)

import os, json, time
from pathlib import Path
from collections import Counter
from unsloth import FastLanguageModel

# prefer the Kaggle-mounted Qwen3-8B (model_sources in kernel metadata) — the
# HF-hub paths broke 2026-07-14; falls back to the hub repo if no mount found
MODEL = "Qwen/Qwen3-8B"
import os as _os
for _root, _dirs, _files in _os.walk("/kaggle/input"):
    if "model.safetensors.index.json" in _files and "config.json" in _files:
        MODEL = _root
        break
THINKING = False
# Transcripts are head/tail-capped at 7500 chars (~2.5-3k tok) + ~1k tok prompt.
MAX_SEQ = 8192
# Output schema is small (4 label lists + <=2 short verbatim questions);
# booking calls are short. 512 leaves ample headroom.
MAX_NEW = 512
START, END = 0, None   # session slicing (row indices), patched by the runner

# Controlled vocabulary (MUST match BOOKING_TOPIC_PROMPT in
# src/emit_topics_booking.py) — drops invented tokens, prints prevalence.
VOCAB = {
    "topics_discussed": {
        "appointment_scheduling", "appointment_reminder", "zoom_logistics",
        "eligibility_screening", "bill_energy_costs", "product_overview",
        "savings", "incentives", "financing", "price",
        "existing_pv_or_quotes", "callback_scheduling", "data_confirmation"},
    "objections_raised": {
        "not_interested", "no_time_now", "already_has_system",
        "already_evaluating_others", "renter_or_moving", "too_expensive",
        "distrust_phone_sales", "decision_maker_absent", "age_or_health",
        "never_requested", "home_unsuitable"},
    "booking_signals": {
        "accepts_appointment", "proposes_own_time", "confirms_attendance",
        "asks_product_questions", "asks_savings_incentives",
        "mentions_high_bill", "requests_info_material", "mentions_timeline"},
    "call_context": {
        "gatekeeper_or_third_party", "wrong_number", "language_barrier",
        "broken_audio", "missed_appointment_followup"},
}
LIST_FIELDS = list(VOCAB.keys())

# --- locate the input ---------------------------------------------------------
TEST = None
for root, _d, files in os.walk("/kaggle/input"):
    if "topics_booking.jsonl" in files:
        TEST = str(Path(root, "topics_booking.jsonl")); break
assert TEST, "topics_booking.jsonl not found under /kaggle/input"
print(f"Model: {MODEL}\nInput: {TEST}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    # 2026-07-14: unsloth remaps Qwen3-8B names to its pre-quantized mirror repo,
    # which broke (missing weight files). Load the named repo exactly instead.
    use_exact_model_name=True,
)
FastLanguageModel.for_inference(model)


def _as_parts(msgs):
    return [{"role": m["role"],
             "content": ([{"type": "text", "text": m["content"]}]
                         if isinstance(m["content"], str) else m["content"])}
            for m in msgs]


def _norm(enc):
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    elif isinstance(enc, dict):
        enc = enc["input_ids"]
    return enc.to("cuda")


def chat_ids(msgs):
    def apply(m, **kw):
        return tokenizer.apply_chat_template(
            m, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", **kw)
    try:
        return _norm(apply(msgs, enable_thinking=THINKING))
    except TypeError:
        pass
    try:
        return _norm(apply(_as_parts(msgs), enable_thinking=THINKING))
    except TypeError:
        pass
    try:
        return _norm(apply(msgs))
    except TypeError:
        return _norm(apply(_as_parts(msgs)))


def parse_extraction(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        d = json.loads(text[s:e + 1])
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def enforce_vocab(ex):
    dropped = 0
    for f in LIST_FIELDS:
        vals = ex.get(f)
        if not isinstance(vals, list):
            ex[f] = []
            continue
        keep = [v for v in vals if isinstance(v, str) and v in VOCAB[f]]
        dropped += len(vals) - len(keep)
        ex[f] = sorted(set(keep))
    # customer_questions: keep as-is but force a short list of strings
    qs = ex.get("customer_questions")
    ex["customer_questions"] = ([q for q in qs if isinstance(q, str)][:2]
                                if isinstance(qs, list) else [])
    return ex, dropped


# --- work list ----------------------------------------------------------------
with open(TEST, encoding="utf-8") as f:
    rows_in = [json.loads(l) for l in f]
work = rows_in[START:END]
print(f"rows {START}..{END if END is not None else len(rows_in)} "
      f"-> {len(work)} calls")

# --- run ----------------------------------------------------------------------
OUT = "/kaggle/working/topics_booking_qwen.jsonl"
prev = {f: Counter() for f in LIST_FIELDS}
fails, dropped_total, overflows, thought = 0, 0, 0, 0
t0 = time.time()

with open(OUT, "w", encoding="utf-8") as out_f:
    for i, ex_row in enumerate(work):
        ids = chat_ids(ex_row["messages"])
        if ids.shape[1] + MAX_NEW > MAX_SEQ:
            overflows += 1
            print(f"  WARNING row {i} ({ex_row.get('call_id','')}): input "
                  f"{ids.shape[1]} tok overflows MAX_SEQ — skipped")
            text = ""
        else:
            out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW,
                                 use_cache=True, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.batch_decode(out[:, ids.shape[1]:],
                                          skip_special_tokens=True)[0]
        if "<think>" in text or "</think>" in text:
            thought += 1
        ex = parse_extraction(text)
        parse_fail = ex is None
        if parse_fail:
            fails += 1
            ex = {}
        ex, dropped = enforce_vocab(ex)
        dropped_total += dropped
        for f in LIST_FIELDS:
            prev[f].update(ex[f])

        out_f.write(json.dumps({
            "call_id": ex_row.get("call_id", ""),
            "source": ex_row.get("source", "xcally"),
            "model": "qwen3-8b-topics-booking",
            "extraction": ex,
            "vocab_dropped": dropped,
            "parse_fail": parse_fail,
        }, ensure_ascii=False) + "\n")

        if (i + 1) % 25 == 0 or (i + 1) == len(work):
            out_f.flush()
            pace = (time.time() - t0) / (i + 1)
            eta = pace * (len(work) - i - 1) / 3600
            print(f"  {i+1}/{len(work)}  {pace:.1f}s/call  ETA {eta:.1f}h  "
                  f"(parse-fails {fails}, vocab-dropped {dropped_total})")

print(f"\n[saved] {OUT}  ({len(work)} rows, {fails} parse-fails, "
      f"{dropped_total} invented tokens dropped, {overflows} overflow-skipped)")
print(f"THINKING flag = {THINKING}; <think> seen on {thought}/{len(work)} calls")

print("\n=== label prevalence ===")
for f in LIST_FIELDS:
    print(f"\n[{f}]")
    for lab, n in prev[f].most_common():
        print(f"  {lab:28s} {n:5d}  ({n/len(work):.0%})")

print("\nDownload topics_booking_qwen.jsonl, then locally run "
      "python -m src.analyze_xcally_booking")
