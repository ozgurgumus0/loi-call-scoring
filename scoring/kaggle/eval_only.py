# %% Standalone LOI eval — recover the test metrics after a session cutoff.
#
# WHY THIS EXISTS: the v2 run trained fine (114 steps, adapter saved to
# /kaggle/working/qwen-loi-lora/final) but the session was cut off before the
# eval cell ran, so we never got the test numbers. This cell regenerates them
# in ~5-10 min WITHOUT retraining.
#
# Works in two situations:
#   (A) SAME session, still alive — the adapter is in /kaggle/working.
#   (B) FRESH notebook — attach the downloaded adapter folder AND test.jsonl
#       as Kaggle datasets; this cell auto-discovers both under /kaggle/input.
#
# It RELOADS the adapter from disk rather than reusing the in-memory `model`,
# so it is unaffected by whatever the (failed) GGUF cell did to the live object.

# Install deps (same set the training notebook used, so versions match the
# adapter). No-op/fast if the session already has them. REQUIRED on a fresh
# notebook — Kaggle does not pre-install unsloth.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "unsloth", "trl<0.9.0", "peft", "accelerate", "bitsandbytes",
                "datasets"], check=True)

import os, json, csv
from pathlib import Path
from collections import Counter
from unsloth import FastLanguageModel
from datasets import load_dataset

# --- locate the trained adapter -------------------------------------------
ADAPTER = None
if Path("/kaggle/working/qwen-loi-lora/final/adapter_config.json").exists():
    ADAPTER = "/kaggle/working/qwen-loi-lora/final"          # same session
else:
    for root, _d, files in os.walk("/kaggle/input"):         # fresh notebook
        if "adapter_config.json" in files:
            ADAPTER = root
            break
assert ADAPTER, "adapter_config.json not found in /kaggle/working or /kaggle/input"

# --- locate test set -------------------------------------------------------
# Prefer test_with_ids.jsonl (rows carry call_id+source -> joinable predictions
# for the contract verdict); fall back to plain test.jsonl.
TEST = None
for want in ("test_with_ids.jsonl", "test.jsonl"):
    for root, _d, files in os.walk("/kaggle/input"):
        if want in files:
            TEST = str(Path(root, want))
            break
    if TEST:
        break
assert TEST, "test(_with_ids).jsonl not found under /kaggle/input (attach the dataset)"
print(f"Adapter: {ADAPTER}\nTest:    {TEST}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=4096, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

test_ds = load_dataset("json", data_files=TEST, split="train")

# LOI scores for the 4 in-band outcomes; ORD = ordinal position for a clean ±1.
BAND = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}
ORD  = {"Freddo": 0, "Tiepido": 1, "Calduccio": 2, "Caldo": 3}

correct = legacy_band = adj_band = n = trunc = 0
mism = []
records = []   # per-call rows for the contract verdict (joined by call_id)
for ex in test_ds:
    msgs = ex["messages"]
    gold = json.loads(msgs[-1]["content"]).get("outcome")
    ids = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    if ids.shape[1] > 4096:
        trunc += 1
    out = model.generate(
        input_ids=ids, max_new_tokens=512, use_cache=True,
        do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]
    pred = pred_score = pred_conf = None
    try:
        s, e = text.find("{"), text.rfind("}")
        pj = json.loads(text[s:e + 1]) if s != -1 else {}
        pred = pj.get("outcome")
        pred_score = pj.get("score")
        pred_conf = pj.get("confidence")
    except Exception:
        pred = None
    n += 1
    if pred == gold:
        correct += 1; legacy_band += 1; adj_band += 1
    else:
        if pred in BAND and gold in BAND:
            if abs(BAND[pred] - BAND[gold]) <= 2:   # legacy metric (matches v1 report)
                legacy_band += 1
            if abs(ORD[pred] - ORD[gold]) <= 1:     # true adjacent-band ±1
                adj_band += 1
        mism.append((gold, pred))
    records.append({
        "call_id": ex.get("call_id", ""),       # present iff test_with_ids.jsonl
        "source": ex.get("source", "zoom"),
        "gold": gold, "pred": pred,
        "pred_score": pred_score, "confidence": pred_conf,
    })

# --- dump per-call predictions for the contract verdict --------------------
# Download this CSV, then locally:
#   python -m src.contract_verdict --model qwen-7b-v3 --pred predictions.csv \
#       --ids data/train/test_ids.json
PRED_CSV = "/kaggle/working/predictions.csv"
with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["call_id", "source", "gold", "pred",
                                      "pred_score", "confidence"])
    w.writeheader(); w.writerows(records)
_have_ids = sum(1 for r in records if r["call_id"])
print(f"\n[saved] {PRED_CSV}  ({len(records)} rows, {_have_ids} with call_id)")
if not _have_ids:
    print("  WARNING: no call_id on any row — you ran on plain test.jsonl. "
          "Upload test_with_ids.jsonl (python -m src.emit_test_with_ids) so "
          "predictions can be joined to conversion_outcome for the verdict.")

print(f"\nTest n = {n}    inputs > 4096 tok: {trunc}   (v2 target = 0)")
print(f"Exact outcome:    {correct}/{n} = {correct/n:.1%}    (v1 = 27.7%)")
print(f"Band, legacy <=2: {legacy_band}/{n} = {legacy_band/n:.1%}    (v1 = 49.2%)")
print(f"Band, true +-1:   {adj_band}/{n} = {adj_band/n:.1%}")
print("\nTop mismatches (gold -> pred):")
for (g, p), c in Counter(mism).most_common(15):
    print(f"  {c:2d}x  {g} -> {p}")
