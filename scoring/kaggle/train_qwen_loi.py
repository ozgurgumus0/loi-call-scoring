"""Kaggle T4 training script for Qwen-LoRA on LOI scoring.

Paste this into a Kaggle notebook (or upload as a script). Each block
delimited by `# %%` can be a separate cell.

Required Kaggle setup (see kaggle/README.md):
  - Accelerator: GPU T4 x2 (or single T4 — single is enough)
  - Dataset attached: a Kaggle Dataset containing train.jsonl / val.jsonl /
    test.jsonl from data/train/ on your local machine
  - The dataset's mount path is typically `/kaggle/input/<dataset-slug>/`
  - CRITICAL: turn ON notebook Persistence (Settings -> Persistence ->
    "Files only"). Without it, a disconnect/timeout/computer-restart starts a
    FRESH session with an EMPTY /kaggle/working and the trained adapter is
    gone. Also download OUTPUT_DIR/'final' the moment training finishes.

Expected wall time: ~8-9 hours on T4 — Qwen 14B QLoRA, 1 epoch, ~1,077 train +
eval on 200 (7B was ~4-5h at 2 epochs).
Expected GPU RAM peak: ~13-15 GB (Qwen 14B in 4-bit + LoRA — tight on a 16 GB T4).

v2 changes vs v1 (27.69% exact / 49.23% band, n=65):
  - condensed QWEN_RUBRIC_PROMPT (~487 tok, was 1324) + tighter char budget,
    so every example fits well under 4096 tok (no truncation at eval time).
  - more data: 595 train / 35 val / 66 test (was 330/34/65).
  DATA-only changes baked into data/train/*.jsonl — nothing changes in this
  script except re-exporting the Kaggle dataset.

v1 changes vs v0 (15% exact / 43% on short calls):
  1. DATA pre-processed locally (src.transcript_prep): same-speaker turns
     collapsed + head/tail truncation so the label is never cut (v0 lost it
     on 76% of calls).
  2. train_on_responses_only — loss on the assistant JSON only (cell 7b).

RUN THIS AS A COMMITTED BATCH (Save Version -> "Save & Run All (Commit)") with
Persistence ON. A committed run executes detached on Kaggle's servers and
PERSISTS everything in /kaggle/working as the version Output, so the adapter and
GGUF survive even if your browser/PC dies (the v2 interactive run was lost
exactly this way). After it finishes, download qwen-loi-lora/final (the 14B
adapter, ~320 MB) from the Output tab. For 14B the in-run GGUF is OFF (cell 10),
so build it afterward with kaggle/export_gguf.py on this committed output.

IMPORTANT ordering: the eval cell (9) runs BEFORE the GGUF export (10) so the
valuable test numbers are captured first. The GGUF export is now build-free and
disk-safe (cell 10) and runs by default; if you ever still lose the eval output,
kaggle/eval_only.py reproduces it from the saved adapter in ~5-10 min.
"""

# %% 1. Install dependencies
# (Skip on re-runs — Kaggle caches installs within a session)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "unsloth", "trl<0.9.0", "peft", "accelerate", "bitsandbytes",
                "datasets"], check=True)

# %% 2. Imports + config
import json
import os
import torch
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

MAX_SEQ_LENGTH = 4096          # T4 OOMs at 8192 during eval. v1 data is
                               # pre-truncated (head/tail) to fit 4096 with
                               # headroom, so the label is never cut.
BASE_MODEL    = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"   # 14B capacity test (QLoRA fits a 16GB T4, tight). 7B was "...Qwen2.5-7B-Instruct-bnb-4bit".
LORA_RANK     = 16
LORA_ALPHA    = 16
NUM_EPOCHS    = 1          # 14B is ~2x slower/step than 7B; at 1,353 ex, 2 epochs
                           # would be ~14h train + ~2h eval = OVER the 12h Kaggle cap.
                           # 1 epoch (~7h train) keeps the committed run safe. (7B used 2.)
LR            = 2e-4
SEED          = 42

# Locate the input dataset. Kaggle's mount layout varies: sometimes it's
# /kaggle/input/<slug>/, sometimes /kaggle/input/datasets/<slug>/, etc.
# Walk the tree to find train.jsonl wherever it ended up.
INPUT_ROOT = Path("/kaggle/input")
DATA_DIR = None
if INPUT_ROOT.exists():
    for root, _dirs, files in os.walk(INPUT_ROOT):
        if "train.jsonl" in files:
            DATA_DIR = Path(root)
            break
assert DATA_DIR is not None, (
    "train.jsonl not found anywhere under /kaggle/input. "
    "Did you attach the dataset to this notebook (right sidebar -> "
    "+ Add Input -> Your Datasets -> loi-qwen-training)?"
)
print(f"Using data from: {DATA_DIR}")

OUTPUT_DIR = Path("/kaggle/working/qwen-loi-lora")
GGUF_DIR   = Path("/kaggle/working/qwen-loi-gguf")

# %% 3. Load base model + tokenizer (4-bit quantized, fits on T4)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = BASE_MODEL,
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = None,    # auto: bf16 on Ampere+, fp16 otherwise
    load_in_4bit    = True,
)

# %% 4. Wrap with LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r                  = LORA_RANK,
    target_modules     = ["q_proj","k_proj","v_proj","o_proj",
                          "gate_proj","up_proj","down_proj"],
    lora_alpha         = LORA_ALPHA,
    lora_dropout       = 0,
    bias               = "none",
    use_gradient_checkpointing = "unsloth",
    random_state       = SEED,
    use_rslora         = False,
    loftq_config       = None,
)

# %% 5. Load datasets
train_ds = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train")
val_ds   = load_dataset("json", data_files=str(DATA_DIR / "val.jsonl"),   split="train")
print(f"train: {len(train_ds)}   val: {len(val_ds)}")

# %% 6. Apply Qwen 2.5 chat template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

def _format(batch):
    convos = batch["messages"]
    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in convos
    ]
    return {"text": texts}

train_ds = train_ds.map(_format, batched=True, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(_format,   batched=True, remove_columns=val_ds.column_names)

# %% 7. Train
trainer = SFTTrainer(
    model            = model,
    tokenizer        = tokenizer,
    train_dataset    = train_ds,
    eval_dataset     = val_ds,
    dataset_text_field = "text",
    max_seq_length   = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing          = False,   # for variable-length structured outputs
    args = TrainingArguments(
        per_device_train_batch_size  = 1,
        per_device_eval_batch_size   = 1,    # eval was OOMing at 2
        gradient_accumulation_steps  = 8,    # effective batch 8
        warmup_ratio                 = 0.03,
        num_train_epochs             = NUM_EPOCHS,
        learning_rate                = LR,
        fp16  = not torch.cuda.is_bf16_supported(),
        bf16  = torch.cuda.is_bf16_supported(),
        logging_steps                = 5,
        # Skip mid-training eval — it OOMed last time without gradient
        # checkpointing. We still get a real test-set eval in cell 10.
        eval_strategy                = "no",
        # Save every 15 steps so a crash loses at most ~15 min of work.
        # Re-running with trainer.train(resume_from_checkpoint=True) picks
        # up from the last checkpoint automatically.
        save_strategy                = "steps",
        save_steps                   = 15,
        save_total_limit             = 2,
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "cosine",
        seed                         = SEED,
        output_dir                   = str(OUTPUT_DIR),
        report_to                    = "none",
    ),
)
# %% 7b. Completion-only loss (v1) — mask everything except the assistant JSON.
# Without this, most of the loss is spent learning to reconstruct the
# transcript; the model barely learns the actual scoring. With it, gradients
# flow only through the JSON answer. Qwen-2.5 chat markers below.
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

# Auto-detect the latest checkpoint (None on fresh runs, path on resume).
# Within a single Kaggle session this lets you re-run cell 7 after a
# crash and pick up where it left off. Across sessions you'd need to
# upload the checkpoint folder as a Kaggle Dataset and adjust OUTPUT_DIR.
def _latest_checkpoint() -> str | None:
    if not OUTPUT_DIR.exists():
        return None
    ckpts = sorted(
        [d for d in OUTPUT_DIR.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.rsplit("-", 1)[1]),
    )
    return str(ckpts[-1]) if ckpts else None

ckpt = _latest_checkpoint()
print(f"Resuming from: {ckpt}" if ckpt else "Starting training from scratch.")
trainer.train(resume_from_checkpoint=ckpt)

# %% 8. Save final LoRA adapter + tokenizer (small, ~160 MB)
model.save_pretrained(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
print(f"LoRA adapter saved to {OUTPUT_DIR / 'final'}")

# %% 9. Eval on the held-out test set  -- RUNS BEFORE GGUF ON PURPOSE.
# Eval is the valuable output and takes ~5-10 min; the optional GGUF export
# below always fails on Kaggle and wastes ~5 min, so it must not run first and
# risk eating the session before the eval prints. If this output is ever lost,
# kaggle/eval_only.py reproduces it from the saved adapter without retraining.
test_ds_raw = load_dataset("json", data_files=str(DATA_DIR / "test.jsonl"), split="train")
FastLanguageModel.for_inference(model)

BAND = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}
ORD  = {"Freddo": 0, "Tiepido": 1, "Calduccio": 2, "Caldo": 3}
correct = legacy_band = adj_band = n = trunc = 0
mismatches = []
for ex in test_ds_raw:
    messages = ex["messages"]
    gold = json.loads(messages[-1]["content"]).get("outcome")
    inputs = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
        return_tensors="pt", max_length=MAX_SEQ_LENGTH, truncation=True,  # clamp INPUT: a 14B eval OOMs on one over-long call
    ).to("cuda")
    if inputs.shape[1] > MAX_SEQ_LENGTH:
        trunc += 1
    out = model.generate(
        input_ids=inputs, max_new_tokens=320, use_cache=True,   # 320 ample for the JSON answer; smaller KV cache on 14B
        do_sample=False,    # greedy; newer transformers reject temperature=0.0
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.batch_decode(out[:, inputs.shape[1]:], skip_special_tokens=True)[0]
    try:
        start, end = text.find("{"), text.rfind("}")
        pred = json.loads(text[start:end + 1]).get("outcome") if start != -1 else None
    except Exception:
        pred = None
    n += 1
    if pred == gold:
        correct += 1; legacy_band += 1; adj_band += 1
    else:
        if pred in BAND and gold in BAND:
            if abs(BAND[pred] - BAND[gold]) <= 2:   # legacy metric (comparable to v1)
                legacy_band += 1
            if abs(ORD[pred] - ORD[gold]) <= 1:     # true adjacent-band ±1
                adj_band += 1
        mismatches.append({"gold": gold, "pred": pred})

print(f"Test set: n = {n}    inputs > {MAX_SEQ_LENGTH} tok: {trunc}   (v2 target = 0)")
print(f"Exact-outcome agreement:   {correct}/{n} = {correct/n:.2%}    (v1 = 27.69%)")
print(f"Within band (legacy <=2):  {legacy_band}/{n} = {legacy_band/n:.2%}    (v1 = 49.23%)")
print(f"Within band (true +-1):    {adj_band}/{n} = {adj_band/n:.2%}")
print(f"First 10 mismatches: {mismatches[:10]}")

# %% 10. Export GGUF for Ollama — BUILD-FREE + DISK-SAFE (runs by default).
# Unsloth's save_pretrained_gguf has failed on every Kaggle run: it tries to
# BUILD llama.cpp (whose build moved Makefile->CMake) and overruns the 20 GB
# /kaggle/working quota (15 GB merged + a 15 GB fp16 GGUF side by side). This
# path avoids BOTH problems:
#   - merge the fp16 weights to /tmp (root filesystem — does NOT count against
#     the 20 GB working quota),
#   - convert straight to q8_0 with llama.cpp's PURE-PYTHON converter (no C++
#     build at all), writing only the ~8 GB result into /kaggle/working.
# In a committed run the .gguf lands in the saved Output — download it directly,
# no second notebook needed. q8_0 keeps ~full quality and needs ~10 GB RAM to
# serve; if your laptop is tight, see deploy/README.md to re-quantize on import.
import subprocess as _sp
import sys as _sys
from pathlib import Path as _P

EXPORT_GGUF = False        # 14B: skip in-run GGUF (merge too big/slow for the 12h cap);
                           # run the standalone kaggle/export_gguf.py on the committed
                           # output afterward (proven build-free path). 7B had this True.
if EXPORT_GGUF:
    try:
        MERGED_TMP = "/tmp/qwen-loi-merged"            # 15 GB on /tmp, off-quota
        model.save_pretrained_merged(MERGED_TMP, tokenizer,
                                     save_method="merged_16bit")
        _LL = _P("/kaggle/tmp/llama.cpp")
        if not (_LL / "convert_hf_to_gguf.py").exists():
            _sp.run(["git", "clone", "--depth", "1",
                     "https://github.com/ggml-org/llama.cpp", str(_LL)], check=True)
        _sp.run([_sys.executable, "-m", "pip", "install", "-q", "-r",
                 str(_LL / "requirements" / "requirements-convert_hf_to_gguf.txt")],
                check=True)
        _OUT = "/kaggle/working/qwen-loi-v2-q8_0.gguf"
        _sp.run([_sys.executable, str(_LL / "convert_hf_to_gguf.py"), MERGED_TMP,
                 "--outfile", _OUT, "--outtype", "q8_0"], check=True)
        print(f"GGUF saved to {_OUT} "
              f"({_P(_OUT).stat().st_size / 1e9:.2f} GB) — download from Output.")
    except Exception as e:
        print(f"GGUF export failed ({e}); the LoRA adapter at "
              f"{OUTPUT_DIR / 'final'} is still in the saved Output — you can "
              f"convert it later (rebuild merged from the adapter, then "
              f"kaggle/export_gguf.py).")
