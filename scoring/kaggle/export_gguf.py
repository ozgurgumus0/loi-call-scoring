"""Export the fine-tuned Qwen-LOI v2 model to GGUF for Ollama.

WHY A SEPARATE SCRIPT: every in-training GGUF attempt has failed because
Unsloth's bundled exporter tries to BUILD llama.cpp, and llama.cpp switched its
build from Makefile -> CMake, which Unsloth's wrapper mishandles (it also blows
the 20 GB disk limit by writing a 15 GB fp16 GGUF next to the 15 GB merged model).

This script sidesteps BOTH problems:
  1. It uses llama.cpp's PURE-PYTHON converter (`convert_hf_to_gguf.py`) and
     quantizes straight to q8_0 in that one step -> NO C++ build at all.
  2. It reads the merged fp16 model from the read-only /kaggle/input mount, so
     the 15 GB input does NOT count against the 20 GB /kaggle/working quota.
     Only the ~8 GB q8_0 output lands in working.

HOW TO RUN (~5 min, CPU notebook is fine — no GPU needed):
  1. New Kaggle notebook. Right sidebar -> + Add Input -> Notebook Output ->
     select your committed `train_qwen_loi` run. Its output (containing
     `qwen-loi-gguf/`, the merged fp16 model Unsloth already produced) mounts
     under /kaggle/input/.
  2. Paste this whole file into one cell and run it.
  3. Download `qwen-loi-v2-q8_0.gguf` from the Output panel.
  4. Locally: see deploy/README.md (`ollama create qwen-loi -f Modelfile`).

q8_0 (8-bit) keeps essentially full quality and needs ~10 GB RAM to serve. If
your laptop is tight on RAM, see deploy/README.md for letting Ollama re-quantize.
"""
import os
import subprocess
import sys
from pathlib import Path

# --- 1. Locate the MERGED fp16 model -----------------------------------------
# The merged model dir has model.safetensors.index.json + config.json; the LoRA
# *adapter* dir has adapter_config.json instead — we want the merged one.
MERGED = None
for base in ("/kaggle/input", "/kaggle/working"):
    if not Path(base).exists():
        continue
    for root, _dirs, files in os.walk(base):
        if "model.safetensors.index.json" in files and "config.json" in files:
            MERGED = root
            break
    if MERGED:
        break
assert MERGED, (
    "Merged fp16 model not found. Attach the committed train_qwen_loi output "
    "(+ Add Input -> Notebook Output) — it contains the qwen-loi-gguf/ folder."
)
print(f"Merged model: {MERGED}")

# Sanity: convert_hf_to_gguf needs the tokenizer + chat template alongside the
# weights. Unsloth's merge saves them; warn loudly if they're missing so you can
# copy them from the adapter's final/ folder (which also has them).
_tok = [f for f in ("tokenizer.json", "tokenizer_config.json")
        if not Path(MERGED, f).exists()]
if _tok:
    print(f"WARNING: missing {_tok} in merged dir — if the convert step errors on "
          f"the tokenizer, copy tokenizer.* from the adapter's final/ folder here.")

# --- 2. Get llama.cpp (only for its pure-Python converter; we do NOT build it) -
LLAMA = Path("/kaggle/tmp/llama.cpp")
if not (LLAMA / "convert_hf_to_gguf.py").exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ggml-org/llama.cpp", str(LLAMA)],
        check=True,
    )
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     str(LLAMA / "requirements" / "requirements-convert_hf_to_gguf.txt")],
    check=True,
)

# --- 3. Convert + quantize to q8_0 in one pure-Python pass --------------------
OUT = Path("/kaggle/working/qwen-loi-v2-q8_0.gguf")
subprocess.run(
    [sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), MERGED,
     "--outfile", str(OUT), "--outtype", "q8_0"],
    check=True,
)
size_gb = OUT.stat().st_size / 1e9
print(f"\nGGUF written: {OUT}  ({size_gb:.2f} GB)")
print("Download it from the Output panel, then follow deploy/README.md.")
