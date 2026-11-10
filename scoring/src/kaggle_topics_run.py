"""Push + run the BOOKING topic-extraction kernel on Kaggle, download output.

Same round-trip as kaggle_run but for the topics-booking pair (own dataset +
kernel slugs, fully isolated from the daily LOI scoring pipeline):
  dataset <KAGGLE_USERNAME>/loi-topics-booking          (file: topics_booking.jsonl)
  kernel  <KAGGLE_USERNAME>/loi-extract-topics-booking  (code: kaggle/extract_topics_booking.py)
  (slugs MUST differ: a kernel and a dataset with the SAME slug 409-conflict
  on kernels push — they share the user's URL namespace)

The --jsonl batch file is copied into the dataset as topics_booking.jsonl (the
fixed name the kernel looks for). Output topics_booking_qwen.jsonl is saved to
--out (default outputs/topics_booking_qwen_latest.jsonl).

Usage:
  python -m src.kaggle_topics_run --jsonl data/train/topics_booking_b1.jsonl          # dry-run
  python -m src.kaggle_topics_run --jsonl data/train/topics_booking_b1.jsonl --go
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .config import ROOT

AUTO = ROOT / "kaggle" / "auto_topics"
DS_DIR = AUTO / "dataset"
KN_DIR = AUTO / "kernel"
CODE_SRC = ROOT / "kaggle" / "extract_topics_booking.py"
DS_SLUG = "<KAGGLE_USERNAME>/loi-topics-booking"
KN_SLUG = "<KAGGLE_USERNAME>/loi-extract-topics-booking"

KAGGLE = [sys.executable, "-m", "kaggle"]


def _sh(cmd, dry):
    print("   $ " + " ".join(c if c != sys.executable else "python" for c in cmd))
    if dry:
        return 0, ""
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if out.strip():
        print("     " + out.strip().replace("\n", "\n     "))
    return p.returncode, out


def _dataset_exists() -> bool:
    name = DS_SLUG.split("/")[-1]
    p = subprocess.run(KAGGLE + ["datasets", "list", "--mine", "-s", name],
                       capture_output=True, text=True)
    return name in (p.stdout or "")


def _write_metadata(jsonl: Path):
    DS_DIR.mkdir(parents=True, exist_ok=True)
    KN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(jsonl, DS_DIR / "topics_booking.jsonl")
    (DS_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": DS_SLUG.split("/")[-1], "id": DS_SLUG,
        "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    shutil.copy(CODE_SRC, KN_DIR / "extract_topics_booking.py")
    (KN_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KN_SLUG, "title": KN_SLUG.split("/")[-1],
        "code_file": "extract_topics_booking.py", "language": "python",
        "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        # PIN the T4 (sm_75) — same P100/Torch trap as kaggle_run.
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DS_SLUG], "competition_sources": [],
        "kernel_sources": [],
        # local Qwen3-8B mount — HF-hub loading broke 2026-07-14, see kaggle_run
        "model_sources": ["qwen-lm/qwen-3/transformers/8b/1"]}, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="batch jsonl to run")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "topics_booking_qwen_latest.jsonl"))
    ap.add_argument("--go", action="store_true", help="execute (default dry-run)")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=32400)
    a = ap.parse_args(argv)
    dry = not a.go

    jsonl = Path(a.jsonl)
    if not jsonl.exists():
        print(f"!! {jsonl} missing — run `python -m src.emit_topics_booking` first.")
        return 2
    n_rows = sum(1 for _ in jsonl.open(encoding="utf-8"))
    print(f"== kaggle_topics_run ==  dataset={DS_SLUG}  kernel={KN_SLUG}  "
          f"jsonl={jsonl.name} rows={n_rows}  (dry-run={dry})")

    if not dry:
        _write_metadata(jsonl)

    print("\n[1] push dataset")
    if dry:
        print("   $ python -m kaggle datasets version|create -p <dataset dir>")
    else:
        if _dataset_exists():
            _sh(KAGGLE + ["datasets", "version", "-p", str(DS_DIR),
                          "-m", f"batch {jsonl.name}", "--dir-mode", "skip"], dry)
        else:
            print("   (first run — creating dataset)")
            _sh(KAGGLE + ["datasets", "create", "-p", str(DS_DIR),
                          "--dir-mode", "skip"], dry)
            # dataset processing lag before a kernel can attach it
            time.sleep(30)

    print("\n[2] push + run kernel (GPU)")
    _sh(KAGGLE + ["kernels", "push", "-p", str(KN_DIR)], dry)

    if a.no_wait or dry:
        print(f"\n[3] poll:    python -m kaggle kernels status {KN_SLUG}")
        print(f"[4] output:  python -m kaggle kernels output {KN_SLUG} -p .")
        if dry:
            print("\nDRY-RUN complete — re-run with --go to execute.")
        return 0

    print("\n[3] polling status")
    waited = 0
    while waited < a.timeout:
        _, out = _sh(KAGGLE + ["kernels", "status", KN_SLUG], dry=False)
        low = out.lower()
        if "kernelworkerstatus.complete" in low or 'status "complete"' in low:
            break
        if ("kernelworkerstatus.error" in low or "kernelworkerstatus.cancel" in low
                or 'status "error"' in low):
            print("!! kernel errored — see Kaggle logs.")
            return 1
        time.sleep(a.poll)
        waited += a.poll

    print("\n[4] download output")
    out_dir = AUTO / "out"
    out_dir.mkdir(exist_ok=True)
    _sh(KAGGLE + ["kernels", "output", KN_SLUG, "-p", str(out_dir)], dry=False)
    got = out_dir / "topics_booking_qwen.jsonl"
    if got.exists():
        shutil.copy(got, a.out)
        print(f"\ntopics -> {a.out}")
        return 0
    print("!! topics_booking_qwen.jsonl not found in kernel output.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
