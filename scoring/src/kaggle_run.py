"""STEP 2 (Kaggle, scriptable) — push the staged jsonl + run the scorer on a
Kaggle GPU via the Kaggle API, then download predictions.csv. No web clicking.

Round-trip (all via `python -m kaggle`; the kaggle.exe shim isn't on PATH):
  1. push data/train/test_prompting_sales.jsonl as the dataset (create first time,
     else a new VERSION)
  2. push the prompt_qwen kernel (GPU + internet) -> it runs on Kaggle's GPU
  3. poll kernel status until complete
  4. download predictions.csv -> predictions_kaggle_latest.csv

Kernel code = kaggle/prompt_qwen.py with START,END auto-patched to 0,None
(an incremental batch fits one session: ~28 min for ~60 calls, inside the free
30h/week quota -> daily-safe).

AUTH: token in ~/.kaggle/access_token (the KGAT_ format) or ~/.kaggle/kaggle.json.
SLUGS: from env KAGGLE_DATASET / KAGGLE_KERNEL, else kaggle/auto/slugs.json.

Usage:
  python -m src.kaggle_run                 # DRY-RUN: prints the exact commands
  python -m src.kaggle_run --go            # push + run + fetch
  python -m src.kaggle_run --go --no-wait  # push+run, don't block on polling
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .config import DATA, ROOT

AUTO = ROOT / "kaggle" / "auto"
DS_DIR = AUTO / "dataset"
KN_DIR = AUTO / "kernel"
SLUGS_FILE = AUTO / "slugs.json"
JSONL = DATA / "train" / "test_prompting_sales.jsonl"
PROMPT_SRC = ROOT / "kaggle" / "prompt_qwen.py"
PRED_OUT = ROOT / "predictions_kaggle_latest.csv"

_scripts_kaggle = Path(sys.executable).parent / "Scripts" / "kaggle.exe"
KAGGLE = ([str(_scripts_kaggle)] if _scripts_kaggle.exists()
          else [sys.executable, "-m", "kaggle"])


def _slugs():
    ds = os.environ.get("KAGGLE_DATASET")
    kn = os.environ.get("KAGGLE_KERNEL")
    if (not ds or not kn) and SLUGS_FILE.exists():
        j = json.loads(SLUGS_FILE.read_text())
        ds = ds or j.get("dataset")
        kn = kn or j.get("kernel")
    return ds, kn


def _has_token() -> bool:
    home = Path.home() / ".kaggle"
    return ((home / "access_token").exists() or (home / "kaggle.json").exists()
            or bool(os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")))


def _preflight():
    # Retry the CLI check — rapid back-to-back subprocess spawns on Windows
    # (refresh_pull fires several right before this) can transiently fail.
    cli_ok = False
    for attempt in range(3):
        if subprocess.run(KAGGLE + ["--version"], capture_output=True).returncode == 0:
            cli_ok = True
            break
        if attempt < 2:
            time.sleep(1)
    if not cli_ok:
        return "kaggle CLI not importable — run: pip install kaggle"
    if not _has_token():
        return ("no API token — put it in ~/.kaggle/access_token (KGAT_ token) "
                "or ~/.kaggle/kaggle.json")
    ds, kn = _slugs()
    if not ds or not kn:
        return "set KAGGLE_DATASET/KAGGLE_KERNEL or fill kaggle/auto/slugs.json"
    return None


def _sh(cmd, dry):
    print("   $ " + " ".join(c if c != sys.executable else "python" for c in cmd))
    if dry:
        return 0, ""
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if out.strip():
        print("     " + out.strip().replace("\n", "\n     "))
    return p.returncode, out


def _dataset_exists(ds: str) -> bool:
    name = ds.split("/")[-1]
    p = subprocess.run(KAGGLE + ["datasets", "list", "--mine", "-s", name],
                       capture_output=True, text=True)
    return name in (p.stdout or "")


def _parse_status(output: str) -> str | None:
    match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', output, re.I)
    if not match:
        match = re.search(r'\bstatus[\"\']?\s*[:=]?\s*[\"\']([A-Z_]+)', output, re.I)
    return match.group(1).upper() if match else None


def _gpu_quota():
    """Read real account quota; SDK 2.2.2 cannot parse whole-second durations."""
    from datetime import timedelta
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kaggle_object import TimeDeltaSerializer
    api = KaggleApi()
    api.authenticate()
    original = TimeDeltaSerializer._from_dict_value
    try:
        TimeDeltaSerializer._from_dict_value = staticmethod(
            lambda value: timedelta(seconds=float(value.rstrip("s"))))
        with api.build_kaggle_client() as client:
            quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    finally:
        TimeDeltaSerializer._from_dict_value = staticmethod(original)
    gpu = quota.gpu_quota
    used = gpu.time_used.total_seconds() / 3600
    reserved = gpu.time_reserved.total_seconds() / 3600
    allowed = gpu.total_time_allowed.total_seconds() / 3600
    return allowed - used - reserved, used, reserved, allowed


def _wait_for_completion(kernel: str, poll: int, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    last_status = None
    while time.monotonic() < deadline:
        rc, output = _sh(KAGGLE + ["kernels", "status", kernel], dry=False)
        status = _parse_status(output) if rc == 0 else None
        if status == "COMPLETE":
            return True
        if status in {"ERROR", "CANCELLED", "CANCELED", "CANCEL_ACKNOWLEDGED"}:
            print(f"!! kernel {status}; no predictions downloaded.")
            return False
        if status:
            last_status = status
        if status == "QUEUED":
            print("   Queued for Kaggle GPU allocation (not yet executing). "
                  "Do not repush: use --go --resume to keep checking this run.")
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
    print(f"!! wait timed out (last status={last_status or 'unknown'}). "
          "Run remains on Kaggle; no old predictions downloaded. "
          "Continue with --go --resume, not another push.")
    return False


def _write_metadata(ds_slug, kn_slug, accelerator="NvidiaTeslaT4"):
    DS_DIR.mkdir(parents=True, exist_ok=True)
    KN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(JSONL, DS_DIR / JSONL.name)
    (DS_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": ds_slug.split("/")[-1], "id": ds_slug,
        "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    code = PROMPT_SRC.read_text(encoding="utf-8")
    code = re.sub(r"START,\s*END\s*=\s*0,\s*\d+", "START, END = 0, None", code)
    (KN_DIR / "prompt_qwen.py").write_text(code, encoding="utf-8")
    (KN_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": kn_slug, "title": kn_slug.split("/")[-1],
        "code_file": "prompt_qwen.py", "language": "python",
        "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        # PIN the T4 (sm_75). API-pushed kernels otherwise default to a P100
        # (sm_60), which the freshly pip-installed Torch 2.10+cu128 no longer
        # builds kernels for -> "CUDA error: no kernel image is available".
        # T4 is the general GPU pool. L4 is restricted to selected competitions,
        # not a fallback for this private business workload (Kaggle announcement 735239).
        "machine_shape": accelerator,
        "dataset_sources": [ds_slug], "competition_sources": [], "kernel_sources": [],
        # 2026-07-14: mount Kaggle's own Qwen3-8B (local shards, no HF download).
        # HF-side resolution broke twice today (unsloth 4-bit mirror missing
        # weights AND sharded-index resolution failing on the official repo).
        "model_sources": ["qwen-lm/qwen-3/transformers/8b/1"]},
        indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="execute (default dry-run)")
    ap.add_argument("--no-wait", action="store_true", help="don't poll for completion")
    ap.add_argument("--resume", action="store_true", help="poll existing run; do not upload or submit again")
    ap.add_argument("--accelerator", choices=["NvidiaTeslaT4"],
                    default=os.environ.get("KAGGLE_ACCELERATOR", "NvidiaTeslaT4"))
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=10800)
    a = ap.parse_args(argv)
    if a.accelerator != "NvidiaTeslaT4":
        ap.error("this workload must use the general T4 pool; L4 is competition-restricted")
    if a.poll <= 0 or a.timeout <= 0:
        ap.error("--poll and --timeout must be positive")
    dry = not a.go

    problem = _preflight()
    if problem:
        print(f"[setup needed] {problem}")
        if a.go:
            return 2
        print("(continuing in dry-run to show the plan)\n")

    ds, kn = _slugs()
    ds = ds or "<user>/loi-qwen-scoring"
    kn = kn or "<user>/loi-prompt-qwen"
    if not JSONL.exists():
        print(f"!! {JSONL} missing — run `python -m src.refresh_pull` first.")
        return 2
    with JSONL.open(encoding="utf-8") as staged:
        n_rows = sum(1 for _ in staged)
    print(f"== kaggle_run ==  dataset={ds}  kernel={kn}  jsonl_rows={n_rows}  (dry-run={dry})")

    if not dry and not a.resume:
        rc, output = _sh(KAGGLE + ["kernels", "status", kn], dry=False)
        status = _parse_status(output) if rc == 0 else None
        if status not in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED", "CANCEL_ACKNOWLEDGED"}:
            print(f"!! existing run status={status or 'unknown'}; refusing duplicate submission. "
                  "Use --go --resume, or cancel the existing run first.")
            return 2
        try:
            remaining, used, reserved, allowed = _gpu_quota()
            print(f"   GPU quota: {remaining:.2f}h available / {allowed:.2f}h "
                  f"({used:.2f}h used, {reserved:.2f}h reserved). "
                  "Available quota does not guarantee an available GPU machine.")
            if remaining <= 0:
                print("!! GPU quota exhausted; not submitting.")
                return 2
        except Exception as error:
            print(f"!! unable to verify GPU quota ({type(error).__name__}); not submitting.")
            return 2
        _write_metadata(ds, kn, a.accelerator)

    print("\n[1] push dataset" if not a.resume else "\n[1] resume: dataset upload skipped")
    if a.resume:
        pass
    elif dry:
        print("   $ python -m kaggle datasets version|create -p <dataset dir> -m 'auto refresh'")
    else:
        if _dataset_exists(ds):
            rc, _ = _sh(KAGGLE + ["datasets", "version", "-p", str(DS_DIR), "-m", "auto refresh",
                                "--dir-mode", "skip"], dry)
        else:
            print("   (first run — creating dataset)")
            rc, _ = _sh(KAGGLE + ["datasets", "create", "-p", str(DS_DIR), "--dir-mode", "skip"], dry)
        if rc != 0:
            return 1

    # 2026-07-16: WAIT until the new dataset version is actually LIVE before
    # running the kernel. Version processing is asynchronous and can take
    # >30 min; kernels attach the latest *processed* version, so pushing the
    # kernel immediately re-scores the previous batch (bit us 3x: 2026-07-08
    # gsheet batch, twice today). Verify by downloading the served file and
    # comparing bytes with what we staged.
    if not dry and not a.resume:
        import tempfile
        import zipfile
        staged_bytes = (DS_DIR / JSONL.name).read_bytes()
        deadline, live = time.time() + 1800, False
        while time.time() < deadline:
            with tempfile.TemporaryDirectory() as td:
                subprocess.run(KAGGLE + ["datasets", "download", ds,
                                         "-f", JSONL.name, "-p", td, "--force"],
                               capture_output=True, text=True)
                got = None
                for p in Path(td).iterdir():
                    if p.suffix == ".zip":
                        with zipfile.ZipFile(p) as z:
                            got = z.read(z.namelist()[0])
                    elif p.name == JSONL.name:
                        got = p.read_bytes()
                if got is not None and got == staged_bytes:
                    live = True
                    break
            print("   dataset version not live yet — waiting 60s...")
            time.sleep(60)
        if not live:
            print("!! dataset version still not live after 30 min — aborting "
                  "before the kernel re-scores the previous batch.")
            return 1
        print("   dataset version LIVE (served file matches staged bytes)")

    if not a.resume:
        print(f"\n[2] push + run kernel ({a.accelerator})")
        rc, _ = _sh(KAGGLE + ["kernels", "push", "-p", str(KN_DIR),
                             "--accelerator", a.accelerator], dry)
        if rc != 0:
            return 1
    else:
        print("\n[2] resume: kernel submission skipped")

    if a.no_wait or dry:
        print(f"\n[3] poll:    python -m kaggle kernels status {kn}")
        print(f"[4] output:  python -m kaggle kernels output {kn} -p .")
        if dry:
            print("\nDRY-RUN complete — re-run with --go to execute.")
        return 0

    print("\n[3] polling status")
    if not _wait_for_completion(kn, a.poll, a.timeout):
        return 1

    print("\n[4] download predictions")
    (AUTO / "out").mkdir(exist_ok=True)
    got = AUTO / "out" / "predictions.csv"
    # Never let Kaggle's local mtime optimization reuse an older batch. A
    # later-created local file can look "newer" than the just-finished remote
    # output and was silently reused on 2026-07-24. Remove it first and force
    # download only the prediction artifact (not the large compiled cache).
    got.unlink(missing_ok=True)
    rc, _ = _sh(KAGGLE + [
        "kernels", "output", kn,
        "-p", str(AUTO / "out"),
        "--force", "--file-pattern", "predictions.csv",
    ], dry=False)
    if rc == 0 and got.exists():
        shutil.copy(got, PRED_OUT)
        # attach leadid locally (call_id -> LEADID from the CRM merge); the
        # kernel only knows call_id, so this join happens after download.
        try:
            import pandas as pd
            from .merge_predictions import attach_leadid
            df = attach_leadid(pd.read_csv(PRED_OUT))
            df.to_csv(PRED_OUT, index=False)
        except Exception as e:
            print(f"   (leadid enrichment skipped: {e})")
        print(f"\npredictions -> {PRED_OUT}")
        print(f"NEXT: python -m src.refresh_ingest {PRED_OUT.name}")
        return 0
    print("!! predictions.csv not found in kernel output.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
