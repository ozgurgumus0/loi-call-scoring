"""Ingest Whisper-generated VTTs (Kaggle loi-whisper-rescue output) into the
normal transcript store, so the standard scoring pipeline picks them up.

For each *rescue* row of the kernel's manifest: copy vtt/<stem>.vtt to
data/transcripts/zoom/<stem>.vtt and patch the call's calls.csv row
(transcript_file, transcript_bytes). Rows whose call already has a usable
Zoom VTT (>= 500 B) are skipped — Whisper never overwrites a native
transcript. calls.csv gets a .bak_whisper backup on first write.

Usage:
  python -m src.ingest_whisper_vtt --out-dir kaggle/whisper/out          # dry-run
  python -m src.ingest_whisper_vtt --out-dir kaggle/whisper/out --go
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .config import DATA

TDIR = DATA / "transcripts" / "zoom"
# Whisper VTTs below this are silence/greeting fragments (verified 2026-07-08:
# 53/54 rescue files were near-silent) — scoring them yields junk cold scores.
MIN_VTT = 2000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="downloaded kernel output (has vtt/ + manifest.csv "
                         "or whisper_results.csv)")
    ap.add_argument("--manifest", help="manifest csv w/ call_id,file,role "
                                       "(default <out-dir>/manifest.csv)")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    vtt_dir = out / "vtt"
    mf_path = Path(a.manifest) if a.manifest else out / "manifest.csv"
    mf = pd.read_csv(mf_path)
    if "role" in mf.columns:
        mf = mf[mf["role"] == "rescue"]

    calls_path = DATA / "calls.csv"
    calls = pd.read_csv(calls_path, low_memory=False)
    calls["call_id"] = calls["call_id"].astype(str)
    idx = calls.set_index("call_id").index

    patched = skipped_native = missing = tiny = 0
    for _, r in mf.iterrows():
        cid = str(r["call_id"])
        stem = Path(str(r["file"])).stem
        src = vtt_dir / f"{stem}.vtt"
        if not src.exists():
            missing += 1
            continue
        if src.stat().st_size < MIN_VTT:
            tiny += 1
            continue
        rel = f"transcripts/zoom/{stem}.vtt"
        dest = DATA / rel
        # never overwrite a usable native Zoom VTT
        if dest.exists() and dest.stat().st_size >= MIN_VTT:
            mask = calls["call_id"] == cid
            has = calls.loc[mask, "transcript_file"].fillna("").str.len().gt(0)
            if has.any():
                skipped_native += 1
                continue
        if a.go:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            mask = calls["call_id"] == cid
            calls.loc[mask, "transcript_file"] = rel
            calls.loc[mask, "transcript_bytes"] = dest.stat().st_size
        patched += 1

    print(f"rescue rows: {len(mf)}  -> patch {patched}, "
          f"native-kept {skipped_native}, vtt-missing {missing}, "
          f"vtt-too-small {tiny}")
    if a.go and patched:
        bak = calls_path.with_suffix(".csv.bak_whisper")
        if not bak.exists():
            shutil.copy2(calls_path, bak)
        calls.to_csv(calls_path, index=False)
        print(f"[patched] {calls_path}  (+backup {bak.name})")
    elif not a.go:
        print("[dry-run] nothing written; use --go")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
