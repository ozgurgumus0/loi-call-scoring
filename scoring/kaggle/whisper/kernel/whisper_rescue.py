"""Kaggle GPU kernel — transcribe the LOI rescue audio batch with Whisper.

Input : /kaggle/input/loi-whisper-audio/  (54 .m4a + manifest.csv)
Output: /kaggle/working/vtt/<stem>.vtt    (WebVTT per call, Italian)
        /kaggle/working/whisper_results.csv (call_id, file, n_segments,
                                             chars, seconds, status)

Model: faster-whisper large-v3, T4 fp16, VAD on, language=it.
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "faster-whisper"], check=True)

from faster_whisper import WhisperModel  # noqa: E402

IN_DIR = Path("/kaggle/input/loi-whisper-audio")
if not (IN_DIR / "manifest.csv").exists():
    # dataset may mount under a different slug or nested dir — locate it
    hits = list(Path("/kaggle/input").rglob("manifest.csv"))
    print("mount listing:", [str(p) for p in Path("/kaggle/input").rglob("*")][:40])
    if not hits:
        raise SystemExit("manifest.csv not found anywhere under /kaggle/input "
                         "(dataset version still processing? re-run)")
    IN_DIR = hits[0].parent
    print("using input dir:", IN_DIR)
OUT_VTT = Path("/kaggle/working/vtt")
OUT_VTT.mkdir(parents=True, exist_ok=True)
RESULTS = Path("/kaggle/working/whisper_results.csv")


def ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


model = WhisperModel("large-v3", device="cuda", compute_type="float16")

rows = []
manifest = list(csv.DictReader(open(IN_DIR / "manifest.csv")))
t0 = time.time()
for i, entry in enumerate(manifest, 1):
    fname = entry["file"]
    audio = IN_DIR / fname
    stem = Path(fname).stem
    status, n_seg, chars, dur = "ok", 0, 0, 0.0
    try:
        segments, info = model.transcribe(str(audio), language="it",
                                          vad_filter=True)
        cues = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            cues.append(f"{ts(seg.start)} --> {ts(seg.end)}\n{text}\n")
            n_seg += 1
            chars += len(text)
        dur = float(info.duration or 0)
        vtt = "WEBVTT\n\n" + "\n".join(cues)
        (OUT_VTT / f"{stem}.vtt").write_text(vtt, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        status = f"error: {type(e).__name__}: {str(e)[:120]}"
    rows.append({"call_id": entry["call_id"], "file": fname,
                 "n_segments": n_seg, "chars": chars,
                 "seconds": round(dur, 1), "status": status})
    el = time.time() - t0
    print(f"[{i}/{len(manifest)}] {fname} segs={n_seg} chars={chars} "
          f"dur={dur:.0f}s elapsed={el/60:.1f}min", flush=True)
    # flush results incrementally so a dead session keeps partial work
    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

print("DONE", len(rows), "files")
