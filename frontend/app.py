"""Local web frontend for the LOI pipeline.

Buttons for Part 1 (score LOIs -> leadid Excel) and Part 2 (build the ads feed),
with live logs, CRM-file upload, status cards, and download links. It just runs
the same commands as LOI_RUNBOOK.md as subprocesses — no new logic.

Run:   python app.py        (or double-click run_frontend.bat)
Open:  http://127.0.0.1:5000

One job at a time (Kaggle can't run two). Closing the browser does NOT stop a
running job — reopen and it re-attaches to the live log.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MVP = REPO / "loi-audio-mvp"
OPTC = REPO / "option-c-value-bidding"
CRM_DIR = MVP / "data" / "crm"
MASTER = MVP / "predictions_qwen3_8b_sales_full.csv"
EXCEL_DIR = MVP / "outputs"
OPTC_STATUS = OPTC / "outputs" / "pipeline_status.json"
GOOGLE_CSV = OPTC / "outputs" / "google_offline_conversions.csv"
META_JSON = OPTC / "outputs" / "meta_capi_payload.json"
JSONL = MVP / "data" / "train" / "test_prompting_sales.jsonl"   # the staged score queue

app = Flask(__name__)

JOBS: dict[str, dict] = {}        # job_id -> {name, status, lines, returncode, started}
RUN_LOCK = threading.Lock()
CURRENT = {"id": None}


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONUNBUFFERED"] = "1"
    e["PYTHONUTF8"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def _run_job(job_id: str, steps: list[tuple[Path, list[str]]]) -> None:
    job = JOBS[job_id]
    try:
        for cwd, cmd in steps:
            shown = " ".join("python" if c == sys.executable else c for c in cmd)
            job["lines"].append(f"$ {shown}   (in {Path(cwd).name})")
            p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 env=_env(), encoding="utf-8", errors="replace")
            for line in iter(p.stdout.readline, ""):
                job["lines"].append(line.rstrip("\n"))
            p.stdout.close()
            rc = p.wait()
            if rc != 0:
                job["status"] = "error"
                job["returncode"] = rc
                job["lines"].append(f"!! step failed (exit {rc})")
                return
        job["status"] = "done"
        job["returncode"] = 0
        job["lines"].append("✓ finished")
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["lines"].append(f"!! {type(e).__name__}: {e}")
    finally:
        CURRENT["id"] = None
        RUN_LOCK.release()


def _start(name: str, steps: list[tuple[Path, list[str]]]) -> str | None:
    if not RUN_LOCK.acquire(blocking=False):
        return None
    job_id = f"{name}-{int(time.time())}"
    JOBS[job_id] = {"name": name, "status": "running", "lines": [],
                    "returncode": None, "started": datetime.now().isoformat(timespec="seconds")}
    CURRENT["id"] = job_id
    threading.Thread(target=_run_job, args=(job_id, steps), daemon=True).start()
    return job_id


def _master_count() -> int:
    if not MASTER.exists():
        return 0
    with MASTER.open(encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _queued() -> int:
    if not JSONL.exists():
        return 0
    with JSONL.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def _newest(folder: Path, pattern: str):
    xs = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return xs[0] if xs else None


def _finfo(p):
    return {"name": p.name,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")} if p else None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    optc = {}
    if OPTC_STATUS.exists():
        try:
            optc = json.loads(OPTC_STATUS.read_text(encoding="utf-8"))
        except Exception:
            optc = {}
    cur = CURRENT["id"]
    job = JOBS.get(cur) if cur else None
    excel = _newest(EXCEL_DIR, "loi_scores_*.xlsx")
    return jsonify({
        "master_count": _master_count(),
        "queued": _queued(),
        "excel": _finfo(excel),
        "crm": _finfo(_newest(CRM_DIR, "zoom_calendar_event_merged*.xlsx")),
        "optc": optc,
        "running": ({"id": cur, "name": job["name"]} if job and job["status"] == "running" else None),
        "downloads": {"excel": bool(excel), "google": GOOGLE_CSV.exists(), "meta": META_JSON.exists()},
    })


@app.route("/api/run/<part>", methods=["POST"])
def api_run(part):
    plans = {
        "fetch": ("fetch", [(MVP, [sys.executable, "-m", "src.refresh_pull"])]),
        "part1": ("part1", [(MVP, [sys.executable, "-m", "src.part1_score", "--go"])]),
        "part2": ("part2", [(OPTC, [sys.executable, "build_clickid_map.py"]),
                            (OPTC, [sys.executable, "run_pipeline.py"])]),
        "excel": ("excel", [(MVP, [sys.executable, "-m", "src.export_loi_excel"])]),
    }
    if part not in plans:
        abort(404)
    name, steps = plans[part]
    jid = _start(name, steps)
    if not jid:
        return jsonify({"error": "a job is already running"}), 409
    return jsonify({"job_id": jid})


@app.route("/api/logs/<job_id>")
def api_logs(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "no such job"}), 404
    since = int(request.args.get("since", 0))
    return jsonify({"status": job["status"], "lines": job["lines"][since:], "total": len(job["lines"])})


@app.route("/api/upload_crm", methods=["POST"])
def upload_crm():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "no file"}), 400
    if not f.filename.lower().endswith(".xlsx"):
        return jsonify({"error": "must be a .xlsx file"}), 400
    CRM_DIR.mkdir(parents=True, exist_ok=True)
    dest = CRM_DIR / f"zoom_calendar_event_merged_{date.today().isoformat()}.xlsx"
    f.save(str(dest))
    return jsonify({"saved": dest.name})


@app.route("/download/<which>")
def download(which):
    m = {"excel": _newest(EXCEL_DIR, "loi_scores_*.xlsx"), "google": GOOGLE_CSV, "meta": META_JSON}
    p = m.get(which)
    if not p or not Path(p).exists():
        abort(404)
    return send_file(str(p), as_attachment=True)


if __name__ == "__main__":
    print("LOI frontend -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)
