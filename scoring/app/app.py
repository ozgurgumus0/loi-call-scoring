"""LOI Audio Scoring — Flask app (replaces the Streamlit demo).

Drag-and-drop UI for scoring Italian sales calls. Accepts audio or a
pre-existing transcript (.txt / .vtt). Transcription uses local Whisper
by default and falls back to OpenAI's Whisper API on request.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv  # noqa: E402
import datetime as _dt  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

from src import rubric  # noqa: E402
from src.build_calls_csv import merge as merge_calls_csv  # noqa: E402
from src.config import (  # noqa: E402
    AUDIO_DIR, BACKEND_ANTHROPIC, BACKEND_OLLAMA, CALLS_CSV, CEILING_MODEL,
    DATA, DEFAULT_BACKEND, EXPECTED_DIST, OLLAMA_MODEL, OUT_SCORES,
    RUBRIC_VERSION, SCORING_MODEL, env,
)
from src.score_llm import (  # noqa: E402
    ollama_reachable, safe_filename, score_call, write_score,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB cap

SUPPORTED_AUDIO_EXTS = {"wav", "mp3", "m4a", "webm", "ogg", "opus", "flac", "mp4"}
SUPPORTED_TRANSCRIPT_EXTS = {"txt", "vtt"}

# Cache the Whisper model across requests.
_whisper_model = None


def _ffmpeg_present() -> bool:
    return shutil.which("ffmpeg") is not None


def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _get_local_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("large-v3", device="auto",
                                      compute_type="auto")
    return _whisper_model


def transcribe_local(audio_path: Path) -> tuple[str, float]:
    model = _get_local_whisper()
    t0 = time.monotonic()
    seg_iter, _info = model.transcribe(
        str(audio_path), language="it", vad_filter=True,
    )
    text = "\n".join(s.text.strip() for s in seg_iter)
    return text, time.monotonic() - t0


def transcribe_openai(audio_path: Path) -> tuple[str, float]:
    import openai
    key = env("OPENAI_API_KEY", required=True)
    client = openai.OpenAI(api_key=key)
    t0 = time.monotonic()
    with audio_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="it",
        )
    return resp.text, time.monotonic() - t0


def parse_transcript(name: str, raw_bytes: bytes) -> str:
    if name.lower().endswith(".vtt"):
        with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False,
                                         dir=tempfile.gettempdir()) as tf:
            tf.write(raw_bytes)
            tf_path = Path(tf.name)
        try:
            import webvtt
            lines = []
            for cap in webvtt.read(str(tf_path)):
                text = cap.text.replace("\n", " ").strip()
                speaker = None
                if ": " in text and len(text.split(": ", 1)[0]) < 40:
                    speaker, text = text.split(": ", 1)
                lines.append(f"{speaker}: {text}" if speaker else text)
            return "\n".join(lines)
        finally:
            tf_path.unlink(missing_ok=True)
    return raw_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------- routes ---
@app.route("/")
def index():
    capabilities = {
        "anthropic_ready": bool(env("ANTHROPIC_API_KEY")),
        "openai_ready": bool(env("OPENAI_API_KEY")),
        "ollama_ready": ollama_reachable(),
        "ollama_model": OLLAMA_MODEL,
        "ffmpeg": _ffmpeg_present(),
        "faster_whisper": _faster_whisper_available(),
        "rubric_version": RUBRIC_VERSION,
        "models": {"default": SCORING_MODEL, "ceiling": CEILING_MODEL,
                   "ollama": OLLAMA_MODEL},
        "default_backend": DEFAULT_BACKEND,
        "expected_dist": EXPECTED_DIST,
        "supported_audio": sorted(SUPPORTED_AUDIO_EXTS),
        "supported_transcript": sorted(SUPPORTED_TRANSCRIPT_EXTS),
    }
    return render_template("index.html", caps=capabilities)


@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/score", methods=["POST"])
def score():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    # Scoring backend (anthropic|ollama) — passed in form.
    score_backend = (request.form.get("score_backend")
                     or DEFAULT_BACKEND).strip()
    if score_backend not in (BACKEND_ANTHROPIC, BACKEND_OLLAMA):
        return jsonify({"error": f"invalid score_backend: {score_backend}"}), 400
    model = request.form.get("model")
    if not model:
        model = SCORING_MODEL if score_backend == BACKEND_ANTHROPIC else OLLAMA_MODEL
    if score_backend == BACKEND_ANTHROPIC and model not in (
            SCORING_MODEL, CEILING_MODEL):
        return jsonify({"error": f"invalid anthropic model: {model}"}), 400
    # Transcription backend (auto|local|openai), only for audio uploads.
    backend = request.form.get("backend") or "auto"
    raw = f.read()

    timings = {"transcribe_sec": 0.0, "score_sec": 0.0}
    transcription_source = None

    # ---- Get transcript
    t0_total = time.monotonic()
    try:
        if ext in SUPPORTED_TRANSCRIPT_EXTS:
            transcript = parse_transcript(f.filename, raw)
            transcription_source = f"transcript-upload-{ext}"
        elif ext in SUPPORTED_AUDIO_EXTS:
            tmp_dir = Path(tempfile.gettempdir()) / "loi_app"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_audio = tmp_dir / f"upload_{int(time.time())}.{ext}"
            tmp_audio.write_bytes(raw)
            try:
                use_local = (
                    backend == "local"
                    or (backend == "auto" and _faster_whisper_available()
                        and _ffmpeg_present())
                )
                if use_local:
                    transcript, dt = transcribe_local(tmp_audio)
                    transcription_source = "local-whisper-large-v3"
                else:
                    transcript, dt = transcribe_openai(tmp_audio)
                    transcription_source = "openai-whisper-1"
                timings["transcribe_sec"] = dt
            finally:
                tmp_audio.unlink(missing_ok=True)
        else:
            return jsonify({"error": f"unsupported file type: .{ext}"}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"transcription failed: {e}",
                        "stage": "transcribe"}), 500

    if not transcript.strip():
        return jsonify({"error": "transcript is empty"}), 400

    # ---- Score
    try:
        call_id = f"upload_{int(time.time())}"
        res = score_call(transcript, call_id=call_id, source="upload",
                         model=model, backend=score_backend)
        write_score(res)
        timings["score_sec"] = res.latency_sec
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"scoring failed: {e}",
                        "stage": "score"}), 500

    timings["total_sec"] = time.monotonic() - t0_total

    return jsonify({
        "ok": True,
        "filename": f.filename,
        "transcription_source": transcription_source,
        "transcript_chars": len(transcript),
        "transcript": transcript,
        "model": res.model,
        "rubric_version": res.rubric_version,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "cost_usd": res.cost_usd,
        "timings": timings,
        "score": res.parsed,
        "raw": res.raw,
    })


# ---------------------------------------------------------------- audio ---
_AUDIO_EXTS = (".webm", ".m4a", ".mp3", ".wav", ".opus", ".ogg", ".flac", ".mp4")


def _audio_stats() -> dict:
    """Scan every plausible audio root, count + size + mtime range."""
    # Both the configured external dir and the in-tree default; whichever exists.
    roots = []
    for r in (AUDIO_DIR, DATA / "audio"):
        if r.exists() and r not in roots:
            roots.append(r)
    if not roots:
        return {"empty": True, "roots": [str(AUDIO_DIR), str(DATA / "audio")]}

    by_source: dict[str, dict] = {
        "genesys": {"count": 0, "bytes": 0},
        "zoom":    {"count": 0, "bytes": 0},
        "other":   {"count": 0, "bytes": 0},
    }
    mtimes: list[float] = []
    used_roots: list[str] = []
    total_count = 0
    total_bytes = 0

    for root in roots:
        used_roots.append(str(root))
        for f in root.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in _AUDIO_EXTS:
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            # Classify: Genesys = .webm; Zoom = anywhere under a "zoom" subdir
            # or .m4a; else other.
            rel = f.relative_to(root)
            parts_lc = [p.lower() for p in rel.parts]
            if f.suffix.lower() == ".webm":
                src = "genesys"
            elif "zoom" in parts_lc or f.suffix.lower() == ".m4a":
                src = "zoom"
            else:
                src = "other"
            by_source[src]["count"] += 1
            by_source[src]["bytes"] += st.st_size
            mtimes.append(st.st_mtime)
            total_count += 1
            total_bytes += st.st_size

    def _fmt(ts: float) -> str:
        return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

    return {
        "empty": total_count == 0,
        "roots": used_roots,
        "total_count": total_count,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / 1024 / 1024 / 1024,
        "by_source": by_source,
        "mtime_min": _fmt(min(mtimes)) if mtimes else "—",
        "mtime_max": _fmt(max(mtimes)) if mtimes else "—",
    }


# ---------------------------------------------------------------- data ----
def _model_family(model: str | None) -> str:
    """Bucket a model string into {claude, ollama, other} for the data view."""
    if not model:
        return "other"
    m = model.lower()
    if m.startswith("claude"):
        return "claude"
    if m.startswith("ollama") or m.startswith("qwen") or m.startswith("llama"):
        return "ollama"
    return "other"


def _load_scores_by_key() -> dict[tuple[str, str], dict[str, dict]]:
    """{(source, call_id): {family: score_dict, ...}} from outputs/scores/*.json.

    Multiple scores per call coexist (Claude + Qwen + Opus), bucketed by
    family. If two files share the same family (e.g. Sonnet + Opus both
    'claude'), the later-mtime wins.
    """
    out: dict[tuple[str, str], dict[str, dict]] = {}
    if not OUT_SCORES.exists():
        return out
    files = sorted(OUT_SCORES.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        src, cid = d.get("source"), d.get("call_id")
        if not src or not cid:
            continue
        s = d.get("score") or {}
        family = _model_family(d.get("model"))
        out.setdefault((src, cid), {})[family] = {
            "outcome": s.get("outcome"),
            "score": s.get("score"),
            "confidence": s.get("confidence"),
            "model": d.get("model"),
            "cost_usd": d.get("cost_usd"),
        }
    return out


def _load_calls(limit: int | None = None) -> list[dict]:
    if not CALLS_CSV.exists():
        return []
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    scores = _load_scores_by_key()
    for r in rows:
        r["has_audio"] = bool((r.get("audio_file") or "").strip())
        r["has_transcript"] = bool((r.get("transcript_file") or "").strip())
        by_family = scores.get((r["source"], r["call_id"]), {})
        claude = by_family.get("claude") or {}
        ollama = by_family.get("ollama") or {}
        # The "primary" model_outcome shown in the table — prefer Claude as
        # the gold standard, fall back to Ollama if that's all we have.
        primary = claude or ollama
        r["has_score"] = bool(primary)
        r["model_outcome"] = primary.get("outcome") or ""
        r["model_score"] = primary.get("score")
        r["model_confidence"] = primary.get("confidence") or ""
        r["model_used"] = primary.get("model") or ""
        # Per-family fields for the comparison columns.
        r["claude_outcome"] = claude.get("outcome") or ""
        r["claude_score"] = claude.get("score")
        r["ollama_outcome"] = ollama.get("outcome") or ""
        r["ollama_score"] = ollama.get("score")
        r["ollama_model"] = ollama.get("model") or ""
        if claude and ollama:
            r["agree"] = (claude.get("outcome") == ollama.get("outcome"))
        else:
            r["agree"] = None
        r["has_label"] = bool((r.get("rep_loi") or "").strip())
        r["has_conversion"] = bool((r.get("conversion_outcome") or "").strip())
    return rows


def _compute_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"empty": True}
    by_source = Counter(r["source"] for r in rows)
    by_type = Counter((r["source"], r.get("call_type") or "—") for r in rows)
    dates = [r.get("call_date") or "" for r in rows]
    dates_nonempty = [d for d in dates if d]
    daily: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        d = r.get("call_date") or ""
        if d:
            daily[d][r["source"]] += 1
    sorted_days = sorted(daily.keys())
    # Last 60 days max for the chart
    chart_days = sorted_days[-60:]
    series = [{"date": d,
               "genesys": daily[d].get("genesys", 0),
               "zoom": daily[d].get("zoom", 0)}
              for d in chart_days]
    chart_max = max((s["genesys"] + s["zoom"] for s in series), default=1) or 1

    # By rep_id top 10
    rep_ctr = Counter(r.get("rep_id") or "—" for r in rows)
    top_reps = rep_ctr.most_common(10)

    # Per-source call_date range for AUDIO-bearing rows only — answers
    # "what date range have I actually fetched audio for?"
    audio_dates: dict[str, list[str]] = {"genesys": [], "zoom": []}
    for r in rows:
        if r["has_audio"] and (r.get("call_date") or "") and r["source"] in audio_dates:
            audio_dates[r["source"]].append(r["call_date"])
    audio_window = {}
    for s, ds in audio_dates.items():
        if ds:
            audio_window[s] = {"min": min(ds), "max": max(ds), "n": len(ds)}
        else:
            audio_window[s] = {"min": "—", "max": "—", "n": 0}

    # Outcome distribution for scored calls
    scored_rows = [r for r in rows if r["has_score"]]
    outcome_ctr = Counter(r["model_outcome"] for r in scored_rows if r["model_outcome"])
    outcome_order = ["Freddo", "Tiepido", "Calduccio", "Caldo",
                     "Non interessato", "Non idoneo"]
    outcome_dist = []
    n_scored = len(scored_rows)
    for o in outcome_order:
        c = outcome_ctr.get(o, 0)
        outcome_dist.append({
            "outcome": o,
            "count": c,
            "pct": (100 * c / n_scored) if n_scored else 0,
        })

    # Claude vs Ollama agreement (only over calls scored by BOTH)
    both = [r for r in rows if r["agree"] is not None]
    n_agree = sum(1 for r in both if r["agree"])
    agreement = {
        "n_both": len(both),
        "n_agree": n_agree,
        "pct": (100 * n_agree / len(both)) if both else None,
        "n_claude_only": sum(1 for r in rows if r["claude_outcome"]
                             and not r["ollama_outcome"]),
        "n_ollama_only": sum(1 for r in rows if r["ollama_outcome"]
                             and not r["claude_outcome"]),
    }

    # Conversion rate by LOI band — real contracts as the arbiter. Computed
    # for the human rep LOI and for the model (Claude), over calls that have a
    # known conversion_outcome (signed/not_signed). This is the headline test:
    # does either LOI actually predict who signs?
    BANDS_IN = ["Freddo", "Tiepido", "Calduccio", "Caldo"]

    def _conv_series(band_key: str, bands: list[str]) -> list[dict]:
        out = []
        for b in bands:
            grp = [r for r in rows
                   if (r.get(band_key) or "") == b and r.get("conversion_outcome")]
            n = len(grp)
            signed = sum(1 for r in grp if r["conversion_outcome"] == "signed")
            out.append({"band": b, "n": n, "signed": signed,
                        "rate": (100 * signed / n) if n else None})
        return out

    n_conv_known = sum(1 for r in rows if r.get("conversion_outcome"))
    n_signed = sum(1 for r in rows if r.get("conversion_outcome") == "signed")
    conversion = {
        "n_known": n_conv_known,
        "n_signed": n_signed,
        "base_rate": (100 * n_signed / n_conv_known) if n_conv_known else None,
        "by_rep_loi": _conv_series("rep_loi", BANDS_IN),
        "by_model": _conv_series("model_outcome",
                                 BANDS_IN + ["Non interessato", "Non idoneo"]),
    }

    return {
        "empty": False,
        "total": len(rows),
        "conversion": conversion,
        "by_source": dict(by_source),
        "by_type": [{"source": s, "type": t, "count": c}
                    for (s, t), c in sorted(by_type.items(),
                                            key=lambda x: -x[1])],
        "date_min": min(dates_nonempty) if dates_nonempty else "—",
        "date_max": max(dates_nonempty) if dates_nonempty else "—",
        "scored": sum(1 for r in rows if r["has_score"]),
        "with_audio": sum(1 for r in rows if r["has_audio"]),
        "with_transcript": sum(1 for r in rows if r["has_transcript"]),
        "with_label": sum(1 for r in rows if r["has_label"]),
        "with_conversion": sum(1 for r in rows if r["has_conversion"]),
        "series": series,
        "chart_max": chart_max,
        "top_reps": [{"rep_id": r, "count": c} for r, c in top_reps],
        "audio_window": audio_window,
        "outcome_dist": outcome_dist,
        "n_scored": n_scored,
        "agreement": agreement,
    }


def _apply_filters(rows: list[dict], q: dict) -> list[dict]:
    src = (q.get("source") or "").strip()
    ct = (q.get("call_type") or "").strip()
    date_from = (q.get("date_from") or "").strip()
    date_to = (q.get("date_to") or "").strip()
    has_score = (q.get("has_score") or "").strip()  # "yes"|"no"|""
    outcome = (q.get("outcome") or "").strip()
    agree_filter = (q.get("agree") or "").strip()  # agree|disagree|claude-only|ollama-only
    conv_filter = (q.get("conversion") or "").strip()  # signed|not_signed
    text = (q.get("q") or "").strip().lower()
    out = []
    for r in rows:
        if src and r["source"] != src:
            continue
        if ct and (r.get("call_type") or "") != ct:
            continue
        if date_from and (r.get("call_date") or "") < date_from:
            continue
        if date_to and (r.get("call_date") or "") > date_to:
            continue
        if has_score == "yes" and not r["has_score"]:
            continue
        if has_score == "no" and r["has_score"]:
            continue
        if outcome and r.get("model_outcome") != outcome:
            continue
        if agree_filter == "agree" and r.get("agree") is not True:
            continue
        if agree_filter == "disagree" and r.get("agree") is not False:
            continue
        if agree_filter == "claude-only" and not (
                r.get("claude_outcome") and not r.get("ollama_outcome")):
            continue
        if agree_filter == "ollama-only" and not (
                r.get("ollama_outcome") and not r.get("claude_outcome")):
            continue
        if conv_filter and (r.get("conversion_outcome") or "") != conv_filter:
            continue
        if text and text not in (r.get("call_id", "").lower() +
                                 r.get("rep_id", "").lower() +
                                 r.get("agent_name", "").lower() +
                                 r.get("topic", "").lower()):
            continue
        out.append(r)
    return out


@app.route("/data")
def data_page():
    rows = _load_calls()
    stats = _compute_stats(rows)
    audio = _audio_stats()
    return render_template("data.html", stats=stats, audio=audio)


@app.route("/data.json")
def data_json():
    rows = _load_calls()
    filtered = _apply_filters(rows, request.args)
    page = max(int(request.args.get("page", "1")), 1)
    size = min(max(int(request.args.get("size", "50")), 1), 500)
    start_i = (page - 1) * size
    page_rows = filtered[start_i:start_i + size]
    # Trim heavy fields for the table response
    keep_cols = ["source", "call_id", "call_date", "duration_sec", "call_type",
                 "rep_id", "agent_name", "host_email", "topic",
                 "has_audio", "has_transcript", "has_score",
                 "model_outcome", "model_score", "model_confidence",
                 "claude_outcome", "claude_score",
                 "ollama_outcome", "ollama_score", "ollama_model", "agree",
                 "rep_loi", "conversion_outcome"]
    slim = [{k: r.get(k) for k in keep_cols} for r in page_rows]
    return jsonify({"total": len(filtered), "page": page, "size": size,
                    "rows": slim})


# ---------------------------------------------------------------- fetch ----
def _build_ingest_cmd(source: str, start: str, end: str,
                      max_count: int) -> list[str]:
    """Build the subprocess argv for ingest_genesys or ingest_zoom."""
    py = sys.executable
    if source == "genesys":
        # ingest_genesys wants ISO 8601 with Z suffix.
        start_iso = f"{start}T00:00:00.000Z"
        end_iso = f"{end}T23:59:59.999Z"
        cmd = [py, "-u", "-m", "src.ingest_genesys",
               "--start", start_iso, "--end", end_iso]
        if max_count:
            cmd += ["--max-calls", str(max_count)]
    elif source == "zoom":
        cmd = [py, "-u", "-m", "src.ingest_zoom",
               "--start", start, "--end", end]
        if max_count:
            cmd += ["--max-meetings", str(max_count)]
    else:
        raise ValueError(f"unknown source: {source}")
    return cmd


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.route("/fetch_stream", methods=["POST"])
def fetch_stream():
    """Run ingest_genesys and/or ingest_zoom and stream their output via SSE."""
    body = request.get_json(silent=True) or {}
    sources = body.get("sources") or []
    start = body.get("start")
    end = body.get("end")
    max_count = int(body.get("max_count") or 0)

    if not start or not end:
        return jsonify({"error": "start and end (YYYY-MM-DD) required"}), 400
    if not sources or not all(s in ("genesys", "zoom") for s in sources):
        return jsonify({"error": "sources must be a list of 'genesys'/'zoom'"}), 400
    # Credential pre-checks
    missing = []
    if "genesys" in sources and not env("GENESYS_CLIENT_ID"):
        missing.append("GENESYS_CLIENT_ID")
    if "zoom" in sources and not env("ZOOM_ACCOUNT_ID"):
        missing.append("ZOOM_ACCOUNT_ID")
    if missing:
        return jsonify({"error": f"missing env: {', '.join(missing)}"}), 400

    def runner():
        yield _sse({"event": "start", "sources": sources,
                    "start": start, "end": end,
                    "max_count": max_count or None})
        for src in sources:
            yield _sse({"event": "source_start", "source": src})
            try:
                cmd = _build_ingest_cmd(src, start, end, max_count)
            except ValueError as e:
                yield _sse({"event": "error", "source": src, "error": str(e)})
                continue
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(ROOT),
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    if line:
                        yield _sse({"event": "log", "source": src, "line": line})
                proc.wait()
            except Exception as e:  # noqa: BLE001
                proc.kill()
                yield _sse({"event": "error", "source": src, "error": str(e)})
                continue
            yield _sse({"event": "source_done", "source": src,
                        "exit_code": proc.returncode})

        # All sources finished — consolidate into calls.csv.
        try:
            r = merge_calls_csv()
            yield _sse({"event": "merge", "total": r.total, "added": r.added,
                        "label_updates": r.label_updates,
                        "skipped_existing": r.skipped_seen})
        except Exception as e:  # noqa: BLE001
            yield _sse({"event": "error", "source": "merge", "error": str(e)})

        yield _sse({"event": "done"})

    return Response(
        stream_with_context(runner()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    # Local-only by default; bind to 0.0.0.0 by setting LOI_BIND=0.0.0.0.
    host = env("LOI_BIND", default="127.0.0.1")
    port = int(env("LOI_PORT", default="8501"))
    app.run(host=host, port=port, debug=False, threaded=True)
