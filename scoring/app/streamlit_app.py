"""LOI Audio Scoring — Streamlit demo.

Drop an Italian sales-call audio file, get a Level of Interest score with
the full decision-tree path and verbatim evidence quotes.

Run:
    streamlit run app/streamlit_app.py

Transcription has two modes:
- Local Whisper (faster-whisper large-v3) — free, needs ffmpeg on PATH.
- OpenAI Whisper API — needs OPENAI_API_KEY; ~$0.006/min of audio.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Make src.* importable when running `streamlit run app/streamlit_app.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import rubric                                  # noqa: E402
from src.config import (                                # noqa: E402
    CEILING_MODEL, EXPECTED_DIST, RUBRIC_VERSION, SCORING_MODEL, env,
)
from src.score_llm import safe_filename, score_call, write_score  # noqa: E402

# ------------------------------------------------------------------ helpers ---
OUTCOME_COLORS = {
    "Caldo":           ("#0a7d2e", "#dcf5e2"),  # text, bg
    "Calduccio":       ("#9a6a00", "#fff3cc"),
    "Tiepido":         ("#a35100", "#ffe1c2"),
    "Freddo":          ("#9b1f1f", "#fcdada"),
    "Non interessato": ("#3e3e3e", "#e8e8e8"),
    "Non idoneo":      ("#2a2a2a", "#cfcfcf"),
}
SUPPORTED_AUDIO_EXTS = ("wav", "mp3", "m4a", "webm", "ogg", "opus", "flac", "mp4")
SUPPORTED_TRANSCRIPT_EXTS = ("txt", "vtt")


@st.cache_resource(show_spinner=False)
def _ffmpeg_present() -> bool:
    return shutil.which("ffmpeg") is not None


@st.cache_resource(show_spinner=False)
def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_resource(show_spinner="Loading Whisper large-v3 (first run only)…")
def _get_local_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("large-v3", device="auto", compute_type="auto")


def transcribe_local(audio_path: Path) -> tuple[str, list[dict], float]:
    """Returns (plain_text, segments, wall_seconds)."""
    model = _get_local_whisper()
    t0 = time.monotonic()
    seg_iter, _info = model.transcribe(
        str(audio_path), language="it", vad_filter=True, word_timestamps=False,
    )
    segments: list[dict] = []
    lines: list[str] = []
    for seg in seg_iter:
        text = seg.text.strip()
        segments.append({"start": seg.start, "end": seg.end, "text": text})
        lines.append(text)
    return "\n".join(lines), segments, time.monotonic() - t0


def parse_uploaded_transcript(name: str, raw_bytes: bytes) -> tuple[str, list[dict]]:
    """Return (plain_text, segments) from a .txt or .vtt upload."""
    if name.lower().endswith(".vtt"):
        # webvtt.read needs a file path
        with tempfile.NamedTemporaryFile(
                suffix=".vtt", delete=False, dir=tempfile.gettempdir()) as tf:
            tf.write(raw_bytes)
            tf_path = Path(tf.name)
        try:
            import webvtt
            segments: list[dict] = []
            lines: list[str] = []
            for cap in webvtt.read(str(tf_path)):
                text = cap.text.replace("\n", " ").strip()
                speaker = None
                if ": " in text and len(text.split(": ", 1)[0]) < 40:
                    speaker, text = text.split(": ", 1)
                segments.append({"start": cap.start_in_seconds,
                                 "end": cap.end_in_seconds,
                                 "speaker": speaker, "text": text})
                lines.append(f"{speaker}: {text}" if speaker else text)
            return "\n".join(lines), segments
        finally:
            tf_path.unlink(missing_ok=True)
    # Plain .txt — keep as-is. Genesys-style [Cliente]/[Agente] lines pass through.
    text = raw_bytes.decode("utf-8", errors="replace")
    segments = [{"start": None, "end": None, "speaker": None, "text": ln.strip()}
                for ln in text.splitlines() if ln.strip()]
    return text, segments


def transcribe_openai(audio_path: Path) -> tuple[str, list[dict], float]:
    import openai
    key = env("OPENAI_API_KEY", required=True)
    client = openai.OpenAI(api_key=key)
    t0 = time.monotonic()
    with audio_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="it",
            response_format="verbose_json",
        )
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in getattr(resp, "segments", []) or []]
    text = resp.text if hasattr(resp, "text") else str(resp)
    return text, segs, time.monotonic() - t0


# ------------------------------------------------------------------ UI -------
st.set_page_config(page_title="LOI Audio Scoring", layout="wide",
                   initial_sidebar_state="expanded")

# Sidebar — config
with st.sidebar:
    st.title("LOI Scoring")
    st.caption(f"Rubric `{RUBRIC_VERSION}`")
    st.divider()

    model = st.selectbox(
        "Scoring model",
        options=[SCORING_MODEL, CEILING_MODEL],
        index=0,
        help="Sonnet for routine scoring; Opus for the ceiling check.",
    )

    st.caption("Transcription only applies to audio uploads.")
    local_ok = _faster_whisper_available() and _ffmpeg_present()
    default_mode = "Local Whisper" if local_ok else "OpenAI Whisper API"
    transcription_mode = st.radio(
        "Transcription (audio mode only)",
        options=["Local Whisper", "OpenAI Whisper API"],
        index=0 if default_mode == "Local Whisper" else 1,
        help="Local is free but needs ffmpeg + faster-whisper. API is "
             "~$0.006/min and needs OPENAI_API_KEY in .env.",
    )

    if transcription_mode == "Local Whisper":
        if not _faster_whisper_available():
            st.warning("`faster-whisper` not installed. Run "
                       "`pip install faster-whisper`.")
        if not _ffmpeg_present():
            st.warning("ffmpeg not on PATH. On Windows: "
                       "`winget install Gyan.FFmpeg` then restart the shell.")
    else:
        if not env("OPENAI_API_KEY"):
            st.warning("`OPENAI_API_KEY` not set in `.env`.")

    st.divider()
    st.caption("Anthropic key is read from `.env` — never typed here.")


# Main area
st.title("Sales-call LOI from audio")
st.caption("Italian sales-call audio (or transcript) → Level of Interest score.")

input_mode = st.radio(
    "Input type",
    options=["Audio file", "Transcript file (.txt / .vtt)"],
    horizontal=True,
    help="Pick 'Transcript file' to skip Whisper entirely — much faster and "
         "fully free.",
)
is_audio = input_mode.startswith("Audio")

if is_audio:
    uploaded = st.file_uploader(
        "Drop an audio file",
        type=list(SUPPORTED_AUDIO_EXTS),
        accept_multiple_files=False,
        help=f"Supported: {', '.join(SUPPORTED_AUDIO_EXTS)}. Italian audio expected.",
    )
else:
    uploaded = st.file_uploader(
        "Drop a transcript file",
        type=list(SUPPORTED_TRANSCRIPT_EXTS),
        accept_multiple_files=False,
        help="`.vtt` (Zoom export) or plain `.txt` (e.g. Genesys `[Cliente]/[Agente]` style).",
    )

if uploaded is None:
    with st.expander("How this works"):
        st.markdown("""
        1. You upload either an **audio file** (a sales call, Italian) or a
           **transcript file** (`.txt` or `.vtt`).
        2. If audio, the app transcribes it (local Whisper or OpenAI API).
           If transcript, it skips this step entirely.
        3. The configured model scores the transcript with the demo rubric.
        4. The app shows the structured output for review.
        """)
    st.stop()

# Preview + Score button row
col_preview, col_action = st.columns([3, 1])
with col_preview:
    if is_audio:
        st.audio(uploaded, format=f"audio/{uploaded.name.rsplit('.', 1)[-1]}")
    else:
        preview = uploaded.getvalue().decode("utf-8", errors="replace")
        st.text_area(
            "Transcript preview",
            preview[:1500] + ("\n…(truncated)" if len(preview) > 1500 else ""),
            height=180, label_visibility="collapsed",
        )
with col_action:
    st.metric("File", uploaded.name, f"{uploaded.size / 1024:.0f} KB")
    go = st.button("Score this call", type="primary", use_container_width=True)

if not go:
    st.stop()

# ---- Get transcript (transcribe audio OR parse uploaded transcript)
trans_box = st.status(
    "Transcribing…" if is_audio else "Loading transcript…", expanded=True,
)
try:
    if is_audio:
        tmp_dir = Path(tempfile.gettempdir()) / "loi_app"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        suffix = "." + uploaded.name.rsplit(".", 1)[-1]
        tmp_audio = tmp_dir / f"upload_{int(time.time())}{suffix}"
        tmp_audio.write_bytes(uploaded.getvalue())
        if transcription_mode == "Local Whisper":
            transcript, segments, t_trans = transcribe_local(tmp_audio)
            trans_source = "local-whisper-large-v3"
        else:
            transcript, segments, t_trans = transcribe_openai(tmp_audio)
            trans_source = "openai-whisper-1"
        trans_box.update(
            label=f"Transcribed in {t_trans:.1f}s via {trans_source}",
            state="complete",
        )
    else:
        transcript, segments = parse_uploaded_transcript(
            uploaded.name, uploaded.getvalue(),
        )
        t_trans = 0.0
        trans_source = "transcript-upload-" + uploaded.name.rsplit(".", 1)[-1]
        trans_box.update(
            label=f"Loaded transcript ({len(transcript):,} chars, "
                  f"{len(segments)} segments)",
            state="complete",
        )
except Exception as e:  # noqa: BLE001
    trans_box.update(label=f"Failed: {e}", state="error")
    st.exception(e)
    st.stop()

if not transcript.strip():
    st.error("Transcript is empty — nothing to score.")
    st.stop()

# ---- Score
score_box = st.status(f"Scoring with `{model}`…", expanded=False)
try:
    res = score_call(transcript, call_id=f"upload_{int(time.time())}",
                     source="upload", model=model)
    write_score(res)  # archives the result to outputs/scores/
    score_box.update(label=f"Scored in {res.latency_sec:.1f}s — "
                     f"${res.cost_usd:.4f}", state="complete")
except Exception as e:  # noqa: BLE001
    score_box.update(label=f"Scoring failed: {e}", state="error")
    st.exception(e)
    st.stop()

parsed = res.parsed

# ---- Render result -----------------------------------------------------------
st.divider()

outcome = parsed.get("outcome", "?")
score_val = parsed.get("score")
fg, bg = OUTCOME_COLORS.get(outcome, ("#222", "#eee"))

# Big outcome banner
st.markdown(
    f"""
<div style="background:{bg};border-radius:10px;padding:18px 22px;
            display:flex;align-items:center;justify-content:space-between;">
  <div style="font-size:32px;font-weight:700;color:{fg};">{outcome}</div>
  <div style="font-size:22px;color:{fg};">
    Score: <b>{score_val if score_val is not None else '—'}</b>
    &nbsp;·&nbsp; Confidence: <b>{parsed.get('confidence', '—')}</b>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Tree path table
st.subheader("Decision tree walk")
tp = parsed.get("tree_path") or {}
q_labels = {
    "q1": ("Q1 — Vuole il nostro prodotto?", tp.get("q1")),
    "q2": ("Q2 — Nelle condizioni di procedere?", tp.get("q2")),
    "q3": ("Q3 — In che fase è?", tp.get("q3")),
    "q4": ("Q4 — Reazione a prezzo / prossimi passi?", tp.get("q4")),
}
cols = st.columns(4)
for i, (k, (label, val)) in enumerate(q_labels.items()):
    with cols[i]:
        st.caption(label)
        if val in (None, "null"):
            st.markdown(":gray[—]")
        else:
            st.markdown(f"**{val}**")

c1, c2, c3 = st.columns(3)
c1.metric("Time signal", parsed.get("time_signal") or "—")
c2.metric("Behavior modifier", "applied" if parsed.get("behavior_modifier_applied")
          else "not applied")
c3.metric("Cost (USD)", f"${res.cost_usd:.4f}")

if parsed.get("behavior_modifier_applied"):
    st.info(f"**Why modifier:** {parsed.get('behavior_modifier_reason') or '—'}")

# Evidence
st.subheader("Evidence the model cited")
ev = parsed.get("evidence_quotes") or []
if ev:
    for q in ev:
        st.markdown(f"> {q}")
else:
    st.caption("(model returned no evidence quotes)")

# Run details
with st.expander("Run details"):
    st.json({
        "model": res.model,
        "rubric_version": res.rubric_version,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "cost_usd": res.cost_usd,
        "latency_sec": res.latency_sec,
        "transcription_mode": trans_source,
        "transcript_chars": len(transcript),
    })

with st.expander("Transcript"):
    st.text_area("transcript", transcript, height=240, label_visibility="collapsed")

with st.expander("Raw model JSON"):
    st.code(json.dumps(parsed, ensure_ascii=False, indent=2), language="json")

st.caption(
    f"Distribution target (per FINAL SURVEY): "
    f"Freddo {EXPECTED_DIST['Freddo']:.0%} · "
    f"Tiepido {EXPECTED_DIST['Tiepido']:.0%} · "
    f"Calduccio {EXPECTED_DIST['Calduccio']:.0%} · "
    f"Caldo {EXPECTED_DIST['Caldo']:.0%}"
)
