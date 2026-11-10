"""Paths, models, rubric version, and .env loading.

Aligned to workflow v3 (docs/loi_audio_mvp_workflow_v3.md). Phase 1 ingest
landed two source corpora (Genesys + Zoom) into a unified `data/calls.csv`
keyed by (source, call_id). All later phases read from that file.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


# --- .env loader (stdlib) ---------------------------------------------------
# Defined + called BEFORE any module constants read os.environ, otherwise
# defaults baked in below would win over values placed in .env.
#
# SECRETS LIVE OUTSIDE SYNCED FOLDERS. The project tree is inside an cloud-synced
# folder, so a .env here uploads every credential to the cloud. Secrets now
# load from `~/loi-audio-data/.env` (same off-sync root as the audio dir).
# Search order (first value for a key wins, via setdefault):
#   1. $LOI_ENV_FILE  (explicit override)
#   2. ~/loi-audio-data/.env  (off-sync default)
#   3. ROOT/.env  (legacy in-project fallback — should be EMPTY of secrets)
ENV_FILE = Path(os.environ.get(
    "LOI_ENV_FILE", str(Path.home() / "loi-audio-data" / ".env")))


def _load_dotenv(*paths: Path) -> None:
    for path in paths:
        if not path or not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_dotenv(ENV_FILE, ROOT / ".env")


# Audio was moved out of cloud sync (PII). LOI_AUDIO_DIR overrides.
AUDIO_DIR = Path(os.environ.get(
    "LOI_AUDIO_DIR", str(Path.home() / "loi-audio-data" / "audio")
))

# Per-source transcript trees populated by Phase 1 ingest (and Phase 2).
TRANSCRIPTS_DIR = DATA / "transcripts"
ZOOM_VTT_DIR = TRANSCRIPTS_DIR / "zoom"
GENESYS_TRANSCRIPT_DIR = TRANSCRIPTS_DIR / "genesys"
ZOOM_EXTRAS_DIR = DATA / "zoom_extras"  # AI summary + next_steps JSONs

# Unified call-level table — one row per (source, call_id). Phase 1 output.
# Replaces the v2 plan of a single `labels.csv`; the file kept that name
# briefly but was renamed to `calls.csv` since labels are still missing.
CALLS_CSV = DATA / "calls.csv"
OMNIBUS_METADATA_CSV = DATA / "omnibus_metadata.csv"

# Per-source raw outputs from the ingest scripts (ingest_genesys, ingest_zoom).
# These feed the consolidated calls.csv via a separate merge step.
GENESYS_METADATA_CSV = DATA / "genesys_metadata.csv"
ZOOM_METADATA_CSV = DATA / "zoom_metadata.csv"
LABELS_CSV = DATA / "labels.csv"  # skeleton appended by both ingest scripts

OUTPUTS = ROOT / "outputs"
OUT_TRANSCRIPTS = OUTPUTS / "transcripts"
OUT_SCORES = OUTPUTS / "scores"
OUT_REPORTS = OUTPUTS / "reports"
OUT_LOGS = OUTPUTS / "logs"

# --- models / versions ------------------------------------------------------
WHISPER_MODEL = "large-v3"
WHISPER_LANG = "it"

# Anthropic. Sonnet for the full run, Opus for the ceiling subset.
SCORING_MODEL = "claude-sonnet-4-6"
CEILING_MODEL = "claude-opus-4-7"

# Ollama (local/GPU, free). Default = the PRODUCTION pick: the *prompted*
# Qwen3-8B (no fine-tune), served with the full rubric + CALIBRATION and ranked
# by closing_readiness (reproduces the validated full-corpus verdict). Pull it
# with `ollama pull qwen3:8b`. Override OLLAMA_BASE_URL / OLLAMA_MODEL via .env
# (e.g. point at the company GPU server, or `qwen-loi` for the fine-tune path).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

# Backend identifiers used across the codebase.
BACKEND_ANTHROPIC = "anthropic"
BACKEND_OLLAMA = "ollama"
DEFAULT_BACKEND = BACKEND_ANTHROPIC

# Bump on any rubric change. Embedded in every score file so we can compare
# rubric versions cleanly in the eval report.
RUBRIC_VERSION = "demo-v1"

# Uniform baseline for the demonstration rubric.
EXPECTED_DIST = {"Freddo": 0.25, "Tiepido": 0.25, "Calduccio": 0.25, "Caldo": 0.25}
OUTCOME_SCORES = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}

def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name} (set it in .env)")
    return val


def ensure_dirs() -> None:
    for d in (AUDIO_DIR, TRANSCRIPTS_DIR, ZOOM_VTT_DIR, GENESYS_TRANSCRIPT_DIR,
              OUT_TRANSCRIPTS, OUT_SCORES, OUT_REPORTS, OUT_LOGS):
        d.mkdir(parents=True, exist_ok=True)
