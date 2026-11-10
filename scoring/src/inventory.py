"""Phase 1 data audit -> outputs/reports/01_inventory.md.

Stdlib only (no pandas) so it runs before Phase 2 deps are installed.
Channel/sample-rate from the container needs ffmpeg (not installed); until then
we infer mono/multi from Genesys n_audio_tracks, which is authoritative for the
diarization decision.
"""
from __future__ import annotations

import csv
import datetime as dt
import statistics
from collections import Counter

from . import config

LABEL_COLS = ["rep_loi", "rep_reasoning", "conversion_outcome"]


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build() -> str:
    meta = _read_csv(config.GENESYS_METADATA_CSV)
    labels = _read_csv(config.LABELS_CSV)
    L: list[str] = []
    a = L.append

    a("# 01 — Data Inventory (Phase 1)\n")
    a(f"_Generated {dt.datetime.now(dt.timezone.utc).isoformat()} — "
      f"source: Genesys Cloud_\n")

    # --- audio ---
    a("## Audio\n")
    if not meta:
        a("No metadata rows yet — run `python -m src.ingest_genesys` first.\n")
        return "\n".join(L)

    durs = [int(r["duration_sec"] or 0) for r in meta]
    bytes_total = sum(int(r["audio_bytes"] or 0) for r in meta)
    on_disk = sum(1 for r in meta
                  if (config.AUDIO_DIR / r["audio_file"]).exists())
    empty = sum(1 for r in meta
                if (config.AUDIO_DIR / r["audio_file"]).exists()
                and (config.AUDIO_DIR / r["audio_file"]).stat().st_size == 0)
    tracks = Counter(r["n_audio_tracks"] for r in meta)
    dirs = Counter(r["direction"] for r in meta)

    a(f"- Calls with audio: **{len(meta)}** ({on_disk} files on disk, "
      f"{empty} empty/unreadable)")
    a(f"- Total audio: **{sum(durs)/3600:.2f} h** "
      f"({bytes_total/1e6:.1f} MB Opus/WEBM)")
    a(f"- Duration sec — min {min(durs)}, "
      f"median {int(statistics.median(durs))}, max {max(durs)}")
    a(f"- Direction: {dict(dirs)}")
    a(f"- Audio tracks per recording: {dict(tracks)}  "
      f"→ {'MONO (1 track) — Phase 3 diarization REQUIRED' if set(tracks)<= {'1'} else 'multi-track present — check stereo split'}\n")

    # --- recording coverage (the headline constraint) ---
    agents = Counter(r["agent_id"] for r in meta)
    queues = Counter(r["queue_id"] for r in meta)
    days = Counter((r["conversation_start"] or "")[:10] for r in meta)
    a("## Recording coverage — CRITICAL CONSTRAINT\n")
    a(f"- Distinct agents in sample: **{len(agents)}** "
      f"| distinct queues: **{len(queues)}** "
      f"(pilot took most-recent calls, dominated by the top recorded agent)")
    a(f"- Date span of pilot audio: {min(days)} → {max(days)}\n")
    a("Recording coverage depends on the source and sample.\n")

    # --- labels ---
    a("## Labels\n")
    a(f"- Label rows: **{len(labels)}**")
    for c in LABEL_COLS:
        filled = sum(1 for r in labels if (r.get(c) or "").strip())
        a(f"  - `{c}` populated: {filled}/{len(labels)}")
    a("")
    a("> **BLOCKER:** `rep_loi`, `rep_reasoning`, `conversion_outcome` are **not "
      "in Genesys**. `data/labels.csv` is a skeleton keyed by `call_id` "
      "(=conversationId) with `rep_id`/`call_date` prefilled. They must be "
      "joined from the survey/CRM system before Phases 4–5 can run.\n")

    # --- join ---
    a("## Audio ↔ Labels join (by call_id)\n")
    audio_ids = {r["conversation_id"] for r in meta}
    label_ids = {r["call_id"] for r in labels}
    both = audio_ids & label_ids
    with_loi = sum(1 for r in labels
                   if r["call_id"] in audio_ids and (r.get("rep_loi") or "").strip())
    a(f"- Calls with audio AND a label row: **{len(both)}**")
    a(f"- Calls with audio AND a non-empty rep_loi: **{with_loi}** "
      f"(target ≥ 50 for the MVP eval)\n")

    # --- checkpoint ---
    a("## Checkpoint questions for sales ops\n")
    a("1. **GDPR/consent**: signed legal basis to process these recordings? "
      "Are external API calls (Anthropic) permitted, or local-only?")
    a("2. **Labels**: where do `rep_loi` / `rep_reasoning` / `conversion_outcome` "
      "live, and what key joins to `conversationId`?")
    a("3. **Mono confirmed** → enable Phase 3 diarization (pyannote)?")
    a("4. **Scale**: approve pulling the full 50–200 set "
      "(12k+ voice calls / 14d are available)?")
    a("5. **Cloud sync**: `data/audio/` is PII inside a synced folder — "
      "exclude from sync or relocate?\n")
    return "\n".join(L)


def main() -> int:
    config.ensure_dirs()
    md = build()
    out = config.OUT_REPORTS / "01_inventory.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
