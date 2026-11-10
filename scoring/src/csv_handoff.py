"""Shared matching and enrichment logic for CSV scheduling exports."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA
from .fill_gsheet_loi import OUT_HEADER, SCORE_COLS, load_scores


CORPUS_START = pd.Timestamp("2026-04-21", tz="UTC")
MIN_VTT = 500


def _norm_meeting_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def enrich_csv(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (original rows, aligned LOI/status output block)."""
    source = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {"meetingUuid", "meetingId", "startTime", "deletedAt"}
    missing = required - set(source.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    rows = source[source["meetingUuid"].str.strip() != ""].copy()
    rows["_row_index"] = rows.index
    rows["start"] = pd.to_datetime(
        rows["startTime"], errors="coerce", utc=True, format="mixed"
    )
    rows["meetingId"] = _norm_meeting_id(rows["meetingId"])

    calls = pd.read_csv(DATA / "calls.csv", low_memory=False)
    zoom = calls[calls["source"] == "zoom"].copy()
    zoom["call_id"] = zoom["call_id"].astype(str)
    zoom = zoom.drop_duplicates("call_id", keep="last")
    zinfo = zoom.set_index("call_id")[["call_type", "transcript_bytes"]]
    corpus_ids = set(zinfo.index)

    metadata = pd.read_csv(DATA / "zoom_metadata.csv", low_memory=False)
    metadata["meeting_id"] = _norm_meeting_id(metadata["meeting_id"])
    metadata["meeting_uuid"] = metadata["meeting_uuid"].astype(str)
    metadata["_start"] = pd.to_datetime(
        metadata["start_time"], errors="coerce", utc=True, format="mixed"
    )
    by_mid = {
        mid: group[["meeting_uuid", "_start"]].copy()
        for mid, group in metadata.groupby("meeting_id")
    }

    def resolve(row):
        uuid = str(row["meetingUuid"]).strip()
        if uuid in corpus_ids:
            return uuid, "scheduled_uuid"
        candidates = by_mid.get(row["meetingId"])
        if candidates is None:
            return "", ""
        candidates = candidates[
            candidates["meeting_uuid"].astype(str).isin(corpus_ids)
        ].copy()
        if candidates.empty:
            return "", ""
        if pd.notna(row["start"]) and candidates["_start"].notna().any():
            candidates["_delta"] = (
                candidates["_start"] - row["start"]
            ).abs().dt.total_seconds()
            best = candidates.sort_values("_delta").iloc[0]
            if float(best["_delta"]) <= 18 * 3600:
                return str(best["meeting_uuid"]), "meeting_id_time"
            return "", ""
        return str(candidates.iloc[0]["meeting_uuid"]), "meeting_id"

    resolved = rows.apply(resolve, axis=1, result_type="expand")
    rows["matched_call_id"], rows["matched_via"] = resolved[0], resolved[1]

    scores = load_scores()
    scored_ids = set(scores.index)
    silent_file = DATA / "whisper_silent_calls.json"
    silent = (
        set(json.loads(silent_file.read_text(encoding="utf-8"))["call_ids"])
        if silent_file.exists() else set()
    )
    now = pd.Timestamp.now(tz="UTC")

    def status(row):
        call_id = row["matched_call_id"]
        if call_id:
            if call_id in scored_ids:
                return "scored"
            if call_id in silent:
                return "verified_silent_no_speech"
            transcript_bytes = pd.to_numeric(
                pd.Series([zinfo.at[call_id, "transcript_bytes"]]),
                errors="coerce",
            ).fillna(0).iloc[0]
            if float(transcript_bytes) < MIN_VTT:
                return "recorded_but_empty_or_no_transcript"
            return "recorded_awaiting_score"
        if str(row.get("deletedAt", "")).strip() not in ("", "NULL"):
            return "cancelled"
        if pd.isna(row["start"]):
            return "no_start_time"
        if row["start"] > now:
            return "scheduled_future"
        if row["start"] < CORPUS_START:
            return "pre_2026-04-21_recording_purged"
        return "no_recording_found"

    rows["loi_match_status"] = rows.apply(status, axis=1)
    enriched = rows.merge(
        scores[SCORE_COLS],
        how="left",
        left_on="matched_call_id",
        right_index=True,
    )
    block = enriched[OUT_HEADER].astype(object).where(pd.notna(enriched), "")

    # Score fields contain numbers as well as text. Explicit object dtype also
    # supports pandas versions that infer StringDtype for an empty-string frame.
    aligned = pd.DataFrame("", index=source.index, columns=OUT_HEADER, dtype=object)
    aligned.loc[rows["_row_index"], OUT_HEADER] = block.values
    return source, aligned
