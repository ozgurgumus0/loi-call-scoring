"""Phase 1 data source: pull sales-call metadata from xCally Motion (the
contact-center platform that replaced Genesys).

Mirrors ingest_genesys.py / ingest_zoom.py. Produces:

  data/xcally_metadata.csv   # one row per recording (idempotent: re-runs skip
                             #   uniqueids already present)
  data/labels.csv            # appended with source=xcally skeleton rows

JOIN KEY: `uniqueid` (the Asterisk call unique id) is used as call_id.

IMPORTANT — NO TRANSCRIPTS YET. xCally's native transcription is OFF on this
instance (every recording, even months old, is transcribeStatus='NEW' with empty
fileText). So this ingester pulls METADATA only — it does NOT download audio and
there is no transcript to fetch. To make these calls scoreable we must first
solve transcription (enable xCally's post-call analytics, or transcribe the
audio ourselves). This ingester records transcribeStatus so we can re-run a
transcript pass once that's solved. No labeling, no spend.

SALES SCOPE: only the SELLER queues are LOI-relevant. Of those, only
`Seller_outbound` currently has recordings (~302); SELLER/Sales/vendite_company
are configured but empty. Booker/BOOKING queues (appointment-setting) are
deliberately EXCLUDED — different conversation type the rubric wasn't built for.

Usage:
  python -m src.ingest_xcally                 # default: the SELLER queues
  python -m src.ingest_xcally --queue Seller_outbound
  python -m src.ingest_xcally --all-queues    # every queue (incl. booking) — not for LOI
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from . import config
from .xcally_client import get

# Sales (LOI-relevant) queues. Only Seller_outbound has recordings today; the
# rest are kept so the ingester picks them up automatically if they start being
# used.
SELLER_QUEUES = [
    "Seller_outbound", "SELLER", "Sales", "vendite_company",
    "SELLER_QUOTATION_ONLY", "Seller_quotation_only_dialer",
]

# PII-safe metadata (NO fileText content / calleridnum / customerName).
FETCH_FIELDS = (
    "id,uniqueid,createdAt,queue,agentName,membername,UserId,duration,format,"
    "numberOfChannels,transcribeStatus,transcriptionStatus,sentiment,"
    "sentimentAnalysisStatus,userDisposition,userSecondDisposition,"
    "userThirdDisposition"
)

META_FIELDS = [
    "recording_id", "uniqueid", "createdAt", "queue", "agentName", "membername",
    "UserId", "duration", "numberOfChannels", "transcribeStatus",
    "transcriptionStatus", "sentiment", "userDisposition",
    "userSecondDisposition", "userThirdDisposition", "call_type", "source",
]

# Seller agents (xCally UserId) == the 21 sales reps matched 1:1 to the Zoom
# rep names (report 07). Their calls are the SALES calls regardless of queue
# (sellers dial direct, queue=None) — that's why queue-name fetch misses them.
SELLER_USER_IDS = {
    86, 119, 101, 115, 81, 116, 90, 80, 83, 118, 103, 117, 120, 178, 93, 102,
    92, 111, 121, 169, 109, 84,
}
# Real conversation threshold (xCally duration is in MILLISECONDS).
MIN_DURATION_MS = 120_000


def _classify_call_type(user_id, queue: str) -> str:
    """sales = seller agent; other = customer-care/account-mgmt; else booking."""
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        uid = None
    if uid in SELLER_USER_IDS:
        return "sales"
    q = (queue or "").upper()
    if "CARE" in q or "ACCOUNT" in q:
        return "other"
    return "booking"


def _sweep_all_real(page: int = 1000) -> list[dict]:
    """Full instance sweep (NO queue filter) → all REAL (>=2min) recordings,
    classified by call_type. Captures seller calls that have queue=None."""
    out: list[dict] = []
    offset = 0
    while True:
        _, body = get("voice/recordings", {
            "fields": FETCH_FIELDS, "limit": page, "offset": offset,
        })
        chunk = body.get("rows", []) if isinstance(body, dict) else (body or [])
        if not chunk:
            break
        for r in chunk:
            try:
                dur = int(float(r.get("duration") or 0))
            except (TypeError, ValueError):
                dur = 0
            if dur < MIN_DURATION_MS:
                continue
            out.append({
                "recording_id": r.get("id", ""),
                "uniqueid": str(r.get("uniqueid") or ""),
                "createdAt": r.get("createdAt", ""),
                "queue": r.get("queue", ""),
                "agentName": r.get("agentName", ""),
                "membername": r.get("membername", ""),
                "UserId": r.get("UserId", ""),
                "duration": r.get("duration", ""),
                "numberOfChannels": r.get("numberOfChannels", ""),
                "transcribeStatus": r.get("transcribeStatus", ""),
                "transcriptionStatus": r.get("transcriptionStatus", ""),
                "sentiment": r.get("sentiment", ""),
                "userDisposition": r.get("userDisposition", ""),
                "userSecondDisposition": r.get("userSecondDisposition", ""),
                "userThirdDisposition": r.get("userThirdDisposition", ""),
                "call_type": _classify_call_type(r.get("UserId"), r.get("queue")),
                "source": "xcally",
            })
        offset += len(chunk)
        total = body.get("count", 0) if isinstance(body, dict) else offset
        if offset >= total:
            break
    # dedupe by uniqueid (keep first)
    seen, deduped = set(), []
    for m in out:
        u = m["uniqueid"]
        if not u or u in seen:
            continue
        seen.add(u)
        deduped.append(m)
    return deduped
LABEL_FIELDS = [
    "call_id", "rep_loi", "rep_reasoning", "conversion_outcome",
    "rep_id", "call_date", "source",
]

XCALLY_META_CSV = config.DATA / "xcally_metadata.csv"


def _log(msg: str, fh) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def _load_done() -> set[str]:
    if not XCALLY_META_CSV.exists():
        return set()
    with XCALLY_META_CSV.open(newline="", encoding="utf-8") as f:
        return {r["uniqueid"] for r in csv.DictReader(f) if r.get("uniqueid")}


def _append_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def _fetch_queue(queue: str, page: int = 500) -> list[dict]:
    """Drain all recordings for one queue (paginated)."""
    rows: list[dict] = []
    offset = 0
    while True:
        _, body = get("voice/recordings", {
            "fields": FETCH_FIELDS, "queue": queue,
            "limit": page, "offset": offset, "sort": "createdAt",
        })
        chunk = body.get("rows", []) if isinstance(body, dict) else (body or [])
        rows.extend(chunk)
        offset += len(chunk)
        total = body.get("count", 0) if isinstance(body, dict) else len(rows)
        if not chunk or len(chunk) < page or offset >= total:
            break
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", help="fetch a single queue by name")
    ap.add_argument("--all-queues", action="store_true",
                    help="fetch EVERY queue (incl. booking) — not LOI-relevant")
    ap.add_argument("--sweep", action="store_true",
                    help="full instance sweep of ALL real (>=2min) recordings, "
                         "classified sales/booking/other by seller UserId; "
                         "OVERWRITES xcally_metadata.csv. Metadata only (no audio).")
    args = ap.parse_args(argv)

    config.ensure_dirs()

    if args.sweep:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = config.OUT_LOGS / f"ingest_xcally_sweep_{ts}.log"
        with log_path.open("w", encoding="utf-8") as fh:
            _log("SWEEP: all recordings, real (>=2min) only, metadata-only", fh)
            rows = _sweep_all_real()
            from collections import Counter
            ct = Counter(r["call_type"] for r in rows)
            _log(f"real recordings: {len(rows)}  call_type={dict(ct)}", fh)
            # overwrite (fresh full snapshot)
            with XCALLY_META_CSV.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=META_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            _log(f"metadata (OVERWRITTEN) -> {XCALLY_META_CSV} ({len(rows)} rows)", fh)
            _log("Audio NOT downloaded; transcripts NOT fetched. Run "
                 "build_calls_csv to merge into calls.csv.", fh)
        print(f"[sweep done] {len(rows)} real recordings -> {XCALLY_META_CSV}")
        return 0
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = config.OUT_LOGS / f"ingest_xcally_{ts}.log"
    done = _load_done()

    if args.queue:
        queues = [args.queue]
    elif args.all_queues:
        _, q = get("voice/queues", {"fields": "name", "limit": 200})
        queues = [r.get("name") for r in (q.get("rows", []) if isinstance(q, dict) else q)]
    else:
        queues = SELLER_QUEUES

    meta_rows: list[dict] = []
    label_rows: list[dict] = []
    fetched = skipped_dupe = no_transcript = 0

    with log_path.open("w", encoding="utf-8") as fh:
        _log(f"queues={queues} already_done={len(done)}", fh)
        try:
            for queue in queues:
                try:
                    recs = _fetch_queue(queue)
                except Exception as e:  # noqa: BLE001
                    _log(f"SKIP queue {queue}: {type(e).__name__}: {str(e)[:120]}", fh)
                    continue
                _log(f"queue {queue!r}: {len(recs)} recordings", fh)
                for r in recs:
                    uid = str(r.get("uniqueid") or "")
                    if not uid or uid in done:
                        skipped_dupe += 1
                        continue
                    done.add(uid)
                    if (r.get("transcribeStatus") or "NEW") != "done":
                        no_transcript += 1
                    meta_rows.append({
                        "recording_id": r.get("id", ""),
                        "uniqueid": uid,
                        "createdAt": r.get("createdAt", ""),
                        "queue": r.get("queue", ""),
                        "agentName": r.get("agentName", ""),
                        "membername": r.get("membername", ""),
                        "UserId": r.get("UserId", ""),
                        "duration": r.get("duration", ""),
                        "numberOfChannels": r.get("numberOfChannels", ""),
                        "transcribeStatus": r.get("transcribeStatus", ""),
                        "transcriptionStatus": r.get("transcriptionStatus", ""),
                        "sentiment": r.get("sentiment", ""),
                        "userDisposition": r.get("userDisposition", ""),
                        "userSecondDisposition": r.get("userSecondDisposition", ""),
                        "userThirdDisposition": r.get("userThirdDisposition", ""),
                        "source": "xcally",
                    })
                    label_rows.append({
                        "call_id": uid, "rep_loi": "", "rep_reasoning": "",
                        "conversion_outcome": "",
                        "rep_id": r.get("UserId", "") or r.get("agentName", ""),
                        "call_date": (r.get("createdAt") or "")[:10],
                        "source": "xcally",
                    })
                    fetched += 1
        finally:
            if meta_rows:
                _append_rows(XCALLY_META_CSV, META_FIELDS, meta_rows)
            if label_rows:
                _append_rows(config.LABELS_CSV, LABEL_FIELDS, label_rows)
            _log(f"DONE fetched={fetched} skipped_dupe={skipped_dupe} "
                 f"without_transcript={no_transcript}", fh)
            _log(f"metadata -> {XCALLY_META_CSV}", fh)
            _log("NOTE: transcripts NOT fetched (xCally transcription is OFF — "
                 "every call is transcribeStatus!=done). These calls are NOT "
                 "scoreable until transcription is solved (enable xCally analytics "
                 "or transcribe audio ourselves). Audio download not implemented.", fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
