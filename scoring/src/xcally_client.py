"""xCally Motion REST client + read-only API discovery.

xCally is the contact-center platform the company moved to (from Genesys); it
is the ongoing source of sales-call recordings + metadata. Basic auth.

.env keys (see .env.example): XCALLY_BASE_URL, and either XCALLY_TOKEN (the
base64 of "user:password") or XCALLY_USER + XCALLY_PASS (token derived).

    python -m src.xcally_client        # discovery: probe + print endpoint schemas

Discovery prints row KEYS only (never values) so we map the schema without
pulling customer PII into logs.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from urllib.parse import urlencode

from .config import env

BASE = (env("XCALLY_BASE_URL") or "").rstrip("/")


def _token() -> str:
    tok = env("XCALLY_TOKEN")
    if tok:
        return tok
    user, pwd = env("XCALLY_USER"), env("XCALLY_PASS")
    if not (user and pwd):
        raise RuntimeError(
            "Set XCALLY_TOKEN, or XCALLY_USER + XCALLY_PASS, in .env"
        )
    return base64.b64encode(f"{user}:{pwd}".encode()).decode()


def get(path: str, params: dict | None = None, timeout: float = 30.0):
    """GET {BASE}/{path}. Returns (http_status, parsed_json)."""
    url = f"{BASE}/{path.lstrip('/')}"
    if params:
        url += "?" + urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Basic {_token()}",
                 "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


# ----------------------------------------------------------------- discovery --
# xCally Motion list endpoints typically return {count, rows:[...]}. Probe the
# resources we care about for the ingest (recordings + call CDR + config).
_CANDIDATES = [
    ("users/8", None),                       # auth sanity (their example)
    ("users", {"limit": 1}),
    ("voice/recordings", {"limit": 2}),       # call audio + CDR — the key one
    ("voice/calls/reports", {"limit": 2}),    # historical CDR
    ("voice/queues", {"limit": 5}),
    ("voice/dispositions", {"limit": 10}),
    ("cdr", {"limit": 2}),
    ("voice/transcripts", {"limit": 2}),      # does xCally do ASR?
]


def _shape(obj) -> str:
    """One-line schema summary — KEYS only, no values (PII-safe)."""
    if isinstance(obj, dict):
        if "rows" in obj and isinstance(obj["rows"], list):
            keys = sorted(obj["rows"][0].keys()) if obj["rows"] else []
            return (f"list  count={obj.get('count')}  rows={len(obj['rows'])}  "
                    f"row_keys={keys}")
        return f"dict  keys={sorted(obj.keys())}"
    if isinstance(obj, list):
        keys = sorted(obj[0].keys()) if obj and isinstance(obj[0], dict) else []
        return f"array  len={len(obj)}  item_keys={keys}"
    return type(obj).__name__


def _main() -> int:
    print(f"BASE = {BASE}")
    print(f"auth = Basic <token len {len(_token())}>")
    for path, params in _CANDIDATES:
        try:
            status, body = get(path, params)
            print(f"[{status}] {path}  ->  {_shape(body)}")
        except urllib.error.HTTPError as e:
            print(f"[{e.code}] {path}  ->  HTTPError: {e.reason}")
        except urllib.error.URLError as e:
            print(f"[ERR] {path}  ->  URLError: {e.reason}")
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {path}  ->  {type(e).__name__}: {e}")
    return 0


# --------------------------------------------------------------- deep dive ---
# PII-safe recording fields (NO customerName / calleridnum / fileText content).
_SAFE_FIELDS = ("id,createdAt,queue,duration,format,transcribeStatus,"
                "transcriptionStatus,sentiment,sentimentAnalysisStatus,"
                "postCallAnalyticsStatus,userDisposition,userSecondDisposition,"
                "agentName,membername,numberOfChannels,type")

_IT_STOP = {"che", "non", "sono", "grazie", "buongiorno", "salve", "allora",
            "perché", "quindi", "cosa", "bene", "sì", "ecco", "per", "con",
            "una", "del", "della", "come", "più", "anche", "questo"}


def _dist(values):
    from collections import Counter
    return dict(Counter(v for v in values).most_common())


def _rows(body):
    return body.get("rows", []) if isinstance(body, dict) else (body or [])


def _explore(n: int = 100) -> int:
    print(f"BASE = {BASE}\n")

    # 1. Queues — names reveal which are sales (no PII).
    try:
        _, q = get("voice/queues", {"fields": "id,name,description,strategy",
                                    "limit": 50})
        rows = _rows(q)
        print(f"== QUEUES ({q.get('count') if isinstance(q, dict) else len(rows)}) ==")
        for r in rows:
            desc = (r.get("description") or "")[:55]
            print(f"  [{r.get('id'):>3}] {str(r.get('name')):<28} {desc}")
    except Exception as e:  # noqa: BLE001
        print(f"queues err: {e}")

    # 2. Recordings sample (safe fields only — we never read PII keys).
    print(f"\n== RECORDINGS sample (n={n}, recent) ==")
    sample = []
    for params in ({"fields": _SAFE_FIELDS, "limit": n, "sort": "-createdAt"},
                   {"fields": _SAFE_FIELDS, "limit": n}):
        try:
            _, body = get("voice/recordings", params)
            sample = _rows(body)
            print(f"  total count = {body.get('count')}   (sample {len(sample)})")
            break
        except urllib.error.HTTPError as e:
            print(f"  retry (sort rejected? {e.code})")
    if sample:
        dates = sorted(r.get("createdAt", "") for r in sample if r.get("createdAt"))
        print(f"  sample date range: {dates[0]} .. {dates[-1]}")
        print(f"  transcribeStatus     : {_dist(r.get('transcribeStatus') for r in sample)}")
        print(f"  transcriptionStatus  : {_dist(r.get('transcriptionStatus') for r in sample)}")
        print(f"  format               : {_dist(r.get('format') for r in sample)}")
        print(f"  numberOfChannels     : {_dist(r.get('numberOfChannels') for r in sample)}")
        print(f"  sentimentAnalysisStat: {_dist(r.get('sentimentAnalysisStatus') for r in sample)}")
        print(f"  postCallAnalyticsStat: {_dist(r.get('postCallAnalyticsStatus') for r in sample)}")
        print(f"  top queues           : {dict(list(_dist(r.get('queue') for r in sample).items())[:12])}")
        print(f"  userDisposition top  : {dict(list(_dist(r.get('userDisposition') for r in sample).items())[:15])}")

        # oldest record date (range low end)
        try:
            _, old = get("voice/recordings", {"fields": "id,createdAt",
                                              "limit": 3, "sort": "createdAt"})
            print(f"  OLDEST createdAt     : {[r.get('createdAt') for r in _rows(old)]}")
        except Exception as e:  # noqa: BLE001
            print(f"  oldest err: {e}")

        # 3. One full record — assess fileText WITHOUT printing its content.
        rid = sample[0].get("id")
        try:
            _, rec = get(f"voice/recordings/{rid}")
            ft = rec.get("fileText") or ""
            words = ft.split()
            it = sum(1 for w in words if w.lower().strip(".,!?;:\"'") in _IT_STOP)
            print(f"\n== fileText probe (recording {rid}) ==")
            print(f"  fileText present={bool(ft)}  chars={len(ft)}  words={len(words)}  "
                  f"italian_stopword_ratio={it/max(len(words),1):.0%}")
            print(f"  fileUri set={bool(rec.get('fileUri'))}  format={rec.get('format')}  "
                  f"duration={rec.get('duration')}  channels={rec.get('numberOfChannels')}")
            tj = rec.get("TranscriptionJobs")
            print(f"  TranscriptionJobs={len(tj) if isinstance(tj, list) else tj}")
        except Exception as e:  # noqa: BLE001
            print(f"  full-record err: {e}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true",
                    help="Deep-dive recordings/queues schema (PII-safe aggregates).")
    ap.add_argument("-n", type=int, default=100, help="Recordings sample size.")
    a = ap.parse_args()
    raise SystemExit(_explore(a.n) if a.explore else _main())
