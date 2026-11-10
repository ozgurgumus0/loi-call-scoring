"""Resolve queue_id GUIDs to human-readable names.

Walks the distinct queue_ids in data/genesys_metadata.csv, fetches each via
GET /api/v2/routing/queues/{id}, and writes a lookup table:

  data/queues.csv  -> queue_id, queue_name, n_recordings, call_type

`call_type` is left blank — fill it in manually (sales / booking / other) once
you've seen the names. The MVP can then filter calls by call_type.

Usage:
  python -m src.resolve_queues
"""
from __future__ import annotations

import csv
import sys
from collections import Counter

from . import config
from .genesys_client import GenesysClient, GenesysError


def main() -> int:
    meta = config.GENESYS_METADATA_CSV
    if not meta.exists():
        print(f"missing {meta}", file=sys.stderr)
        return 1

    with meta.open(newline="", encoding="utf-8") as f:
        counts = Counter(r["queue_id"] for r in csv.DictReader(f) if r["queue_id"])
    print(f"distinct queues: {len(counts)}  (recordings tagged with a queue)")

    gc = GenesysClient()
    rows = []
    for qid, n in counts.most_common():
        try:
            q = gc._api_request("GET", f"/api/v2/routing/queues/{qid}")
            name = q.get("name", "?") if q else "?"
        except GenesysError as e:
            name = f"<ERROR: {e}>"
        rows.append({"queue_id": qid, "queue_name": name,
                     "n_recordings": n, "call_type": ""})
        print(f"  {n:>5}  {qid}  {name}")

    out = config.DATA / "queues.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["queue_id", "queue_name",
                                          "n_recordings", "call_type"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
