"""Create an aligned JSON enrichment block for the artifact-tool CSV writer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .csv_handoff import enrich_csv
from .fill_gsheet_loi import OUT_HEADER


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    source, block = enrich_csv(args.csv_path)
    payload = {
        "source": str(Path(args.csv_path)),
        "source_rows": len(source),
        "headers": OUT_HEADER,
        "rows": block.fillna("").astype(str).values.tolist(),
        "status_counts": block["loi_match_status"].value_counts().to_dict(),
        "rows_with_loi_band": int(block["loi_band"].astype(str).ne("").sum()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["status_counts"], indent=2))
    print(f"rows with loi_band: {payload['rows_with_loi_band']}")
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

