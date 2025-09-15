#!/usr/bin/env python3
"""
export_captions.py

Export an ordered captions file (JSON) for a deck, based on Pass 2 results.
Only includes slides that are kept (keep2=true) and preserves the exact order
from significant_v2.json. This avoids any mismatch when generating annotated PDFs.

Usage:
  python tools/export_captions.py slides/<base> --field student_concept --out captions.json

Outputs:
  slides/<base>/captions.json (by default)

Schema:
  {
    "slides_dir": "slides/<base>",
    "field": "student_concept",
    "captions": [
      {"page": "0001_0:00:00.png", "text": "This slide demonstrates ..."},
      ... ordered ...
    ]
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export ordered captions from Pass 2 (significant_v2.json)")
    ap.add_argument("slides_dir", help="Path like slides/<base>")
    ap.add_argument("--field", default="student_concept", help="Field to export as caption text (e.g., student_concept)")
    ap.add_argument("--out", default="captions.json", help="Output filename (default: captions.json in slides_dir)")
    args = ap.parse_args(argv)

    slides_dir = Path(args.slides_dir).resolve()
    sig_v2 = slides_dir / "significant_v2.json"
    if not sig_v2.is_file():
        raise SystemExit(f"Missing Pass 2 JSON: {sig_v2}")

    data = json.loads(sig_v2.read_text(encoding="utf-8"))
    captions: List[Dict[str, str]] = []
    for r in data.get("results", []):
        if not r.get("keep2"):
            continue
        page = str(r.get("page", "")).strip()
        if not page:
            continue
        text = str(r.get(args.field, "")).strip()
        captions.append({"page": page, "text": text})

    out_path = slides_dir / args.out
    out_obj = {"slides_dir": str(slides_dir), "field": args.field, "captions": captions}
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

