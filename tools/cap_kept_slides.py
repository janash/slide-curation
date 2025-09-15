#!/usr/bin/env python3
"""
cap_kept_slides.py

Cap the number of kept slides in significant.json/.csv to top-N by score,
then rebuild curated.pdf from those keeps (no LLM calls).

Usage:
  python tools/cap_kept_slides.py slides/<base> --max 10

Effects:
  - Edits significant.json: sets keep=true only for top-N by score; others false.
  - Edits significant.csv: syncs keep column to match JSON.
  - Rebuilds curated.pdf from keep=true using rebuild_curated_from_significant.py.

Notes:
  - Preserves original slide order in the PDF; selection is based on score.
  - If multiple slides tie at the boundary, selection is stable by original order.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Set


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cap_keeps(slides_dir: Path, max_keep: int) -> int:
    slides_dir = slides_dir.resolve()
    sig_json = slides_dir / "significant.json"
    sig_csv = slides_dir / "significant.csv"
    if not sig_json.is_file() or not sig_csv.is_file():
        raise SystemExit("Missing significant.json or significant.csv")

    data = load_json(sig_json)
    results: List[Dict[str, Any]] = list(data.get("results", []))
    # Build list of kept with scores and preserve index for stable tie-break
    kept = [(i, r.get("page"), float(r.get("score", 0.0))) for i, r in enumerate(results) if r.get("keep")]
    if not kept:
        # Nothing kept; nothing to do
        return 0
    # Sort by score desc, then by original order (index) for stability
    kept_sorted = sorted(kept, key=lambda t: (-t[2], t[0]))
    top = kept_sorted[:max_keep]
    keep_pages: Set[str] = {page for _, page, _ in top}

    # Update JSON keeps
    new_results: List[Dict[str, Any]] = []
    for r in results:
        p = str(r.get("page"))
        r = dict(r)
        r["keep"] = p in keep_pages
        new_results.append(r)
    data["results"] = new_results
    save_json(sig_json, data)

    # Update CSV keeps (in-place overwrite)
    rows: List[Dict[str, Any]] = []
    with sig_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    # Normalize page key name
    page_key = "page" if any("page" in r for r in rows) else ("file" if any("file" in r for r in rows) else "page")
    if page_key not in (fieldnames or []):
        fieldnames = [page_key] + [fn for fn in (fieldnames or [])]
    # Ensure keep in fieldnames
    if "keep" not in (fieldnames or []):
        fieldnames = (fieldnames or []) + ["keep"]
    for r in rows:
        p = str(r.get(page_key, ""))
        r["keep"] = "True" if p in keep_pages else "False"
    with sig_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return len(keep_pages)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cap kept slides to top-N by score and rebuild curated.pdf")
    ap.add_argument("slides_dir", help="slides/<base>")
    ap.add_argument("--max", type=int, required=True, help="Maximum number of slides to keep")
    args = ap.parse_args(argv)

    sd = Path(args.slides_dir)
    kept = cap_keeps(sd, args.max)

    # Rebuild curated.pdf from edited significant.json
    from subprocess import run, CalledProcessError
    cp = run(["python", "tools/rebuild_curated_from_significant.py", str(sd)])
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)
    print(json.dumps({"slides_dir": str(sd), "kept_after_cap": kept}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

