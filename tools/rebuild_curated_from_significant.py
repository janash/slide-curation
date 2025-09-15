#!/usr/bin/env python3
"""
rebuild_curated_from_significant.py

Rebuild curated.pdf deterministically from an existing significant.json
without invoking any LLM. This expects a per-video directory structure:

  slides/<base>/
    pages/*.png
    significant.json  # contains .results[] with {page, keep, ...}

Usage:
  python tools/rebuild_curated_from_significant.py slides/<base>

It writes/overwrites slides/<base>/curated.pdf using only results with keep=true,
preserving source order as listed in significant.json.

Requires: img2pdf (prefer running inside the slides-ocr conda env).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import img2pdf


def rebuild(slides_dir: Path) -> int:
    slides_dir = slides_dir.resolve()
    sig = slides_dir / "significant.json"
    pages_dir = slides_dir / "pages"
    out_pdf = slides_dir / "curated.pdf"

    if not sig.is_file():
        raise SystemExit(f"Missing significant.json: {sig}")
    if not pages_dir.is_dir():
        raise SystemExit(f"Missing pages directory: {pages_dir}")

    data = json.loads(sig.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        raise SystemExit(f"No results in {sig}")

    keep_pages: List[str] = [r.get("page") for r in results if r.get("keep")]
    # Validate files exist in pages/
    paths = []
    missing = []
    for name in keep_pages:
        p = pages_dir / str(name)
        if p.is_file():
            paths.append(p)
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"Missing PNG(s) under {pages_dir}: {missing[:5]}{'...' if len(missing)>5 else ''}")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert([str(p) for p in paths]))

    print(json.dumps({
        "slides_dir": str(slides_dir),
        "kept": len(paths),
        "output_pdf": str(out_pdf),
        "policy": data.get("policy"),
    }, indent=2))
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild curated.pdf from significant.json (no LLM)")
    ap.add_argument("slides_dir", help="Path like slides/<base>")
    args = ap.parse_args(argv)
    return rebuild(Path(args.slides_dir))


if __name__ == "__main__":
    raise SystemExit(main())

