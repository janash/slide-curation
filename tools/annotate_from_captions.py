#!/usr/bin/env python3
"""
annotate_from_captions.py

Generate an annotated PDF using an explicit, ordered captions file. This guarantees
1:1 matching between pages and captions. Captions are drawn as real PDF text (ReportLab)
in a white panel below the image (no overlay).

Usage:
  python tools/annotate_from_captions.py slides/<base>/captions.json --out curated_annotated_from_captions.pdf

captions.json schema:
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
from typing import List, Tuple

from PIL import Image


def _wrap_reportlab(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    """Wrap text using ReportLab font metrics so width matches drawing.

    - Wrap at spaces; if a single token exceeds max_width, break at character level.
    """
    from reportlab.pdfbase import pdfmetrics
    if not text:
        return []

    def text_w(s: str) -> float:
        return pdfmetrics.stringWidth(s, font_name, font_size)

    def break_long_token(token: str) -> List[str]:
        parts: List[str] = []
        s = token
        while s:
            # find longest prefix that fits
            lo, hi = 1, len(s)
            fit = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if text_w(s[:mid]) <= max_width:
                    fit = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            parts.append(s[:fit])
            s = s[fit:]
        return parts

    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        cand = (f"{cur} {w}" if cur else w)
        if text_w(cand) <= max_width:
            cur = cand
            continue
        if cur:
            lines.append(cur)
            cur = ""
        if text_w(w) > max_width:
            chunks = break_long_token(w)
            lines.extend(chunks[:-1])
            cur = chunks[-1]
        else:
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build_pdf_from_captions(captions_path: Path, out_path: Path) -> None:
    data = json.loads(captions_path.read_text(encoding="utf-8"))
    slides_dir = Path(data["slides_dir"]).resolve()
    pages_dir = slides_dir / "pages"
    items: List[Tuple[Path, str]] = []
    for item in data.get("captions", []):
        p = pages_dir / item.get("page", "")
        txt = str(item.get("text", "")).strip()
        if p.is_file():
            items.append((p, txt))

    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import black, white
    from reportlab.lib.utils import ImageReader

    c = None
    for img_path, text in items:
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        base_size = max(14, min(28, W // 60))
        pad = max(12, base_size // 2)
        # Measure wrapped text using ReportLab metrics to match drawing width (points == pixels here)
        lines = _wrap_reportlab(text, "Helvetica", base_size, W - 2 * pad)
        line_height = int(base_size * 1.25)
        panel_h = pad + (len(lines) * line_height if lines else 0) + pad

        if c is None:
            c = canvas.Canvas(str(out_path), pagesize=(W, H + panel_h))
        else:
            c.setPageSize((W, H + panel_h))

        # Panel background
        c.setFillColor(white)
        c.rect(0, 0, W, panel_h, fill=1, stroke=0)
        # Image
        c.drawImage(ImageReader(str(img_path)), 0, panel_h, width=W, height=H, preserveAspectRatio=False, mask='auto')
        # Text (top-down so first line appears above subsequent lines)
        c.setFillColor(black)
        c.setFont("Helvetica", base_size)
        y = panel_h - pad - line_height
        for ln in lines:
            c.drawString(pad, y, ln)
            y -= line_height
        c.showPage()

    if c is not None:
        c.save()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate an annotated PDF strictly from captions.json order")
    ap.add_argument("captions_json", help="Path to captions.json produced by export_captions.py")
    ap.add_argument("--out", default="curated_annotated_from_captions.pdf", help="Output PDF filename")
    args = ap.parse_args(argv)

    captions_path = Path(args.captions_json).resolve()
    if not captions_path.is_file():
        raise SystemExit(f"Missing captions file: {captions_path}")
    out_path = captions_path.parent / args.out
    build_pdf_from_captions(captions_path, out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
