#!/usr/bin/env python3
"""
annotate_slides.py

Create an annotated PDF for a deck by overlaying the per-slide
concept and reason onto each kept slide, without modifying curated.pdf.

Inputs under slides/<base>/:
  - pages/*.png
  - significant.csv (preferred) or significant.json

Output:
  - slides/<base>/<out> (default: curated_annotated.pdf)

Usage:
  python tools/annotate_slides.py slides/<base> --out curated_annotated.pdf

Notes:
  - No LLM. Uses PIL to draw overlays and img2pdf to build the PDF.
  - If significant.csv is missing, falls back to significant.json.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import List, Tuple

import img2pdf
from PIL import Image, ImageDraw, ImageFont


def _load_decisions_csv(csv_path: Path, concept_field: str = "concept", keep_field: str = "keep") -> List[Tuple[str, str, str]]:
    """Return list of (page, concept, reason) for rows with keep=true, in order."""
    out: List[Tuple[str, str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            keep_val = str(row.get(keep_field, "")).strip().lower()
            keep = keep_val in ("true", "1", "yes")
            if not keep:
                continue
            page = str(row.get("page", "")).strip()
            if not page:
                continue
            concept = str(row.get(concept_field, "")).strip()
            reason = str(row.get("reason", "")).strip()
            out.append((page, concept, reason))
    return out


def _load_decisions_json(json_path: Path, concept_field: str = "concept", keep_field: str = "keep") -> List[Tuple[str, str, str]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str, str]] = []
    for r in data.get("results", []):
        if not r.get(keep_field):
            continue
        page = str(r.get("page", "")).strip()
        if not page:
            continue
        concept = str(r.get(concept_field, "")).strip()
        reason = str(r.get("reason", "")).strip()
        out.append((page, concept, reason))
    return out


def _load_decisions_json_ordered(json_path: Path, concept_field: str = "concept", keep_field: str = "keep") -> List[Tuple[str, str, str]]:
    """Load decisions in the exact order of the JSON results array.

    This ensures the annotation order matches the selection tool's output (e.g., Pass 2).
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str, str]] = []
    results = data.get("results", [])
    for r in results:
        if not r.get(keep_field):
            continue
        page = str(r.get("page", "")).strip()
        if not page:
            continue
        concept = str(r.get(concept_field, "")).strip()
        reason = str(r.get("reason", "")).strip()
        out.append((page, concept, reason))
    return out


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try DejaVuSans (usually shipped with Pillow); fall back to default bitmap font.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Wrap text to fit within max_width, breaking long tokens if needed.

    - Normal behavior: wrap on spaces.
    - If a single token exceeds max_width, break it into smaller chunks that fit.
    """
    if not text:
        return []

    def break_long_token(token: str) -> List[str]:
        # Split extremely long tokens (e.g., long identifiers/URLs) to avoid overflow
        parts: List[str] = []
        s = token
        while s:
            # Binary search for the longest prefix that fits
            lo, hi = 1, len(s)
            fit = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if draw.textlength(s[:mid], font=font) <= max_width:
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
        candidate = (f"{cur} {w}" if cur else w).strip()
        if draw.textlength(candidate, font=font) <= max_width:
            cur = candidate
            continue
        # Current line can't fit the next token; emit current line
        if cur:
            lines.append(cur)
            cur = ""
        # If the token itself is too long, break it into chunks
        if draw.textlength(w, font=font) > max_width:
            chunks = break_long_token(w)
            # All but the last chunk become full lines
            for ch in chunks[:-1]:
                lines.append(ch)
            cur = chunks[-1]
        else:
            cur = w
    if cur:
        lines.append(cur)
    # Final safeguard: ensure no line exceeds max_width
    safe_lines: List[str] = []
    for ln in lines:
        if draw.textlength(ln, font=font) <= max_width:
            safe_lines.append(ln)
        else:
            for ch in break_long_token(ln):
                safe_lines.append(ch)
    return safe_lines


def _annotate_image_overlay(img_path: Path, concept: str, reason: str) -> Image.Image:
    im = Image.open(img_path).convert("RGBA")
    W, H = im.size
    # Scale font relative to width; clamp for readability
    base_size = max(14, min(28, W // 60))
    font = _load_font(base_size)
    draw = ImageDraw.Draw(im, mode="RGBA")

    # Compose overlay text
    parts: List[str] = []
    # If displaying student_concept (v2), do not prefix with "Concept:" and skip reason by default.
    concept_is_sentence = False
    try:
        # heuristically detect if concept already looks like a sentence
        concept_is_sentence = bool(concept) and concept.strip().lower().startswith("this slide")
    except Exception:
        concept_is_sentence = False
    if concept:
        parts.append(concept if concept_is_sentence else f"Concept: {concept}")
    # reason handled by caller via include_reason flow; keep fallback for overlay path
    if reason and not concept_is_sentence:
        parts.append(f"Reason: {reason}")
    text = "\n".join(parts)

    # Layout: bottom overlay with padding
    pad = max(12, base_size // 2)
    max_text_width = W - 2 * pad
    # Wrap each logical line separately and then combine
    lines: List[str] = []
    for logical in parts:
        lines.extend(_wrap_text(draw, logical, font, max_text_width))
    if not lines:
        return im.convert("RGB")

    # Measure height
    line_height = int(font.getbbox("A")[3] - font.getbbox("A")[1]) + 2
    overlay_height = pad + len(lines) * line_height + pad

    # Draw semi-transparent panel
    panel_top = H - overlay_height
    panel = Image.new("RGBA", (W, overlay_height), (0, 0, 0, 160))
    im.alpha_composite(panel, dest=(0, panel_top))

    # Draw text (white)
    y = panel_top + pad
    for ln in lines:
        draw.text((pad, y), ln, font=font, fill=(255, 255, 255, 255))
        y += line_height

    return im.convert("RGB")


def _annotate_image_below(img_path: Path, concept: str, reason: str) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    base_size = max(14, min(28, W // 60))
    font = _load_font(base_size)
    draw = ImageDraw.Draw(im)

    parts: List[str] = []
    concept_is_sentence = False
    try:
        concept_is_sentence = bool(concept) and concept.strip().lower().startswith("this slide")
    except Exception:
        concept_is_sentence = False
    if concept:
        parts.append(concept if concept_is_sentence else f"Concept: {concept}")
    if reason and not concept_is_sentence:
        parts.append(f"Reason: {reason}")
    pad = max(12, base_size // 2)
    max_text_width = W - 2 * pad
    lines: List[str] = []
    for logical in parts:
        lines.extend(_wrap_text(draw, logical, font, max_text_width))
    if not lines:
        return im

    line_height = int(font.getbbox("A")[3] - font.getbbox("A")[1]) + 2
    panel_height = pad + len(lines) * line_height + pad

    out = Image.new("RGB", (W, H + panel_height), "white")
    out.paste(im, (0, 0))
    d2 = ImageDraw.Draw(out)
    y = H + pad
    for ln in lines:
        d2.text((pad, y), ln, font=font, fill=(0, 0, 0))
        y += line_height
    return out


def build_pdf(images: List[Image.Image], out_path: Path) -> None:
    # Convert PIL images to JPEG bytes for img2pdf to keep size reasonable
    bufs: List[bytes] = []
    for im in images:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        bufs.append(buf.getvalue())
    with open(out_path, "wb") as f:
        f.write(img2pdf.convert(bufs))


def build_pdf_searchable_below(decisions: List[Tuple[Path, str, str]], out_path: Path) -> None:
    """Build a PDF where the slide image stays raster, but the caption text below is real, searchable text.

    decisions: list of (img_path, concept, reason)
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.colors import black, white
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        # Fallback: if reportlab is not available, use raster pipeline
        imgs = []
        for p, concept, reason in decisions:
            imgs.append(_annotate_image_below(p, concept, reason))
        build_pdf(imgs, out_path)
        return

    c = None
    for idx, (img_path, concept, reason) in enumerate(decisions, start=1):
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        base_size = max(14, min(28, W // 60))
        # Use ImageReader on the original file path for embedding

        # Prepare wrapped lines using PIL for width metrics (pixel == point assumption)
        pil_draw = ImageDraw.Draw(im)
        font = _load_font(base_size)
        pad = max(12, base_size // 2)
        lines: List[str] = []
        parts: List[str] = []
        if concept:
            parts.append(f"Concept: {concept}")
        if reason:
            parts.append(f"Reason: {reason}")
        for logical in parts:
            lines.extend(_wrap_text(pil_draw, logical, font, W - 2 * pad))
        line_height = int(font.getbbox("A")[3] - font.getbbox("A")[1]) + 2
        panel_h = pad + (len(lines) * line_height if lines else 0) + pad

        # Create canvas on first page or reuse
        if c is None:
            c = canvas.Canvas(str(out_path), pagesize=(W, H + panel_h))
        else:
            c.setPageSize((W, H + panel_h))

        # Draw image at top (origin at bottom-left); caption panel below
        # Draw caption background
        c.setFillColor(white)
        c.rect(0, 0, W, panel_h, fill=1, stroke=0)
        # Draw the image
        c.drawImage(ImageReader(str(img_path)), 0, panel_h, width=W, height=H, preserveAspectRatio=False, mask='auto')

        # Draw text lines as real text (top-down so first line at top of panel)
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
    ap = argparse.ArgumentParser(description="Create an annotated PDF overlaying concept and reason on kept slides.")
    ap.add_argument("slides_dir", help="Path like slides/<base>")
    ap.add_argument("--out", default="curated_annotated.pdf", help="Output PDF filename (default: curated_annotated.pdf)")
    ap.add_argument("--placement", choices=["overlay","below"], default="overlay", help="Place text panel as overlay or entirely below the image (default: overlay)")
    ap.add_argument("--source", choices=["v1","v2"], default="v1", help="Use v1 significant or v2 (reviewed) outputs")
    ap.add_argument("--concept-field", default="concept", help="Field to display for concept (e.g., student_concept in v2)")
    ap.add_argument("--searchable-text", action="store_true", help="Emit text as real PDF text (searchable). Supported for placement=below.")
    ap.add_argument("--include-reason", action="store_true", help="Include 'Reason' text under captions (default off for v2, on for v1)")
    args = ap.parse_args(argv)

    slides_dir = Path(args.slides_dir).resolve()
    pages_dir = slides_dir / "pages"
    if args.source == "v2":
        csv_path = slides_dir / "significant_v2.csv"
        json_path = slides_dir / "significant_v2.json"
        keep_field = "keep2"
    else:
        csv_path = slides_dir / "significant.csv"
        json_path = slides_dir / "significant.json"
        keep_field = "keep"
    if not pages_dir.is_dir():
        raise FileNotFoundError(f"Missing pages directory: {pages_dir}")
    decisions: List[Tuple[str, str, str]] = []
    # Prefer JSON order for v2 to match the selection tool's sequence exactly
    if args.source == "v2" and json_path.is_file():
        decisions = _load_decisions_json_ordered(json_path, concept_field=args.concept_field, keep_field=keep_field)
    elif csv_path.is_file():
        decisions = _load_decisions_csv(csv_path, concept_field=args.concept_field, keep_field=keep_field)
    elif json_path.is_file():
        decisions = _load_decisions_json(json_path, concept_field=args.concept_field, keep_field=keep_field)
    else:
        raise FileNotFoundError("Need significant.csv or significant.json to locate keeps and metadata")

    if not decisions:
        raise RuntimeError("No kept slides found in metadata")

    out_pdf = slides_dir / args.out
    if args.searchable_text and args.placement == "below":
        items = []
        for page, concept, reason in decisions:
            p = pages_dir / page
            if p.is_file():
                # For v2/student_concept, suppress reason unless explicitly requested
                if args.source == "v2" and not args.include_reason:
                    reason = ""
                items.append((p, concept, reason))
        build_pdf_searchable_below(items, out_pdf)
    else:
        images: List[Image.Image] = []
        for page, concept, reason in decisions:
            img_path = pages_dir / page
            if not img_path.is_file():
                # Skip missing images
                continue
            if args.placement == "below":
                if args.source == "v2" and not args.include_reason:
                    reason = ""
                images.append(_annotate_image_below(img_path, concept, reason))
            else:
                images.append(_annotate_image_overlay(img_path, concept, reason))
        build_pdf(images, out_pdf)
    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
