#!/usr/bin/env python3
"""
review_kept_slides.py (Pass 2)

Second-pass curation over the slides kept by Pass 1. Reviews only the kept
sequence with forward/backward context (windowed) to:
  - Drop in-progress or redundant states in favor of clearer finals
  - Produce student-facing concept labels (2–6 words)

Inputs under slides/<base>/:
  - pages/*.png
  - index.csv
  - significant.json (from Pass 1)

Outputs under slides/<base>/:
  - significant_v2.json
  - significant_v2.csv
  - curated_v2.pdf

Prompt is loaded from one of:
  1) env LECTURE_REVIEW_PROMPT (path)
  2) prompts/kept_review.md
  3) tools-local prompts next to this file
  4) Fallback inline minimal prompt

Usage example:
  OPENAI_API_KEY=... LECTURE_REVIEW_PROMPT=prompts/kept_review.md \
  python tools/review_kept_slides.py slides/week3/3.1_03_class_constructors \
    --model gpt-4o --window 8 --stride 3 --threshold 1.0
"""
from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import img2pdf
from tenacity import retry, wait_exponential, stop_after_attempt

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class ReviewDecision:
    page: str
    keep2: bool
    score: float
    student_concept: str


def _load_prompt() -> str:
    import os
    cands: List[Path] = []
    envp = os.environ.get("LECTURE_REVIEW_PROMPT") or os.environ.get("REVIEW_PROMPT")
    if envp:
        cands.append(Path(envp))
    cands.append(Path("prompts/kept_review.md"))
    try:
        here = Path(__file__).resolve()
        cands.append(here.parent.parent / "prompts" / "kept_review.md")
        cands.append(here.parent / "prompts" / "kept_review.md")
    except Exception:
        pass
    for p in cands:
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except Exception:
            continue
    return (
        "You will be shown a short sequence of already-kept slides. "
        "For each slide, decide keep=true/false, and if kept, provide a concise student-facing concept (2–6 words)."
    )


def _b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def _review_window(model: str, items: List[Tuple[int, Path]], temperature: float) -> List[ReviewDecision]:
    if OpenAI is None:
        raise RuntimeError("openai SDK not available. pip install openai>=1.0.0")
    client = OpenAI()
    prompt = _load_prompt()
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": (
            "Return ONLY JSON with 'results' as an array. "
            "Each item: {page:<name>, keep:<bool>, score:<0..1>, student_concept:<text>}"
        )},
    ]
    for idx, p in items:
        content.append({"type": "text", "text": f"slide_index: {idx} | page: {p.name}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_b64(p)}", "detail": "high"}
        })
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Respond only with valid JSON formatted as requested."},
            {"role": "user", "content": content},
        ],
    )
    txt = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(txt)
    except Exception:
        start = txt.find("{"); end = txt.rfind("}")
        data = json.loads(txt[start:end+1]) if start>=0 and end>=0 else {"results": []}

    out: List[ReviewDecision] = []
    for r in data.get("results", []):
        try:
            out.append(ReviewDecision(
                page=str(r.get("page")),
                keep2=bool(r.get("keep", False)),
                score=float(r.get("score", 0.0)),
                student_concept=str(r.get("student_concept", "")),
            ))
        except Exception:
            continue
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pass 2 review over kept slides to produce student-facing concepts and drop partials.")
    ap.add_argument("slides_dir", help="Path like slides/<base>")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=1.0, help=">=1.0 means use explicit keep flag only")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args(argv)

    slides_dir = Path(args.slides_dir).resolve()
    pages_dir = slides_dir / "pages"
    index_csv = slides_dir / "index.csv"
    sig1 = slides_dir / "significant.json"
    if not pages_dir.is_dir():
        raise SystemExit(f"Missing pages dir: {pages_dir}")
    if not index_csv.is_file():
        raise SystemExit(f"Missing index.csv: {index_csv}")
    if not sig1.is_file():
        raise SystemExit(f"Missing Pass 1 significant.json: {sig1}")

    v1 = json.loads(sig1.read_text(encoding="utf-8"))
    kept_pages = [r["page"] for r in v1.get("results", []) if r.get("keep")]
    files = [pages_dir / p for p in kept_pages if (pages_dir / p).is_file()]

    # Make windows
    def make_windows(n: int, w: int, s: int):
        i=0
        while i < n:
            yield list(range(i, min(i+w, n)))
            if i + w >= n:
                break
            i += s

    # Gather decisions per page across windows
    all_scores: Dict[str, List[ReviewDecision]] = {f.name: [] for f in files}
    for idxs in make_windows(len(files), args.window, args.stride):
        items = [(i, files[i]) for i in idxs]
        decs = _review_window(args.model, items, args.temperature)
        by_page = {d.page: d for d in decs}
        for i in idxs:
            name = files[i].name
            d = by_page.get(name)
            if d:
                all_scores[name].append(d)

    # Reduce per page
    final: List[ReviewDecision] = []
    for f in files:
        arr = all_scores.get(f.name, [])
        if not arr:
            final.append(ReviewDecision(page=f.name, keep2=False, score=0.0, student_concept=""))
        else:
            best = max(arr, key=lambda x: x.score)
            if args.threshold >= 1.0 - 1e-9:
                keep_flag = any(x.keep2 for x in arr)
            else:
                keep_flag = any(x.keep2 for x in arr) or (best.score >= args.threshold)
            concept = best.student_concept
            final.append(ReviewDecision(page=f.name, keep2=keep_flag, score=best.score, student_concept=concept))

    # Write v2 JSON
    out_json = slides_dir / "significant_v2.json"
    payload = {
        "model": args.model,
        "window": args.window,
        "stride": args.stride,
        "threshold": args.threshold,
        "results": [asdict(d) for d in final],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build v2 CSV by merging index + v2 decisions
    df = pd.read_csv(index_csv)
    # Normalize to 'page' filename column like other tools
    if "page" in df.columns and not df["page"].astype(str).str.endswith(".png").any():
        df = df.rename(columns={"page":"page_num"})
    if "file" in df.columns:
        df = df.rename(columns={"file":"page"})
    elif "filename" in df.columns:
        df = df.rename(columns={"filename":"page"})
    elif "name" in df.columns:
        df = df.rename(columns={"name":"page"})

    v2_df = pd.DataFrame([asdict(d) for d in final])
    merged = df.merge(v2_df, on="page", how="left")
    out_csv = slides_dir / "significant_v2.csv"
    merged.to_csv(out_csv, index=False)

    # Build curated_v2.pdf
    keep_paths = [str(pages_dir / d.page) for d in final if d.keep2]
    out_pdf = slides_dir / "curated_v2.pdf"
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(keep_paths))

    print(json.dumps({
        "kept_count": len(keep_paths),
        "total_v1_kept": len(files),
        "output_json": str(out_json),
        "output_csv": str(out_csv),
        "output_pdf": str(out_pdf),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

