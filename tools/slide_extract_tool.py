#!/usr/bin/env python3
"""
slide_extract_tool.py  (MODEL-ONLY, CONTEXT-AWARE)

Agent-callable tool that uses an **image-capable LLM only** (no heuristics) to
look at already-extracted slides and pick the significant ones **in context**.

Inputs (from your extractor):
  slides/<base>/pages/*.png
  slides/<base>/index.csv (must include a 'page' column matching PNG names)

Outputs (written to slides/<base>/):
  - significant.json : per-slide {page, score, keep, reason}
  - significant.csv  : index.csv augmented with score/keep/reason
  - curated.pdf      : PDF containing only kept slides, in source order

Environment:
  OPENAI_API_KEY must be set. No other config required.

Primary design goals:
  - **Model-only** visual judgment. No blank/entropy filters, no dedupe heuristics.
  - **Context-aware**: the model sees *windows* of consecutive slides to decide what matters
    relative to neighbors (e.g., real transitions vs tiny scrolls).
  - **Deterministic-ish**: defaults to temperature=0.0. Your threshold governs final keep.

CLI examples:
  python tools/slide_extract_tool.py slides/1.2_03_errors_exceptions --model gpt-4o
  python tools/slide_extract_tool.py slides/1.2_03_errors_exceptions --window 10 --stride 8 --threshold 0.55

Agent entrypoint:
  run_tool(payload: dict) -> RunResult

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
from PIL import Image
import imagehash

# --- OpenAI client (SDK v1) ---
try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None

TOOL_SCHEMA: Dict[str, Any] = {
    "name": "select_significant_slides",
    "description": (
        "Model-only, context-aware slide selection. Inputs: slides/<base>/ with pages/ and index.csv. "
        "Outputs: significant.json, significant.csv, curated.pdf in that directory."
    ),
    "args_schema": {
        "type": "object",
        "properties": {
            "slides_dir": {"type": "string", "description": "Path like slides/<base>"},
            "model": {"type": "string", "default": "gpt-4o", "description": "Image-capable model"},
            "threshold": {"type": "number", "default": 0.6, "description": "Score cutoff in [0,1]"},
            "max_slides": {"type": "integer", "default": 0, "description": "0 = no cap; otherwise limit kept slides"},
            "window": {"type": "integer", "default": 8, "description": "Slides per context window"},
            "stride": {"type": "integer", "default": 6, "description": "Window step; can be < window for overlap"},
            "temperature": {"type": "number", "default": 0.0, "description": "Sampling temperature"},
            "policy": {"type": "string", "default": "code", "description": "Selection rubric hint (lecture|code|math|doc)"},
            "compare_last": {"type": "boolean", "default": False, "description": "Sequential mode: compare each candidate to the last kept slide (no rolling windows)"},
            "include_kept_concepts": {"type": "boolean", "default": True, "description": "When comparing to last kept, also pass a running list of concepts already kept"},
            "concept_context": {"type": "integer", "default": 20, "description": "Max number of most recent kept concepts to include in the prompt"},
            "pane_dedupe": {"type": "boolean", "default": False, "description": "Before calling the model, drop frames whose output pane region is unchanged vs last kept"},
            "pane_side": {"type": "string", "default": "right", "description": "Which side contains the output/terminal pane: right|left|bottom|top"},
            "pane_frac": {"type": "number", "default": 0.35, "description": "Fraction of width/height to consider as the output pane region"},
            "pane_threshold": {"type": "integer", "default": 2, "description": "Max phash distance to consider the pane unchanged"},
            "start": {"type": "integer", "default": 0, "description": "Optional start page index (1-based) or page_num"},
            "end": {"type": "integer", "default": 0, "description": "Optional end page index (1-based) or page_num"}
        },
        "required": ["slides_dir"]
    },
    "returns": {
        "type": "object",
        "properties": {
            "kept_count": {"type": "integer"},
            "total": {"type": "integer"},
            "output_json": {"type": "string"},
            "output_csv": {"type": "string"},
            "output_pdf": {"type": "string"}
        }
    }
}

@dataclass
class SlideDecision:
    page: str
    score: float
    keep: bool
    reason: str
    # Optional short concept label (2–6 words) summarizing the core idea shown
    concept: str = ""

POLICY_HINTS = {
    "lecture": "Prefer title/section slides, new concepts/definitions, big diagrams/plots, summaries. Avoid tiny incremental changes.",
    "code":    "Prefer large diffs, new functions/APIs, architecture diagrams. Avoid tiny scrolls/caret moves.",
    "math":    "Prefer theorem statements, key equations/derivation steps, result tables. Avoid trivial index tweaks.",
    # Documentation-centric lectures that include code but shouldn't keep typing/scroll frames.
    "doc":     (
        "Prefer final, readable code blocks, section/summary slides, and finished examples. "
        "Avoid typing/scroll frames, caret moves, partial/in-progress code, or negligible visual change. "
        "Keep only stable slides where a unit of code or explanation is complete."
    ),
}

def _load_prompt() -> str:
    """Load the lecture extraction prompt robustly regardless of script location.

    Search order:
      1) Environment variable LECTURE_EXTRACTION_PROMPT (path)
      2) CWD: prompts/lecture_extraction.md
      3) Repo root inferred from this file: ../prompts/lecture_extraction.md
      4) tools-local: ./prompts/lecture_extraction.md (next to this file)
      5) Fallback minimal inline prompt
    """
    import os
    candidates = []
    env_path = os.environ.get("LECTURE_EXTRACTION_PROMPT") or os.environ.get("PROMPT_PATH")
    if env_path:
        candidates.append(Path(env_path))
    # Default prompt path
    candidates.append(Path("prompts/lecture_extraction.md"))
    try:
        here = Path(__file__).resolve()
        # repo root if this file is under tools/
        candidates.append(here.parent.parent / "prompts" / "lecture_extraction.md")
        # local prompts next to the script (rare)
        candidates.append(here.parent / "prompts" / "lecture_extraction.md")
    except Exception:
        pass
    for p in candidates:
        try:
            if p and p.is_file():
                return p.read_text(encoding="utf-8")
        except Exception:
            continue
    # Fallback minimal prompt if file missing
    return (
        "You are a meticulous slide curator. You will be shown a SEQUENCE of slides from a lecture.\n"
        "Judge significance visually relative to neighbors; keep titles/sections, new concepts, key diagrams, code diffs, summaries; skip near-duplicates and tiny scrolls."
    )


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _pane_region_hash(path: Path, side: str, frac: float):
    with Image.open(path) as im:
        im = im.convert("RGB")
        W, H = im.size
        frac = max(0.05, min(0.95, float(frac)))
        if side == "right":
            box = (int(W * (1.0 - frac)), 0, W, H)
        elif side == "left":
            box = (0, 0, int(W * frac), H)
        elif side == "bottom":
            box = (0, int(H * (1.0 - frac)), W, H)
        else:  # top
            box = (0, 0, W, int(H * frac))
        crop = im.crop(box)
        return imagehash.phash(crop)


def _load_selection_hint(slides_dir: Path, extra_file: Optional[str] = None) -> str:
    """Load optional per-deck extra instructions to refine selection behavior.

    Search order:
      1) explicit file path if provided
      2) slides/<base>/selection_hint.md
      3) environment variable SELECTION_HINT
    Returns empty string if none found.
    """
    import os
    # explicit file
    if extra_file:
        p = Path(extra_file)
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except Exception:
            pass
    # per-deck
    try:
        p = slides_dir / "selection_hint.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    # env
    try:
        hint = os.environ.get("SELECTION_HINT", "")
        return hint or ""
    except Exception:
        return ""


@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
def _score_window(model: str, window_items: List[Tuple[int, Path]], temperature: float, policy: str, extra_hint: str) -> List[SlideDecision]:
    if OpenAI is None:
        raise RuntimeError("openai SDK not available. pip install openai>=1.0.0")
    client = OpenAI()

    prompt_text = _load_prompt()
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": "Policy: " + POLICY_HINTS.get(policy, POLICY_HINTS["lecture"])},
        {"type": "text", "text": (
            "Return ONLY strict JSON with top-level key 'results' as an array of objects, one per slide, in the SAME ORDER. "
            "Each object must include: {page: <filename>, keep: <true|false>, score: <0..1>, reason: <detailed explanation>, concept: <short label>}."
        )}
    ]
    if extra_hint:
        content.append({"type": "text", "text": "Additional selection guidance for THIS lecture:"})
        content.append({"type": "text", "text": extra_hint})
    for idx, p in window_items:
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
            {"role": "system", "content": "Respond only with valid JSON. Put all per-slide decisions under key 'results' as an array, in the same order as provided."},
            {"role": "user", "content": content},
        ],
    )
    txt = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(txt)
    except Exception:
        start = txt.find("{"); end = txt.rfind("}")
        data = json.loads(txt[start:end+1]) if start>=0 and end>=0 else {"results": []}

    out: List[SlideDecision] = []
    for r in data.get("results", []):
        try:
            out.append(SlideDecision(
                page=str(r.get("page")),
                score=float(r.get("score", 0.0)),
                keep=bool(r.get("keep", False)),
                reason=str(r.get("reason", "")),
                concept=str(r.get("concept", "")),
            ))
        except Exception:
            continue
    return out


@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
def _score_against_last(
    model: str,
    current_item: Tuple[int, Path],
    last_kept_item: Optional[Tuple[int, Path]],
    temperature: float,
    policy: str,
    extra_hint: str,
    kept_concepts: Optional[List[str]] = None,
    concept_context: int = 20,
    include_kept_concepts: bool = True,
) -> SlideDecision:
    """Ask the model to decide on the CURRENT slide given the LAST KEPT as reference.

    Returns a single SlideDecision for the current page.
    """
    if OpenAI is None:
        raise RuntimeError("openai SDK not available. pip install openai>=1.0.0")
    client = OpenAI()

    prompt_text = _load_prompt()
    idx_cur, p_cur = current_item
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": "Policy: " + POLICY_HINTS.get(policy, POLICY_HINTS["lecture"])},
        {"type": "text", "text": (
            "You will be shown the LAST KEPT slide and the CURRENT candidate. "
            "Keep CURRENT only if it introduces meaningful new information beyond LAST KEPT: new code blocks, finalized definitions, parameter/API changes, new outputs/errors, section dividers. "
            "Avoid keeping minor scroll/caret/highlight moves, autocomplete popups, partial typing, or frames that differ only by trivial repositioning. "
            "Respond only with JSON: {results:[{page, keep, score, reason, concept}]}."
        )},
    ]
    if extra_hint:
        content.append({"type": "text", "text": "Additional selection guidance for THIS lecture:"})
        content.append({"type": "text", "text": extra_hint})
    # Append a short list of concepts already covered to reduce duplicates
    if include_kept_concepts and kept_concepts:
        uniq = []
        seen = set()
        for c in kept_concepts[-max(1, concept_context):]:
            c2 = (c or "").strip()
            if not c2:
                continue
            if c2.lower() in seen:
                continue
            seen.add(c2.lower())
            uniq.append(c2)
        if uniq:
            content.append({"type": "text", "text": "Concepts already kept so far (avoid redundant keeps unless new info is added):"})
            content.append({"type": "text", "text": "; ".join(uniq)})
    if last_kept_item is not None:
        idx_last, p_last = last_kept_item
        content.append({"type": "text", "text": f"last_kept_page: {p_last.name}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_b64(p_last)}", "detail": "high"}
        })
    else:
        content.append({"type": "text", "text": "No last kept slide available (first decision)."})
    content.append({"type": "text", "text": f"current_slide: {p_cur.name}"})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_b64(p_cur)}", "detail": "high"}
    })

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Respond only with valid JSON. Put the decision for the current slide under key 'results' as an array with exactly one object."},
            {"role": "user", "content": content},
        ],
    )
    txt = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(txt)
    except Exception:
        start = txt.find("{"); end = txt.rfind("}")
        data = json.loads(txt[start:end+1]) if start>=0 and end>=0 else {"results": []}

    # Support either an array under results or a single object at root
    obj = None
    if isinstance(data.get("results"), list) and data["results"]:
        obj = data["results"][0]
    elif all(k in data for k in ("page","keep","score")):
        obj = data
    if obj is None:
        return SlideDecision(page=p_cur.name, score=0.0, keep=False, reason="no-decision", concept="")
    try:
        return SlideDecision(
            page=str(obj.get("page") or p_cur.name),
            score=float(obj.get("score", 0.0)),
            keep=bool(obj.get("keep", False)),
            reason=str(obj.get("reason", "")),
            concept=str(obj.get("concept", "")),
        )
    except Exception:
        return SlideDecision(page=p_cur.name, score=0.0, keep=False, reason="parse-error", concept="")


@dataclass
class RunResult:
    kept_count: int
    total: int
    output_json: str
    output_csv: str
    output_pdf: str


def run_tool(payload: Dict[str, Any]) -> RunResult:
    slides_dir = Path(payload["slides_dir"]).resolve()
    model = payload.get("model", "gpt-4o")
    threshold = float(payload.get("threshold", 0.6))
    max_slides = int(payload.get("max_slides", 0))
    window = int(payload.get("window", 8))
    stride = int(payload.get("stride", 6))
    temperature = float(payload.get("temperature", 0.0))
    policy = str(payload.get("policy", "code"))
    compare_last = bool(payload.get("compare_last", False))
    include_kept_concepts = bool(payload.get("include_kept_concepts", True))
    concept_context = int(payload.get("concept_context", 20))
    pane_dedupe = bool(payload.get("pane_dedupe", False))
    pane_side = str(payload.get("pane_side", "right"))
    pane_frac = float(payload.get("pane_frac", 0.35))
    pane_threshold = int(payload.get("pane_threshold", 2))
    extra_file = payload.get("extra_file")
    start = int(payload.get("start", 0))
    end = int(payload.get("end", 0))

    pages_dir = slides_dir / "pages"
    index_csv = slides_dir / "index.csv"
    out_json = slides_dir / "significant.json"
    out_csv = slides_dir / "significant.csv"
    out_pdf = slides_dir / "curated.pdf"

    if not pages_dir.is_dir():
        raise FileNotFoundError(f"Missing pages directory: {pages_dir}")
    if not index_csv.is_file():
        raise FileNotFoundError(f"Missing index.csv: {index_csv}")

    df = pd.read_csv(index_csv)
    # Normalize to a column that contains actual PNG filenames under pages/
    col = None
    if "page" in df.columns and df["page"].astype(str).str.endswith(".png").any():
        col = "page"
    elif "file" in df.columns:
        col = "file"
        # If a numeric page column already exists, preserve it as page_num
        if "page" in df.columns and not df["page"].astype(str).str.endswith(".png").any():
            df = df.rename(columns={"page": "page_num"})
        df = df.rename(columns={"file": "page"})
    else:
        for alt in ("filename", "name"):
            if alt in df.columns:
                col = alt
                df = df.rename(columns={alt: "page"})
                break
    if col is None:
        raise ValueError("index.csv must contain a filename column (e.g., 'file' or 'page') matching PNGs in pages/")

    # Optional subrange selection (by page_num if present, else by 1-based row index)
    if start or end:
        if "page_num" in df.columns:
            if start:
                df = df[df["page_num"] >= start]
            if end:
                df = df[df["page_num"] <= end]
        else:
            total_rows = len(df)
            lo = max(1, start) if start else 1
            hi = min(end, total_rows) if end else total_rows
            df = df.iloc[lo-1:hi]

    files = [pages_dir / p for p in df["page"].map(str).tolist()]
    total = len(files)

    # Load optional per-deck selection hint once
    extra_hint = _load_selection_hint(slides_dir, extra_file=extra_file)

    final: List[SlideDecision] = []
    if compare_last:
        # Sequential mode: bootstrap first slide as kept to establish a reference
        last_kept_idx: Optional[int] = None
        last_kept_path: Optional[Path] = None
        kept_concepts: List[str] = []
        for i, cur in enumerate(files):
            if i == 0:
                # Keep first slide as an anchor
                d = SlideDecision(page=cur.name, score=1.0, keep=True, reason="bootstrap first keep", concept="intro")
                final.append(d)
                last_kept_idx, last_kept_path = i, cur
                kept_concepts.append(d.concept or "intro")
                continue
            # Optional prefilter: if the designated pane region is unchanged vs last kept, skip without calling the model
            if pane_dedupe and last_kept_path is not None:
                try:
                    h_last = _pane_region_hash(last_kept_path, pane_side, pane_frac)
                    h_cur = _pane_region_hash(cur, pane_side, pane_frac)
                    if (h_cur - h_last) <= pane_threshold:
                        final.append(SlideDecision(page=cur.name, score=0.0, keep=False, reason="pane-unchanged prefilter", concept=""))
                        continue
                except Exception:
                    # If hashing fails, fall back to model decision
                    pass

            dec = _score_against_last(
                model,
                (i, cur),
                (last_kept_idx, last_kept_path) if last_kept_path is not None else None,
                temperature,
                policy,
                extra_hint,
                kept_concepts=kept_concepts,
                concept_context=concept_context,
                include_kept_concepts=include_kept_concepts,
            )
            # Apply threshold semantics
            keep_flag = dec.keep if threshold >= 1.0 - 1e-9 else (dec.keep or dec.score >= threshold)
            final.append(SlideDecision(page=cur.name, score=dec.score, keep=keep_flag, reason=dec.reason, concept=dec.concept))
            if keep_flag:
                last_kept_idx, last_kept_path = i, cur
                if dec.concept:
                    kept_concepts.append(dec.concept)
    else:
        # Model-only scoring over overlapping windows (original behavior)
        all_scores: Dict[str, List[SlideDecision]] = {f.name: [] for f in files}

        def make_windows(n: int, w: int, s: int):
            i = 0
            while i < n:
                yield list(range(i, min(i+w, n)))
                if i + w >= n:
                    break
                i += s

        if stride >= window:
            try:
                print(f"[slide_extract_tool] Warning: stride ({stride}) >= window ({window}); no overlap.\n"
                      f"Consider stride < window (e.g., window={window}, stride={max(1, window-2)}) for boundary stability.")
            except Exception:
                pass

        for idxs in make_windows(total, window, stride):
            window_items = [(i, files[i]) for i in idxs]
            decisions = _score_window(model, window_items, temperature, policy, extra_hint)
            # assign back by filename
            dec_by_page = {d.page: d for d in decisions}
            for i in idxs:
                p = files[i].name
                d = dec_by_page.get(p)
                if d:
                    all_scores[p].append(d)

        # Reduce multiple window decisions per slide: take max score and OR keep flags
        for f in files:
            arr = all_scores.get(f.name, [])
            if not arr:
                final.append(SlideDecision(page=f.name, score=0.0, keep=False, reason="no-decision", concept=""))
            else:
                best = max(arr, key=lambda x: x.score)
                if threshold >= 1.0 - 1e-9:
                    keep_flag = any(x.keep for x in arr)
                else:
                    keep_flag = any(x.keep for x in arr) or (best.score >= threshold)
                final.append(SlideDecision(page=f.name, score=best.score, keep=keep_flag, reason=best.reason, concept=best.concept))

    # Apply optional cap
    kept = [d for d in final if d.keep]
    if max_slides and len(kept) > max_slides:
        kept_sorted = sorted(kept, key=lambda x: x.score, reverse=True)[:max_slides]
        keep_pages = {d.page for d in kept_sorted}
        final = [SlideDecision(d.page, d.score, d.page in keep_pages, d.reason) for d in final]

    # Write JSON
    out_json_data = {"model": model, "threshold": threshold, "policy": policy,
                     "window": window, "stride": stride,
                     "results": [asdict(d) for d in final]}
    out_json.write_text(json.dumps(out_json_data, indent=2), encoding="utf-8")

    # Write CSV (merge)
    score_df = pd.DataFrame([asdict(d) for d in final])
    merged = df.merge(score_df, on="page", how="left")
    merged.to_csv(out_csv, index=False)

    # Build curated PDF
    keep_paths = [pages_dir / d.page for d in final if d.keep]
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert([str(p) for p in keep_paths]))

    return RunResult(
        kept_count=len(keep_paths),
        total=total,
        output_json=str(out_json),
        output_csv=str(out_csv),
        output_pdf=str(out_pdf),
    )


# --- CLI wrapper ---

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=TOOL_SCHEMA["description"])
    ap.add_argument("slides_dir", help="Path like slides/<base>")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--max-slides", type=int, default=0, help="0 = no cap")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--policy", choices=["lecture","code","math","doc"], default="code")
    ap.add_argument("--compare-last", action="store_true", help="Sequential mode: compare current to last kept (no windows)")
    ap.add_argument("--include-kept-concepts", action="store_true", help="Include running list of kept concepts in the prompt (default on)")
    ap.add_argument("--no-include-kept-concepts", dest="include_kept_concepts", action="store_false")
    ap.add_argument("--concept-context", type=int, default=20, help="Max recent kept concepts to include (default 20)")
    ap.add_argument("--pane-dedupe", action="store_true", help="Drop frames when output pane region unchanged vs last kept before calling the model")
    ap.add_argument("--pane-side", choices=["right","left","bottom","top"], default="right")
    ap.add_argument("--pane-frac", type=float, default=0.35)
    ap.add_argument("--pane-threshold", type=int, default=2)
    ap.add_argument("--extra-file", default=None, help="Optional path to per-deck selection hint text")
    ap.add_argument("--start", type=int, default=0, help="Start page index (1-based) or page_num")
    ap.add_argument("--end", type=int, default=0, help="End page index (1-based) or page_num")
    args = ap.parse_args(argv)

    res = run_tool({
        "slides_dir": args.slides_dir,
        "model": args.model,
        "threshold": args.threshold,
        "max_slides": args.max_slides,
        "window": args.window,
        "stride": args.stride,
        "temperature": args.temperature,
        "policy": args.policy,
        "compare_last": args.compare_last,
        "include_kept_concepts": args.include_kept_concepts,
        "concept_context": args.concept_context,
        "pane_dedupe": args.pane_dedupe,
        "pane_side": args.pane_side,
        "pane_frac": args.pane_frac,
        "pane_threshold": args.pane_threshold,
        "start": args.start,
        "end": args.end,
        "extra_file": args.extra_file,
    })

    print(json.dumps(asdict(res), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
