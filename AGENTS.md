# Repository Guidelines (Updated)

Goal: Reproducible pipeline — Step 1 extraction (no LLM) → Step 2 selection (LLM) → Pass 2 review (student‑facing concepts) → Annotate (searchable) → Combine. Run per‑video or per‑folder.

Non‑negotiable protocol
- ALWAYS run Step 2 (LLM curation) before any week‑level combine or OCR. Never combine Step 1 PDFs directly — that wastes time and bloats the deck.

Current default practice
- Preferred selection: policy=code with compare‑last (`--compare-last --include-kept-concepts`) and `--threshold 1.0` for explicit keeps‑only. This reduces near‑duplicates by comparing against the last kept slide and tracking already‑kept concepts.
- Optional for split‑view: add pane prefilter to skip frames where the output/terminal pane is unchanged (`--pane-dedupe --pane-side right --pane-frac 0.35 --pane-threshold 2`).
- Documentation‑centric decks (e.g., writing docstrings): use policy=doc plus a per‑deck selection hint to keep only finished docstrings and section/summary slides, avoiding typing/fades. Cap to ~8–10 slides where appropriate.
<!-- Accessibility/tagging content removed for now -->

## Quick Checklist (Per Video / Folder)

- Extract: `conda run -n slides-ocr python tools/extract_slides.py "<video>.mp4" --scene 0.25 --fps 1.0 --hash 4 --min-gap 2.0 --max-candidates 150 --outdir slides`
- Verify: `slides/<base>/pages/` and `slides/<base>/index.csv` look correct.
- Select (required): `OPENAI_API_KEY=… conda run -n slides-ocr python tools/slide_extract_tool.py slides/<base> --model gpt-4o --threshold 1.0 --policy code --compare-last --include-kept-concepts`
- Inspect: `slides/<base>/curated.pdf` and `slides/<base>/significant.csv`.
- Pass 2 (required): `OPENAI_API_KEY=… LECTURE_REVIEW_PROMPT=prompts/kept_review.md conda run -n slides-ocr python tools/review_kept_slides.py slides/<base> --model gpt-4o --window 8 --stride 3 --threshold 1.0` → `curated_v2.pdf`
- Export captions (ordered, Pass 2 keeps only):
  - `conda run -n slides-ocr python tools/export_captions.py slides/<base> --field student_concept --out captions.json`
- Annotate from captions (searchable text, below image):
  - `conda run -n slides-ocr python tools/annotate_from_captions.py slides/<base>/captions.json --out curated_annotated_from_captions.pdf`
- Combine (module, optional if module spans multiple videos):
  - `pdfunite slides/<week>/<module>_*/curated_annotated_from_captions.pdf slides/<week>/<module>/curated_annotated_from_captions.pdf`
  - If `pdfunite` is unavailable, use pikepdf (see Combine guidance below).
- Combine (week):
  - `pdfunite slides/<week>/*/curated_annotated_from_captions.pdf slides/<week>/curated_annotated_from_captions.pdf`

Outputs
- Per‑video: `slides/<base>/curated.pdf`, `slides/<base>/curated_v2.pdf`, `slides/<base>/captions.json`, `slides/<base>/curated_annotated_from_captions.pdf`
- Per‑week (combined annotated): `slides/<week>/curated_annotated_from_captions.pdf`

## Step 1 — Extraction (no LLM)

- `conda run -n slides-ocr python tools/extract_slides.py "<video>.mp4" --scene 0.25 --fps 1.0 --hash 4 --min-gap 2.0 --max-candidates 150 --outdir slides`
- Produces: `slides/<base>/pages/`, `slides/<base>/index.csv`, and `<base>.pdf` (kept pages). First frame always; last only if not duplicate.

## Step 2 — Selection (LLM)

- Prompt (default): `prompts/lecture_extraction.md` (set `LECTURE_EXTRACTION_PROMPT=...` to override)
- Run (preferred): `conda run -n slides-ocr python tools/slide_extract_tool.py slides/<base> --model gpt-4o --threshold 1.0 --policy code --compare-last --include-kept-concepts`
- For noisier code sessions: `--policy code_strict` to prefer only finished code (no mid-typing/partials/autocomplete/scroll noise), often with `--stride 3` and optional `--max-slides N`.
- Produces: `significant.json`, `significant.csv`, `curated.pdf`
- Batching: use `--start/--end` for long decks.
- In general, you should always use `--policy code`.

Preferred: duplicate‑resistant compare‑last mode
- Purpose: reduce near‑duplicate keeps when output/error pane stays static while code pane scrolls/edits.
- How it works: the model sees the LAST KEPT slide plus the CURRENT candidate and decides if the current adds meaningful new information. The prompt also includes a short running list of concepts already kept to avoid redundancy.
- Recommended run (code decks):
  - `OPENAI_API_KEY=… LECTURE_EXTRACTION_PROMPT=prompts/lecture_extraction.md \
     conda run -n slides-ocr python tools/slide_extract_tool.py slides/<base> --model gpt-4o --threshold 1.0 --policy code \
     --compare-last --include-kept-concepts`
- Optional prefilter for split‑view terminals: skip frames whose output pane is unchanged vs last kept:
  - Add `--pane-dedupe --pane-side right --pane-frac 0.35 --pane-threshold 2` (adjust side/frac if your layout differs).
- Notes:
  - Rolling windows (`--window/--stride`) remain supported but are not required when using `--compare-last`.
  - `selection_hint.md` is still honored and appended to the prompt.

Enhanced selection controls (what we learned)
- New policy=doc: optimized for documentation lectures that include code, but should not keep typing/scroll frames. Prefers final, readable code/docstrings and section/summary slides; avoids partial/in‑progress content.
- New policy=code_strict: for code-heavy lectures that must exclude incomplete code. Keeps only finished statements/blocks with balanced quotes/brackets; rejects partial tokens (e.g., 'qui', 'prin'), caret/cursor/selection frames, autocomplete popups, and tiny scroll changes.
- Per‑deck selection hints: The selector will append `slides/<base>/selection_hint.md` (if present) to the prompt. Use it to specify lecture intent. Example contents for docstrings:
  - “Keep only slides with finished Python docstrings where both opening and closing triple quotes are visible; avoid typing/caret frames, partial docstrings, fades/transitions, and tiny scrolls. Prefer the clearest final instance. Budget ~10 slides.”
- Explicit keeps‑only: With `--threshold 1.0`, the tool now treats the run as “explicit keeps only,” ignoring score pass‑through. A page is kept only if the model sets `keep=true`.
- Overlap: For dense content, use `--stride 3`. For a lighter pass (to reduce edge over‑keeping), `--stride 6` can help. Always keep `stride < window`.
- Hard cap: Use `--max-slides N` to cap final keeps (e.g., 10) without a second pass.

Hard rule
- Do not build a week/course combined deck from Step 1 PDFs. First produce `slides/<base>/curated.pdf` (Step 2), then combine curated PDFs and OCR the combined result.

### significant.json and significant.csv (contract and behavior)

- Schema (JSON): top-level metadata plus `results[]` in slide order
  - `{ model, threshold, policy, window, stride, results: [{ page, score, keep, reason }] }`
  - `page`: must match an actual filename under `pages/` (e.g., `0003_0:02:47.png`).
  - `score`: float in [0,1]; `keep`: boolean; `reason`: short rationale (<= ~160 chars).
- Schema (CSV): `index.csv` augmented by merge on page filename
  - Input may provide `page` or `file`; the tool normalizes to `page`.
  - If `index.csv` has a numeric `page` column, it is preserved as `page_num` to avoid collision.
- Reduction with overlapping windows
  - The model scores slides in overlapping windows (`--window`, `--stride`).
  - Multiple decisions per page are reduced to a single row by: `score = max(scores)` and
    `keep = any(dec.keep)` when `threshold >= 1.0` (explicit keeps‑only), otherwise `keep = any(dec.keep) OR (max(score) >= threshold)`.
  - For “explicit keeps only,” use `--threshold 1.0` (recommended). Lower thresholds will keep high‑score frames even if the model’s `keep` was false.
- Range selection semantics
  - `--start/--end` use `page_num` if present; otherwise they use 1‑based row indices from `index.csv`.
  - When batching long decks, run multiple ranges and let the tool overwrite/append to the same outputs.
- Determinism and retry
  - Defaults to `temperature=0.0`; responses are JSON‑formatted (`response_format={type: json_object}`).
  - If the model returns invalid JSON, the tool attempts minimal recovery; rerun if needed.
- Editing after the fact
  - You can safely hand‑edit `significant.csv` and flip `keep` to trim or add slides.
  - To rebuild `curated.pdf` after edits, rerun the same `slide_extract_tool.py` command for that deck (it will re‑merge and re‑emit `curated.pdf`).
  <!-- Accessibility pipeline references removed -->
  - Tools (no LLM) to enforce limits or rebuild:
    - `tools/rebuild_curated_from_significant.py slides/<base>`: rebuilds curated.pdf from existing significant.json.
    - `tools/cap_kept_slides.py slides/<base> --max N`: caps keeps to top‑N by score, syncs JSON/CSV, rebuilds curated.pdf.
- Troubleshooting
  - Unexpected extra slides: ensure `--threshold 1.0` for explicit keeps‑only; otherwise high scores can pass the threshold and be kept.
  - Missing rows in JSON: verify `index.csv` has a filename column (`page` or `file`) that matches `pages/*.png`.
  - Duplicate pages across windows: expected; the tool de‑duplicates by filename using the reduction rule above.
  - Off‑by‑one ranges: confirm whether `page_num` exists; range filters behave differently when only row indices are available.

### Windowing best practices (legacy)

If using the legacy windowed selector (without `--compare-last`):
- Use overlapping windows: set `--stride` strictly less than `--window`.
- Recommended defaults: `--window 8 --stride 6` (moderate overlap). For dense scrolls/code, increase overlap e.g., `--window 8 --stride 3`.
- Why: overlap stabilizes decisions at window boundaries; the reduction rule (max score, OR keep) smooths out edge effects.
- If you previously ran with no overlap (`stride >= window`), re-run Step 2 with overlap; it will overwrite `significant.json/.csv` and `curated.pdf` in place.
- Chunking ranges: when batching with `--start/--end`, ensure adjacent chunks overlap by at least `(window - stride)` slides to maintain cross-boundary context.

### Selection principles (conceptual)

- Prefer: title/section dividers; new concepts/definitions; key diagrams/plots; major code diffs; summaries/conclusions.
- Avoid: tiny scrolls, caret moves, minor formatting changes, duplicate frames with negligible visual change.
- Keep reasons short and specific (one clause, ≤ ~20 words).
- Treat model keep as primary signal; use `--threshold 1.0` so only explicit keeps pass. Scores are supportive, not primary.
- Ground all decisions in filenames under `pages/`; never infer by index alone.

Documentation‑heavy lectures (docstrings)
- Use `--policy doc` and provide a `selection_hint.md` requiring fully finished docstrings (both opening and closing triple quotes present) and avoiding typing/caret/fade frames.
- Consider `--stride 6` and `--max-slides 8..10` to keep the summary tight.

<!-- Removed legacy two-stage describe/select option -->

<!-- Pass 2 section moved above Step 3 -->

## Pass 2 — Kept Review (required, student-facing)

- Goal: refine the Pass 1 keep set using forward/backward context and produce concise, student-facing statements (instead of justifications).
- Prompt: `prompts/kept_review.md` (or set `LECTURE_REVIEW_PROMPT=prompts/kept_review.md`). The prompt always keeps the title and section/divider slides, prefers final over partial states, and avoids near-duplicates.
- Run:
  - `OPENAI_API_KEY=… LECTURE_REVIEW_PROMPT=prompts/kept_review.md \\
     conda run -n slides-ocr python tools/review_kept_slides.py slides/<base> --model gpt-4o --window 8 --stride 3 --threshold 1.0`
- Produces:
  - `significant_v2.json`, `significant_v2.csv` (fields include `keep2` and `student_concept`)
  - `curated_v2.pdf` (final kept set after review)

## Step 3 — Captions → Annotate (searchable)

- Export captions (ordered, Pass 2 keeps only):
  - `conda run -n slides-ocr python tools/export_captions.py slides/<base> --field student_concept --out captions.json`
- Annotate from captions (ReportLab, searchable text below image):
  - `conda run -n slides-ocr python tools/annotate_from_captions.py slides/<base>/captions.json --out curated_annotated_from_captions.pdf`
- Combine (week):
  - `pdfunite slides/<week>/*/curated_annotated_from_captions.pdf slides/<week>/curated_annotated_from_captions.pdf`
  - Avoid double counting: if you created module‑level combined PDFs (e.g., `slides/<week>/<module>/curated_annotated_from_captions.pdf`), do not also include their constituent per‑video PDFs when building the week deck.
  - Fallback when `pdfunite` is unavailable: combine with pikepdf in the conda env, for example:
    - `conda run -n slides-ocr python - <<'PY'`
    - `from pikepdf import Pdf; import glob; w=Pdf.new();`
    - `parts=sorted(glob.glob('slides/<week>/*/curated_annotated_from_captions.pdf'))`
    - `[w.pages.extend(Pdf.open(p).pages) for p in parts]; w.save('slides/<week>/curated_annotated_from_captions.pdf')`
    - `PY`

## Make Targets

- `make select-one BASE=<base> MODEL=gpt-4o`: Step 2 for a single deck (uses `OPENAI_API_KEY`). For compare‑last, run the script directly with flags shown above.

## Batch Scripts

- `tools/batch_week.sh <week>`: End‑to‑end pipeline for a week’s videos (extract → select → Pass 2 → captions → annotate) and then combine the week deck.
  - Resumable: skips videos that already have `curated_annotated_from_captions.pdf`.
  - Applies clean naming to per‑video folders and Step 1 PDFs.
  - Requires `OPENAI_API_KEY` (load from `keys.env`).

Ad‑hoc examples
- Pass 2 review: `OPENAI_API_KEY=… LECTURE_REVIEW_PROMPT=prompts/kept_review.md conda run -n slides-ocr python tools/review_kept_slides.py slides/<base> --model gpt-4o --window 8 --stride 3 --threshold 1.0`
- Annotate v2 (student‑facing, below caption): `conda run -n slides-ocr python tools/annotate_slides.py slides/<base> --source v2 --concept-field student_concept --placement below --out curated_annotated_below_v2.pdf`
- Combine annotated v2 for a week: `pdfunite slides/<week>/*/curated_annotated_below_v2.pdf slides/<week>/curated_annotated_below.pdf`

## Build / Inspect

- List: `rg --files -g "**/*.mp4"`
- ffprobe: `ffprobe -hide_banner -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate -of default=nw=1 "file.mp4"`
- Transcode: `ffmpeg -i in.mp4 -c:v libx264 -preset slow -crf 23 -c:a aac -b:a 128k out.mp4`

## Project Structure & Conventions

- Videos in repo root or `videos/<folder>/`. Outputs mirror under `slides/<folder>/`.
- Filenames: lowercase, snake_case; two‑digit sequence (`_01`). Allowed: `[a-z0-9_\.]`.
 - Clean names: strip vendor/junk suffixes (e.g., parenthetical encodings), nested extensions, and parentheses; keep only the semantic base.
 - Normalize: lowercase; use underscores; collapse repeats; preserve numeric prefixes like `4.1`.
- Zero‑pad single‑digit sequence segments to two digits (e.g., `_1_` → `_01_`, trailing `_1` → `_01`).
- Apply consistently to per‑video folder names and Step 1 PDFs before selection/Pass 2.

- Module vs video structure
  - A module (e.g., `4.1_introspection`) can have multiple videos (e.g., `4.1_introspection_01_overview`, `4.1_introspection_02_builtins_help_type`, `4.1_introspection_03_builtins_isinstance_dir`).
  - Keep per‑video outputs in their own cleaned folders; an optional module‑level combined annotated deck may live at `slides/<week>/<module>/curated_annotated_from_captions.pdf`.

## Environment & Keys

- Conda env `slides-ocr` with tesseract, poppler (pdftotext/pdfseparate/pdfunite), pikepdf, ocrmypdf, img2pdf.
- `OPENAI_API_KEY` for Step 2 (selection) and Step 3 metadata (a11y.json).
- Load API key from `keys.env` before running API-powered tools:
  - If keys.env contains `OPENAI_API_KEY=...`:
    - Bash/zsh: `set -a; source keys.env; set +a`
  - If keys.env contains only the raw key (one line):
    - Bash/zsh: `export OPENAI_API_KEY=$(cat keys.env)`
  - Fish: `for l in (cat keys.env | string match -rv '^#'); set -x (string split '=' $l); end`
  - Or inline: `OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' keys.env | cut -d= -f2-) ...`

## Quality & Commits

- Target: 1080p or 720p; stable fps (24/30). Verify audio and duration.
- Conventional Commits; one logical change per commit; include brief rationale/ffprobe lines.

## Common Notes

- Prefer per‑video loop; use `--start/--end` to batch long decks.
- Combined decks are optional; current repo focuses on per‑video outputs.

OCR rotation policy (important)
- We never auto‑rotate pages during OCR because these are screen recordings. Rotation can misfire on code or diagram slides and produce sideways pages.
- The `tools/ocr_only.py` tool disables rotation by default. Use `--rotate` only if you explicitly need it (not recommended).

Troubleshooting & caveats (recent)
- If selection over‑keeps typing/scroll frames in documentation lectures, use `policy=doc`, add a `selection_hint.md` requiring closed triple quotes, set `--threshold 1.0`, and optionally `--max-slides`.
- If Acrobat reports structural errors with experimental tagging, fall back to headers‑only which only sets document‑level metadata and is stable across viewers.
- `generate_a11y_metadata.py` now recovers from minor JSON output deviations; if a batch fails, re‑run just that deck.

## Lessons Learned / Field Notes

- Conda env: always run tools with `conda run -n slides-ocr ...` to ensure `ffmpeg`, `ocrmypdf`, `Pillow`, `pandas`, and dependencies are available.
- Week grouping: for a weekly batch, pass `--outdir slides/week1` to Step 1 so outputs land in `slides/week1/<base>/...`.
- Step 1 PDF location: the extractor writes a kept-pages PDF as `slides/<label>/<base>.pdf` (sibling to `<base>/`), while later steps write into `slides/<label>/<base>/` (e.g., `curated.pdf`). This is expected.
- Keys file: current `keys.env` can contain only the raw key (one line). Load with `export OPENAI_API_KEY=$(cat keys.env)`; avoid printing or committing the key.
- Step 2 execution (preferred overall):
  - `OPENAI_API_KEY=… LECTURE_EXTRACTION_PROMPT=prompts/lecture_extraction_2.md \
     conda run -n slides-ocr python tools/slide_extract_tool.py slides/<label>/<base> --model gpt-4o --threshold 1.0 --policy code --compare-last --include-kept-concepts`
  - Optional split‑view prefilter: append `--pane-dedupe --pane-side right --pane-frac 0.35 --pane-threshold 2`.
  - Outputs: `significant.json`, `significant.csv`, `curated.pdf` under the per‑video folder.
  - For documentation decks, consider `--policy doc` and an intent `selection_hint.md`.
- Verifying Step 2: `rg --files -g "slides/<label>/**/significant.*"` and `rg --files -g "slides/<label>/**/curated.pdf"` should show one set per deck.
- OCR at scale: `conda run -n slides-ocr python tools/ocr_only.py slides/<label> --all` creates `curated_ocr.pdf` inside each `slides/<label>/<base>/`. The command is idempotent; rerunning replaces or reuses outputs.
- Makefile caveat: `combine-all` references `tools/combine_curated.py`, which may be absent. Prefer per‑video flow and week‑level outputs under `slides/<label>/` until that script is available (see `backup/tools/a11y/*` for references).
- Long runs and sandboxes: long `conda run` operations can exceed interactive timeouts; check for created files (`rg`/`ls`) before retrying. Per‑deck re-runs are safe.
- Parameter sanity: the Quick Checklist values worked well for week1 (`--scene 0.25 --fps 1.0 --hash 4 --min-gap 2.0 --max-candidates 150`). Adjust `--threshold` in Step 2 if selection is too sparse/dense.

- Module aggregation: when a module spans multiple videos (e.g., `4.1_*`), you may create a module‑level annotated deck by combining the per‑video `curated_annotated_from_captions.pdf` files into `slides/<week>/<module>/curated_annotated_from_captions.pdf`.
- Week aggregation: build `slides/<week>/curated_annotated_from_captions.pdf` from per‑video annotated PDFs; exclude module‑level combined PDFs to avoid duplicates.
- If `pdfunite` is missing, combine with pikepdf inside the `slides-ocr` env (see Step 3 section for an example snippet).
