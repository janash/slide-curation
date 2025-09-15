# Slides Pipeline (Human Guide)

Recommended workflow: use Codex CLI and start by having the agent read AGENTS.md.

Quick start with Codex CLI
- Open this repo in Codex CLI.
- Put your MP4s under `videos/<week>/` (e.g., `videos/week4/…`).
- First prompt: “Read AGENTS.md and summarize the pipeline we’ll follow.”
- Then ask: “Process week4 end-to-end (extract → select → Pass 2 → annotate) and combine the week deck.”
- The agent will use the standardized flags and produce outputs under `slides/<week>/…`.

Conda environment
- Create once: `conda env create -f environment.yml`
- Activate: `conda activate slides-ocr`
- The agent and Make targets assume the environment name is `slides-ocr`.

This repo builds concise, student-facing slide decks from screen-recorded videos in a reproducible way.

Outcome per video
- curated.pdf: Pass 1 keeps (LLM selection)
- curated_v2.pdf: Pass 2 refined keeps (student-facing)
- captions.json: ordered student_concept captions
- curated_annotated_from_captions.pdf: annotated, searchable PDF (final per-video)

Outcome per week
- slides/<week>/curated_annotated_from_captions.pdf: combined annotated deck from all videos in that week

Non-negotiable
- Never combine Step 1 PDFs. Always run Step 2 (LLM selection) first.

Project layout (inputs → outputs)
- Inputs: place videos in `videos/<week>/` (e.g., `videos/week1`, `videos/week4`).
- Outputs: tools write to `slides/<week>/<base>/…` and `slides/<week>/<base>.pdf`, with a combined week deck at `slides/<week>/curated_annotated_from_captions.pdf`.

## Prerequisites

- Conda environment: `slides-ocr` (see environment.yml) with ffmpeg, poppler (pdf utilities), pikepdf, ocrmypdf, Pillow, pandas, reportlab
- OpenAI key in `keys.env`
  - Bash/zsh: `export OPENAI_API_KEY=$(cat keys.env)` (or `set -a; source keys.env; set +a` if it contains `OPENAI_API_KEY=...`)

## TL;DR (Per Video)

1) Step 1 — Extract pages (no LLM)
- `conda run -n slides-ocr python tools/extract_slides.py "<video>.mp4" --scene 0.25 --fps 1.0 --hash 4 --min-gap 2.0 --max-candidates 150 --outdir slides`
- Produces `slides/<base>/pages/`, `slides/<base>/index.csv`, and `slides/<label>/<base>.pdf`

2) Step 2 — Select (LLM, compare-last)
- `OPENAI_API_KEY=… conda run -n slides-ocr python tools/slide_extract_tool.py slides/<base> --model gpt-4o --threshold 1.0 --policy code --compare-last --include-kept-concepts`
- Produces `significant.json/csv` and `curated.pdf`

3) Pass 2 — Kept review (student-facing)
- `OPENAI_API_KEY=… LECTURE_REVIEW_PROMPT=prompts/kept_review.md conda run -n slides-ocr python tools/review_kept_slides.py slides/<base> --model gpt-4o --window 8 --stride 3 --threshold 1.0`
- Produces `significant_v2.json/csv` (adds `keep2`, `student_concept`) and `curated_v2.pdf`

4) Captions → Annotate (searchable text below image)
- `conda run -n slides-ocr python tools/export_captions.py slides/<base> --field student_concept --out captions.json`
- `conda run -n slides-ocr python tools/annotate_from_captions.py slides/<base>/captions.json --out curated_annotated_from_captions.pdf`

5) Combine (week)
- `pdfunite slides/<week>/*/curated_annotated_from_captions.pdf slides/<week>/curated_annotated_from_captions.pdf`
- No pdfunite? Use pikepdf in the env (see Combine below).

## Naming Conventions

- Clean names always: lowercase, underscores, collapse repeats; remove vendor/junk suffixes, parentheses, nested extensions; preserve numeric prefixes (e.g., `4.1`).
- Zero-pad single-digit segments (`_1_` → `_01_`). Apply to per-video folders and Step 1 PDFs.
- Module vs video: a module (e.g., `4.1_introspection`) may contain multiple videos (e.g., `_01_overview`, `_02_builtins_help_type`, `_03_builtins_isinstance_dir`). Keep per-video outputs in their own folders. You may also create an optional module-level combined annotated deck at `slides/<week>/<module>/curated_annotated_from_captions.pdf`.

## Recommended Defaults

- Step 2: `--policy code --compare-last --include-kept-concepts --threshold 1.0` (explicit keeps-only)
- Optional: `--pane-dedupe --pane-side right --pane-frac 0.35 --pane-threshold 2` for split-view terminals
- No need for windowed selection when using compare-last. For legacy windowing, use overlap (`--window 8 --stride 3/6`).

## Combine (Fallback without pdfunite)

- Combine module (example):
  - `conda run -n slides-ocr python - <<'PY'`
  - `from pikepdf import Pdf; import glob; w=Pdf.new();`
  - `parts=sorted(glob.glob('slides/week4/4.1_introspection_*/curated_annotated_from_captions.pdf'))`
  - `[w.pages.extend(Pdf.open(p).pages) for p in parts]; w.save('slides/week4/4.1_introspection/curated_annotated_from_captions.pdf')`
  - `PY`

- Combine week (avoid module duplicates):
  - `conda run -n slides-ocr python - <<'PY'`
  - `from pikepdf import Pdf; import glob; w=Pdf.new();`
  - `parts=sorted(glob.glob('slides/week4/*/curated_annotated_from_captions.pdf'))`
  - `[w.pages.extend(Pdf.open(p).pages) for p in parts if not p.endswith('/4.1_introspection/curated_annotated_from_captions.pdf')]`
  - `w.save('slides/week4/curated_annotated_from_captions.pdf')`
  - `PY`

## What the folders mean (Step 1)

- `candidates/`: sparse scene-cut candidates (often 1–3 for screen recordings)
- `dense/`: uniform 1 fps sample (coverage)
- `pages/`: deduped kept frames (perceptual hash + min gap). Step 2 reads this.
- Logs: `ffmpeg_showinfo.log` (scene), `ffmpeg_dense_showinfo.log` (dense)

## Why two passes?

- Pass 1 (selection): decide keep/drop efficiently with compare-last to reduce near duplicates; produce curated.pdf and brief reasons.
- Pass 2 (kept review): use forward/backward context to refine keeps, prefer final states, generate student-facing `student_concept`; produce curated_v2.pdf.

## Batch (Recommended for a full week)

- `tools/batch_week.sh week4`
  - Runs extract → select → Pass 2 → captions → annotate for every video in the week and combines the week deck.
  - Resumable: skips videos that already have `curated_annotated_from_captions.pdf`.

## Troubleshooting

- Extra slides sneaking in: ensure `--threshold 1.0` (explicit keeps-only).
- Missing rows in JSON: `index.csv` must have filename column matching `pages/*.png`.
- Combine tool missing: use the pikepdf fallback snippets above.
- Long runs: re-run per-video steps; tools are idempotent and overwrite outputs safely.
- OCR: `tools/ocr_only.py` disables rotation by default; don’t rotate code slides.

## Make/Utilities

- `make select-one BASE=<base> MODEL=gpt-4o` — runs Step 2 for a single deck (for compare-last flags, run the script directly as shown above).

---
If you’re unsure where to start: run the TL;DR commands for one video, check `curated_annotated_from_captions.pdf`, then scale up with `tools/batch_week.sh <week>`.
