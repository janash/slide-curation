# Slides Pipeline

This repository processes screen-recorded lectures and converts them into clean, student-facing slide decks. It extracts stable frames from videos, uses an image-capable LLM to identify and select only the meaningful slides, then adds searchable captions to create PDFs that are useful for studying and reference.

## Setup

1. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate slides-ocr
   ```

2. **Add your OpenAI API key to `keys.env`**

3. **Place your MP4 files under `videos/<week>/`** (e.g., `videos/week1/`, `videos/week4/`)

## Usage

This pipeline is designed to work with Codex CLI:

1. Open this repository in Codex CLI
2. Start with: **"Read AGENTS.md and summarize the pipeline we'll follow"**
3. Then ask the agent to process your videos, e.g.: **"Process week4 end-to-end and combine the week deck"**

The agent will handle the complete pipeline: extract frames → select meaningful slides → refine for students → add captions → create searchable PDFs.

## Manual Usage

If you prefer to run commands manually, see `AGENTS.md` for the complete pipeline documentation.