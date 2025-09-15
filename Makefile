.PHONY: a11y-ocr a11y-gen a11y-gen-all combine-all combine-ocr select-one tag-json
.PHONY: select-week combine-week ocr-week
.PHONY: describe select-from-descriptions

# Usage examples:
#   make a11y-gen BASE=1.2_03_errors_exceptions MODEL=gpt-4o-mini
#   make a11y-ocr BASE=1.2_03_errors_exceptions
#   make a11y-gen-all MODEL=gpt-4o-mini
#   make combine-all

SLIDES_DIR ?= slides
MODEL ?= gpt-4o-mini
POLICY ?= code
WINDOW ?= 8
STRIDE ?= 3

a11y-gen:
	# Generate a11y.json via LLM for one deck (requires OPENAI_API_KEY)
	conda run -n slides-ocr python tools/generate_a11y_metadata.py $(SLIDES_DIR)/$(BASE) --model $(MODEL) --batch-size 10

a11y-ocr:
	# OCR curated.pdf to curated_ocr.pdf for one deck
	conda run -n slides-ocr python tools/ocr_only.py $(SLIDES_DIR)/$(BASE)

a11y-gen-all:
	# Generate a11y.json for all decks (requires OPENAI_API_KEY); continues on errors
	@for d in $(SLIDES_DIR)/*; do \
	  if [ -d "$$d" ] && [ -f "$$d/curated.pdf" ]; then \
	    echo "Generating a11y for $$d"; \
	    conda run -n slides-ocr python tools/generate_a11y_metadata.py "$$d" --model $(MODEL) --batch-size 10 || true; \
	  fi; \
	done


combine-all:
	# Build combined curated PDF (from kept pages)
	python tools/combine_curated.py $(SLIDES_DIR)/combined_curated.pdf

combine-ocr:
	# OCR combined curated PDF
	conda run -n slides-ocr python tools/ocr_only.py $(SLIDES_DIR) --combined

# Step 2: Select significant slides for one deck
select-one:
	@if [ -z "$(BASE)" ]; then echo "Please set BASE=slides subdir (e.g., 1.2_03_errors_exceptions)"; exit 2; fi
	@echo "Selecting significant slides for $(SLIDES_DIR)/$(BASE) using $(MODEL)"
	OPENAI_API_KEY=$${OPENAI_API_KEY:?Set OPENAI_API_KEY} python tools/slide_extract_tool.py $(SLIDES_DIR)/$(BASE) --model $(MODEL) --threshold 1.0 --window $(WINDOW) --stride $(STRIDE) --policy $(POLICY)

# New: Describe slides (VLM) into slide_descriptions.json
describe:
	@if [ -z "$(BASE)" ]; then echo "Please set BASE=slides subdir (e.g., 2.1_09_dunder_methods)"; exit 2; fi
	OPENAI_API_KEY=$${OPENAI_API_KEY:?Set OPENAI_API_KEY} python tools/describe_slides.py $(SLIDES_DIR)/$(BASE) --model $(MODEL)

# New: Select from descriptions (text-only or deterministic rules)
select-from-descriptions:
	@if [ -z "$(BASE)" ]; then echo "Please set BASE=slides subdir (e.g., 2.1_09_dunder_methods)"; exit 2; fi
	# Use TEXT_MODEL if set; default falls back to $(MODEL)
	OPENAI_API_KEY=$${OPENAI_API_KEY:?Set OPENAI_API_KEY} python tools/select_from_descriptions.py $(SLIDES_DIR)/$(BASE) --text-model $${TEXT_MODEL:-$(MODEL)} --budget $${BUDGET:-8} $${RULES_ONLY:+--rules-only}

# Tag from JSON (Step 3 optional tagging)
tag-json:
	@if [ -z "$(BASE)" ]; then echo "Please set BASE=slides subdir (e.g., 1.2_03_errors_exceptions)"; exit 2; fi
	@if [ -z "$(TITLE)" ]; then echo "Please set TITLE=\"Document Title\""; exit 2; fi
	conda run -n slides-ocr python tools/tag_from_json.py $(SLIDES_DIR)/$(BASE)/curated_ocr.pdf $(SLIDES_DIR)/$(BASE)/a11y.json --out $(SLIDES_DIR)/$(BASE)/curated_accessible.pdf --title "$(TITLE)"

# Step 2 for an entire week folder (videos/<WEEK> order)
select-week:
	@if [ -z "$(WEEK)" ]; then echo "Please set WEEK=week2 (folder under videos/)"; exit 2; fi
	@echo "Selecting across videos/$(WEEK) in video filename order"
	@for v in $$(ls -1 videos/$(WEEK)/*.mp4 | sort); do \
	  b=$$(basename $$v .mp4); \
	  echo "==> $$b"; \
	  OPENAI_API_KEY=$${OPENAI_API_KEY:?Set OPENAI_API_KEY} conda run -n slides-ocr python tools/slide_extract_tool.py $(SLIDES_DIR)/$(WEEK)/$$b --model $(MODEL) --threshold 1.0 --window $(WINDOW) --stride $(STRIDE) --policy $(POLICY) || exit $$?; \
	done

# Combine curated PDFs for a week (must exist already)
combine-week:
	@if [ -z "$(WEEK)" ]; then echo "Please set WEEK=week2 (folder under slides/)"; exit 2; fi
	@echo "Combining curated PDFs for slides/$(WEEK) into slides/$(WEEK)/curated.pdf"
	conda run -n slides-ocr bash -lc 'pdfunite $$(for p in videos/$(WEEK)/*.mp4; do b=$${p##*/}; b=$${b%.mp4}; echo slides/$(WEEK)/$$b/curated.pdf; done) slides/$(WEEK)/curated.pdf'

# OCR the combined week deck to curated_ocr.pdf (no rotation by default)
ocr-week:
	@if [ -z "$(WEEK)" ]; then echo "Please set WEEK=week2 (folder under slides/)"; exit 2; fi
	@echo "OCR slides/$(WEEK)/curated.pdf -> slides/$(WEEK)/curated_ocr.pdf (no rotation)"
	conda run -n slides-ocr python tools/ocr_only.py slides/$(WEEK) --combined || cp slides/$(WEEK)/curated.pdf slides/$(WEEK)/curated_ocr.pdf
