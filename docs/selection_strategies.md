# Selection Hints & Custom Pass 2 Prompts

Not all videos are code walkthroughs. Product demos, creative apps (Blender, Photoshop), or craft tutorials (origami) benefit from tighter guidance so the LLM keeps actionable frames and writes useful captions. This note explains how to tailor selection hints (Step 2) and Pass 2 prompts for those scenarios.

## Selection Hint Playbook

Create `slides/<week>/<base>/selection_hint.md` *before* running Step 2. The file is appended to the system prompt, so use concise imperative language.

Guidelines:

- **Describe the target actions**: call out what qualifies as a “distinct step” (e.g., new UI panel opened, fold completed, render preview updated).
- **Prioritize visual clarity**: mention that the slide should show visible controls, buttons, crease directions, or reference edges—whatever the learner needs to replicate the action.
- **Cull duplicates**: explicitly tell the model to drop frames where only the cursor or hands shift slightly and nothing new happens.
- **Name directions/axes**: for physical demos, request mention of “left/right flap,” “mountain vs. valley fold,” rotation, etc., so the model knows these cues matter.

Example (Blender):
```
Focus on clear Blender workflow milestones for 2D→3D conversion. Keep slides where the UI visibly changes to show a new tool, modifier, panel, or render result. Prefer frames that reveal which buttons, menu paths, or hotkeys to use (e.g., clicking Image → Open, enabling Grease Pencil modifiers, starting renders). Drop tiny cursor moves, repeated static views, or frames that do not convey a distinct action the student must perform.
```

Example (Origami):
```
Keep slides that show a distinct origami fold state or action: aligning corners, reversing creases, opening pockets, forming 3D walls, or showing the finished hexagon. Prefer frames where the instructor’s hands clearly demonstrate what to do and where reference edges/creases are visible. Drop frames where nothing new happened (minor hand motion, same crease slowly tightening, idle narration). Prioritize steps that mention direction (mountain vs. valley), rotation, or which flap to lift so students can follow along.
```

## Dedicated Pass 2 Prompts

The default `prompts/kept_review.md` optimizes for programming explanations. Clone it when you need action-oriented captions:

1. Copy the prompt, e.g. `cp prompts/kept_review.md prompts/kept_review_blender.md`.
2. Replace the instructions so `student_concept` sentences describe exact steps (“Click Add > Image > Reference…”) or physical motions (“Fold the right flap diagonally…”).
3. Point Pass 2 at the new file:  
   ```bash
   OPENAI_API_KEY=$(cat keys.env) \
   LECTURE_REVIEW_PROMPT=prompts/kept_review_blender.md \
   conda run -n slides-ocr python tools/review_kept_slides.py ... 
   ```
4. Export captions after Pass 2 so the annotated deck inherits the richer instructions.

### Naming Variant Decks

When you want to compare policies (e.g., code vs. doc), duplicate the Step 1 folder before rerunning Step 2:

```bash
cp -R slides/week_misc/demo slides/week_misc/demo_doc
```

Run the alternate policy in the copy and emit uniquely named artifacts (`captions_doc.json`, `curated_doc_annotated_from_captions.pdf`, etc.) so both versions remain available for review.

## Checklist

- [ ] Write `selection_hint.md` with concrete keep/drop rules for the domain.
- [ ] Clone `prompts/kept_review.md` and tailor the instructions for the desired callouts.
- [ ] Set `LECTURE_REVIEW_PROMPT` to the custom file before Pass 2.
- [ ] Rename duplicate decks if you need side-by-side comparisons.
- [ ] Re-run captions/annotation after Pass 2 so summaries match the new prompt.
