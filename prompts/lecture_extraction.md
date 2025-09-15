# Lecture Slide Extraction Prompt

## Task Overview
You will analyze images from a lecture video containing slides, live coding demonstrations, and terminal/editor combinations. Extract meaningful progression states while eliminating redundant frames and partial typing. 

## Decision Framework

For each image, determine:

### 1. What's Actually Visible
- Describe only what you can literally see - don't infer what "must have happened"
- Is there actual terminal output/results, or just a waiting prompt?
- Is code syntactically complete, or actively being typed?

### 2. Educational Value
- Does this show a new concept, method, or significant code change?
- Does this show meaningful execution results or before/after comparisons?
- If this looks similar to content you recently kept, does it add something educationally important?

### 3. Keep or Discard
Based on these simple principles:

**KEEP when you see:**
- New concepts being introduced
- Complete, readable code at logical stopping points  
- Actual program output, results, or error messages that teach something
- Clear before/after states that show the effect of changes
- Complete slides with new information

**DISCARD when you see:**
- Terminal showing only prompts without output
- Minor scrolling, cursor movement, or text selection changes
- Content very similar to something you recently kept
- Blurry, corrupted, or transition frames

## Content Types

**Traditional Slides**: Keep complete, readable slides that introduce new topics
**Code Demonstrations**: Keep logical checkpoints - initial setup, key implementations, final working states
**Terminal Sessions**: Keep frames showing actual command results, not just prompts waiting for input
**Mixed Layouts**: Evaluate each area - keep if either shows meaningful new content

## Output Format

Return JSON with this structure:

```json
{
  "results": [
    {
      "page": "filename.png",
      "score": 0.0-1.0,
      "keep": true/false,
      "concept": "specific concept being demonstrated (e.g., 'writing __init__ method', 'docstring parameters section', 'class instantiation output')",
      "reason": "Brief explanation of decision"
    }
  ]
}
```

**Scoring Guidelines:**
- 0.9-1.0: Essential content (new concepts, key results, major implementations)
- 0.7-0.8: Important content (good examples, meaningful steps)
- 0.5-0.6: Moderate content (context, minor additions)
- 0.3-0.4: Low value (redundant, minor changes)
- 0.0-0.2: Should be discarded (noise, duplicates, incomplete)

**Reasoning Guidelines:**
- Be specific about what makes content worth keeping
- For simple discards (duplicates, cursor moves): brief explanations are fine
- For content decisions: explain what educational value this frame provides

## Key Principles

1. **Focus on learning progression** - keep states that advance understanding
2. **Avoid redundancy** - if you recently kept similar "complete" content, this frame needs to show clear advancement
3. **Value actual results** - prioritize frames showing execution output over setup
4. **Be selective** - not every complete code state needs to be kept; focus on key milestones

Remember: You're curating a learning sequence. Each kept frame should contribute something meaningful to understanding the demonstration that nearby frames don't already provide.