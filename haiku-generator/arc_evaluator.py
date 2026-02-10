#!/usr/bin/env python3
"""
Feature 10: Collection-Level Arc Evaluation

Reads the entire assembled book and evaluates the reader experience:
opening strength, variety, pacing, emotional arc, dead spots,
transitions, ending strength, and overall coherence.
May suggest reorderings that trigger a rebuild.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0, max_retries=5)
MODEL = "claude-haiku-4-5-20251001"  # Evaluation task — Haiku handles well


def evaluate_collection_arc(
    anthology_text: str,
    title: str
) -> Tuple[Dict, float]:
    """Evaluate the full reading experience of the assembled anthology.

    Args:
        anthology_text: Complete anthology text (with sections, intros, haiku)
        title: Collection title

    Returns:
        Tuple of (evaluation_dict, cost)
        evaluation_dict includes scores, dead_spots, suggested_reorderings, summary
    """
    # Truncate if extremely long (keep within token limits)
    max_chars = 50000
    if len(anthology_text) > max_chars:
        anthology_text = anthology_text[:max_chars] + "\n\n[... truncated for evaluation ...]"

    prompt = f"""You are a poetry editor reading a complete haiku anthology for the first
time. Read through the entire collection below and evaluate the READER EXPERIENCE.

"{title}"

{anthology_text}

Evaluate on these dimensions (score each 0-10):
1. OPENING STRENGTH: Does the collection begin compellingly? Is the first
   haiku in each section strong enough to draw readers in?
2. VARIETY: Sufficient range of subjects, tones, and imagery across the
   full collection? Or do certain images/themes repeat too often?
3. PACING: Good rhythm between quiet and intense poems? Does the reading
   feel dynamic or monotonous?
4. EMOTIONAL ARC: Does the collection build toward something? Is there
   a sense of journey or progression?
5. DEAD SPOTS: Are there stretches that feel repetitive, flat, or where
   energy drops noticeably?
6. TRANSITIONS: Do section transitions work well? Does moving from one
   season/section to the next feel natural?
7. ENDING STRENGTH: Does the collection close memorably? Does the final
   haiku in each section and the collection as a whole leave an impression?
8. OVERALL COHERENCE: Does it feel like a unified collection from a single
   sensibility, or a random assortment?

Return ONLY a JSON object:
{{
  "scores": {{
    "opening": 0,
    "variety": 0,
    "pacing": 0,
    "emotional_arc": 0,
    "dead_spots": 0,
    "transitions": 0,
    "ending": 0,
    "coherence": 0
  }},
  "overall_score": 0.0,
  "dead_spots_detail": [
    {{"section": "section name", "approximate_position": "beginning/middle/end", "reason": "..."}}
  ],
  "suggested_reorderings": [
    {{"section": "section name", "suggestion": "description of what to move and why"}}
  ],
  "opening_suggestion": "which haiku or type of haiku would make a stronger opener",
  "closing_suggestion": "which haiku or type of haiku would make a stronger closer",
  "strengths": ["list of 2-3 collection strengths"],
  "summary": "2-3 sentence overall assessment of the reading experience"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}]
    )

    evaluation = {}
    try:
        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError):
        evaluation = {
            'scores': {
                'opening': 7, 'variety': 7, 'pacing': 7,
                'emotional_arc': 7, 'dead_spots': 7, 'transitions': 7,
                'ending': 7, 'coherence': 7
            },
            'overall_score': 7.0,
            'dead_spots_detail': [],
            'suggested_reorderings': [],
            'summary': 'Evaluation parse failed; defaulting to neutral scores.'
        }

    # Calculate overall if not provided
    if 'overall_score' not in evaluation or not evaluation['overall_score']:
        scores = evaluation.get('scores', {})
        if scores:
            evaluation['overall_score'] = sum(scores.values()) / len(scores)

    usage = response.usage
    cost = (usage.input_tokens * 1.00 / 1_000_000) + (usage.output_tokens * 5.00 / 1_000_000)

    return evaluation, cost


def apply_arc_suggestions(
    haiku_analyses: List[Dict],
    suggestions: Dict
) -> bool:
    """Apply arc evaluation feedback by reordering the flat haiku list.

    Performs targeted fixes based on arc evaluation dimension scores:
    1. Opener selection — pick a strong, inviting poem for position 1
    2. Closer selection — pick a resonant, memorable poem for the end
    3. Dead spot dispersal — break up consecutive same-theme clusters
    4. Pacing enforcement — alternate quiet/intense when pacing is low

    All fixes are computational (no extra API calls).
    Modifies haiku_analyses in-place.

    Returns:
        True if modifications were made, False otherwise
    """
    scores = suggestions.get('scores', {})

    # If all scores are 7+, nothing to fix
    if all(v >= 7 for v in scores.values()):
        return False

    modified = False

    # 1. Fix opener if opening score is weak
    if scores.get('opening', 10) < 7:
        if _fix_opener(haiku_analyses):
            modified = True
            print(f"    Arc fix: swapped opener")

    # 2. Fix closer if ending score is weak
    if scores.get('ending', 10) < 7:
        if _fix_closer(haiku_analyses):
            modified = True
            print(f"    Arc fix: swapped closer")

    # 3. Disperse dead spots
    if scores.get('variety', 10) < 6 or scores.get('dead_spots', 10) < 6:
        if _disperse_clusters(haiku_analyses):
            modified = True
            print(f"    Arc fix: dispersed repetitive clusters")

    # 4. Enforce pacing if score is low
    if scores.get('pacing', 10) < 6:
        if _enforce_pacing(haiku_analyses):
            modified = True
            print(f"    Arc fix: improved pacing")

    return modified


def _fix_opener(section_haiku: List[Dict]) -> bool:
    """Pick a strong but inviting poem for position 1.

    Good openers are high quality but not the absolute best (save that
    for the middle). Inviting, accessible tones are preferred.
    """
    inviting_tones = {'quiet', 'contemplative', 'peaceful', 'gentle', 'serene', 'observational'}
    max_quality = max(h.get('quality', 7.0) for h in section_haiku)

    best_idx = -1
    best_score = -1.0
    for i, h in enumerate(section_haiku):
        if i == 0:
            continue  # Skip current opener
        quality = h.get('quality', 7.0)
        tone = h.get('tone', '').lower()

        opener_score = quality
        if any(t in tone for t in inviting_tones):
            opener_score += 1.5
        # Penalize the absolute best poem — save it for the middle
        if quality == max_quality:
            opener_score -= 2.0

        if opener_score > best_score:
            best_score = opener_score
            best_idx = i

    if best_idx < 0:
        return False

    # Only swap if it's a meaningful improvement
    current_quality = section_haiku[0].get('quality', 7.0)
    if section_haiku[best_idx].get('quality', 7.0) < current_quality - 1.0:
        return False

    section_haiku[0], section_haiku[best_idx] = section_haiku[best_idx], section_haiku[0]
    return True


def _fix_closer(section_haiku: List[Dict]) -> bool:
    """Pick a resonant, memorable poem for the final position.

    Good closers leave an impression — contemplative, slightly melancholic,
    or with a sense of resolution. High turn scores help.
    """
    closing_tones = {'contemplative', 'melancholic', 'serene', 'quiet', 'peaceful', 'reflective'}
    last = len(section_haiku) - 1

    best_idx = -1
    best_score = -1.0
    for i, h in enumerate(section_haiku):
        if i == last or i == 0:
            continue  # Skip current closer and opener
        quality = h.get('quality', 7.0)
        tone = h.get('tone', '').lower()
        turn = h.get('turn_score', 5.0)

        closer_score = quality + (turn * 0.3)
        if any(t in tone for t in closing_tones):
            closer_score += 1.5

        if closer_score > best_score:
            best_score = closer_score
            best_idx = i

    if best_idx < 0:
        return False

    section_haiku[last], section_haiku[best_idx] = section_haiku[best_idx], section_haiku[last]
    return True


def _disperse_clusters(section_haiku: List[Dict]) -> bool:
    """Break up runs of 3+ consecutive poems with the same theme."""
    changed = False

    for _ in range(3):  # Multiple passes for overlapping clusters
        found = False
        i = 0
        while i < len(section_haiku) - 2:
            theme_a = section_haiku[i].get('theme', '').lower()
            if not theme_a:
                i += 1
                continue

            run_length = 1
            for j in range(i + 1, len(section_haiku)):
                if section_haiku[j].get('theme', '').lower() == theme_a:
                    run_length += 1
                else:
                    break

            if run_length >= 3:
                # Move the middle poem away from the cluster
                mid = i + run_length // 2
                # Don't move opener or closer
                if mid == 0 or mid == len(section_haiku) - 1:
                    i += run_length
                    continue
                poem = section_haiku.pop(mid)
                # Insert halfway across the section
                insert_pos = (mid + len(section_haiku) // 2) % len(section_haiku)
                insert_pos = max(1, min(insert_pos, len(section_haiku) - 1))
                section_haiku.insert(insert_pos, poem)
                found = True
                changed = True

            i += run_length

        if not found:
            break

    return changed


def _enforce_pacing(section_haiku: List[Dict]) -> bool:
    """Alternate quiet/intense poems for better pacing.

    Preserves opener (pos 0) and closer (last). Rearranges middle
    poems to alternate between quiet and intense tones.
    """
    if len(section_haiku) < 6:
        return False

    quiet_tones = {'quiet', 'contemplative', 'peaceful', 'gentle', 'serene'}
    intense_tones = {'intense', 'joyful', 'melancholic', 'powerful', 'dramatic', 'vivid'}

    opener = section_haiku[0]
    closer = section_haiku[-1]
    middle = section_haiku[1:-1]

    quiet = []
    intense = []
    neutral = []

    for h in middle:
        tone = h.get('tone', '').lower()
        if any(t in tone for t in quiet_tones):
            quiet.append(h)
        elif any(t in tone for t in intense_tones):
            intense.append(h)
        else:
            neutral.append(h)

    # Only rearrange if there's meaningful variety to work with
    if not quiet or not intense:
        return False
    if len(quiet) < 2 and len(intense) < 2:
        return False

    # Build alternating sequence
    reordered = []
    qi, ii, ni = 0, 0, 0
    use_quiet = True

    while qi < len(quiet) or ii < len(intense) or ni < len(neutral):
        if use_quiet and qi < len(quiet):
            reordered.append(quiet[qi])
            qi += 1
        elif not use_quiet and ii < len(intense):
            reordered.append(intense[ii])
            ii += 1
        elif ni < len(neutral):
            reordered.append(neutral[ni])
            ni += 1
        elif qi < len(quiet):
            reordered.append(quiet[qi])
            qi += 1
        elif ii < len(intense):
            reordered.append(intense[ii])
            ii += 1
        use_quiet = not use_quiet

    section_haiku[:] = [opener] + reordered + [closer]
    return True


def print_arc_evaluation(evaluation: Dict):
    """Pretty-print the arc evaluation results."""
    scores = evaluation.get('scores', {})
    overall = evaluation.get('overall_score', 0)

    print(f"\n  Collection Arc Evaluation:")
    print(f"  {'─' * 40}")
    for dimension, score in scores.items():
        label = dimension.replace('_', ' ').title()
        bar = '█' * int(score) + '░' * (10 - int(score))
        print(f"  {label:<20} {bar} {score}/10")
    print(f"  {'─' * 40}")
    print(f"  {'Overall':<20} {'█' * int(overall) + '░' * (10 - int(overall))} {overall:.1f}/10")

    strengths = evaluation.get('strengths', [])
    if strengths:
        print(f"\n  Strengths:")
        for s in strengths:
            print(f"    + {s}")

    dead_spots = evaluation.get('dead_spots_detail', [])
    if dead_spots:
        print(f"\n  Dead spots:")
        for ds in dead_spots:
            print(f"    - {ds.get('section', '?')}: {ds.get('reason', '')}")

    summary = evaluation.get('summary', '')
    if summary:
        print(f"\n  Summary: {summary}")


if __name__ == "__main__":
    # Test with a small example
    test_text = """
======================================================================
                         Dew on Iron
                          by Kenji Watanabe
======================================================================

# INTRODUCTION

A brief collection exploring the intersection of nature and industry...

======================================================================
# SPRING
======================================================================

1.
spring rain falling—
the earthworm crosses
a wet stone path

2.
cherry petals drift
across the factory floor
swept up by robots

======================================================================
# SUMMER
======================================================================

3.
traffic light changing—
the cyclist's breath
fogs the morning air

======================================================================
                     End of Dew on Iron
======================================================================
"""

    print("Arc Evaluation Test")
    print("=" * 50)

    evaluation, cost = evaluate_collection_arc(test_text, "Dew on Iron")
    print_arc_evaluation(evaluation)
    print(f"\nCost: ${cost:.4f}")
