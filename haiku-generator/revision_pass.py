#!/usr/bin/env python3
"""
Feature 1: Revision Pass

Takes near-miss haiku (scores 6.0-7.5) from evaluation and sends them
to Claude with specific feedback for improvement. Recovered haiku are
re-filtered and re-scored before joining the final selection.
"""

import re
import json
import os
from typing import List, Dict, Tuple
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL = "claude-sonnet-4-20250514"


def identify_near_misses(
    scored_haiku: List[Dict],
    min_score: float = 6.0,
    max_score: float = 7.5,
    max_count: int = 100
) -> List[Dict]:
    """Extract haiku in the near-miss score range.

    Args:
        scored_haiku: List of {text, score} dicts from evaluation
        min_score: Minimum score to consider for revision
        max_score: Maximum score (above this, already good enough)
        max_count: Maximum number of near-misses to revise

    Returns:
        List of near-miss haiku dicts, sorted by score descending
    """
    near_misses = [
        h for h in scored_haiku
        if min_score <= h.get('score', 0) <= max_score
    ]
    # Sort by score descending — revise the most promising first
    near_misses.sort(key=lambda x: x.get('score', 0), reverse=True)
    return near_misses[:max_count]


def revise_haiku_batch(
    near_misses: List[Dict],
    persona: Dict,
    batch_size: int = 25
) -> Tuple[List[str], float]:
    """Send near-miss haiku to Claude for revision.

    Args:
        near_misses: List of {text, score} dicts
        persona: Persona dictionary for voice consistency
        batch_size: Number of haiku per API call

    Returns:
        Tuple of (revised_haiku_texts, total_cost)
    """
    all_revised = []
    total_cost = 0.0

    # Build persona context
    name = persona.get('name', 'the poet')
    characteristic = persona.get('characteristic', '')
    locations = persona.get('locations', {})
    current = locations.get('current', '')

    persona_brief = f"{name}, whose characteristic style is: {characteristic}"
    if current:
        persona_brief += f". Currently lives in {current}"

    for i in range(0, len(near_misses), batch_size):
        batch = near_misses[i:i + batch_size]

        haiku_text = ""
        for j, h in enumerate(batch):
            haiku_text += f"[{j}] (score: {h.get('score', 6.5):.1f})\n{h['text']}\n\n"

        prompt = f"""You are revising near-miss haiku to elevate them to publication quality.
Each haiku below was evaluated and scored — promising but not quite there.

Write from the perspective of {persona_brief}.

For each haiku, provide a REVISED version that:
- Sharpens vague or generic imagery into something specific and concrete
- Strengthens the kireji (turn/juxtaposition) between images
- Replaces any clichéd words with fresh alternatives
- Improves sonic quality (consider how it sounds read aloud)
- Maintains the core image or observation that made it promising

IMPORTANT: The revision should be a clear improvement, not a complete rewrite.
Keep the essential spirit of the original.

{haiku_text}

Return ONLY a JSON array: [{{"index": 0, "revised": "line 1\\nline 2\\nline 3", "changes": "brief note on what was changed"}}]"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group())
                for r in results:
                    revised = r.get('revised', '')
                    if revised:
                        # Clean up any escaped newlines
                        revised = revised.replace('\\n', '\n')
                        lines = [l.strip() for l in revised.split('\n') if l.strip()]
                        if len(lines) == 3:
                            all_revised.append('\n'.join(lines))
        except (json.JSONDecodeError, AttributeError):
            pass  # Skip batch on parse failure

        usage = response.usage
        cost = (usage.input_tokens * 3.00 / 1_000_000) + (usage.output_tokens * 15.00 / 1_000_000)
        total_cost += cost

        print(f"  Revision batch {i // batch_size + 1}/{(len(near_misses) + batch_size - 1) // batch_size}: "
              f"{len(all_revised)} revised so far (${cost:.4f})")

    return all_revised, total_cost


def revision_pipeline(
    all_scored: List[Dict],
    selected_texts: List[str],
    persona: Dict,
    existing_hashes: set,
    min_score: float = 6.0,
    max_score: float = 7.5,
    max_revisions: int = 100
) -> Tuple[List[Dict], float]:
    """Full revision pipeline: identify near-misses, revise, re-filter, re-score.

    Args:
        all_scored: Complete list of {text, score} from first evaluation pass
        selected_texts: Texts already selected in first pass (to avoid duplicates)
        persona: Persona dictionary
        existing_hashes: Set of MD5 hashes for duplicate detection
        min_score: Minimum score for near-miss range
        max_score: Maximum score for near-miss range
        max_revisions: Maximum number of haiku to attempt revising

    Returns:
        Tuple of (recovered_haiku, total_cost)
        recovered_haiku: List of {text, score} for successfully revised haiku
    """
    total_cost = 0.0

    # Step 1: Identify near-misses (not already selected)
    selected_set = set(selected_texts)
    candidates = [h for h in all_scored if h['text'] not in selected_set]
    near_misses = identify_near_misses(candidates, min_score, max_score, max_revisions)

    if not near_misses:
        print("  No near-misses found for revision")
        return [], 0.0

    print(f"  Found {len(near_misses)} near-misses (scores {min_score}-{max_score})")

    # Step 2: Revise
    revised_texts, revision_cost = revise_haiku_batch(near_misses, persona)
    total_cost += revision_cost

    if not revised_texts:
        print("  No successful revisions")
        return [], total_cost

    print(f"  Got {len(revised_texts)} revised haiku")

    # Step 3: Re-filter through computational filters
    from quality_filter import apply_computational_filters
    filtered = apply_computational_filters(revised_texts, existing_hashes)
    print(f"  After computational filters: {len(filtered)}")

    if not filtered:
        return [], total_cost

    # Step 4: Remove any that duplicate already-selected haiku
    filtered = [h for h in filtered if h not in selected_set]
    print(f"  After deduplication with selected: {len(filtered)}")

    if not filtered:
        return [], total_cost

    # Step 5: Re-score the revised haiku
    scored_revised, score_cost = _score_revised(filtered)
    total_cost += score_cost

    # Only keep haiku that score >= max_score (they need to earn their spot)
    recovered = [h for h in scored_revised if h['score'] >= max_score]
    print(f"  Recovered {len(recovered)} haiku scoring >= {max_score}")

    return recovered, total_cost


def _score_revised(haiku_list: List[str]) -> Tuple[List[Dict], float]:
    """Score revised haiku using the same evaluation criteria as the main pipeline."""
    if not haiku_list:
        return [], 0.0

    formatted = "\n\n".join([f"[{j}]\n{h}" for j, h in enumerate(haiku_list)])

    prompt = f"""Evaluate these haiku and score each 0-10.

SCORING CRITERIA:
- Concrete, specific imagery (not vague or abstract)
- Fresh perspective or unexpected observation
- Clear, precise language
- Captures a moment or image effectively
- Avoids clichés and overused phrases
- Strong juxtaposition/turn between images

Return ONLY a JSON array: [{{"index": 0, "score": 8.5}}]

Haiku:

{formatted}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    scored = []
    try:
        text = response.content[0].text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            for score_data in scores:
                idx = score_data.get('index', 0)
                if idx < len(haiku_list):
                    scored.append({
                        'text': haiku_list[idx],
                        'score': score_data.get('score', 7.0)
                    })
    except (json.JSONDecodeError, AttributeError):
        for h in haiku_list:
            scored.append({'text': h, 'score': 7.0})

    usage = response.usage
    cost = (usage.input_tokens * 3.00 / 1_000_000) + (usage.output_tokens * 15.00 / 1_000_000)

    return scored, cost


if __name__ == "__main__":
    # Test with some example near-miss haiku
    test_scored = [
        {'text': "morning dew glistens\non the garden flowers blooming\nbirds sing their sweet songs", 'score': 6.5},
        {'text': "the old clock ticking\ndusty shelves hold memories\ntime passes slowly", 'score': 6.8},
        {'text': "rain on the window\nstreetlights blur and shimmer there\ncars pass quietly", 'score': 7.0},
    ]

    test_persona = {
        'name': 'Maria Santos',
        'characteristic': 'writes with sharp urban observation',
        'locations': {'current': 'São Paulo, Brazil'},
    }

    print("Revision Pass Test")
    print("=" * 50)

    near_misses = identify_near_misses(test_scored, min_score=6.0, max_score=7.5)
    print(f"Found {len(near_misses)} near-misses")

    revised, cost = revise_haiku_batch(near_misses, test_persona)
    print(f"\nRevised {len(revised)} haiku:")
    for r in revised:
        print(f"  {r.replace(chr(10), ' / ')}")
    print(f"\nCost: ${cost:.4f}")
