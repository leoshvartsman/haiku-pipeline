#!/usr/bin/env python3
"""
Feature 8: Scholarly Apparatus

Generates brief scholarly notes on 10-15 standout poems, explaining
imagery, allusions, seasonal references, and craft. Added as a "Notes"
section at the back of the book.
"""

import re
import json
import os
from typing import List, Dict, Tuple
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0, max_retries=5)
MODEL = "claude-sonnet-4-20250514"


def select_standout_poems(
    haiku_analyses: List[Dict],
    count: int = 12
) -> List[Dict]:
    """Select the most noteworthy poems for annotation.

    Selection criteria (in priority order):
    1. Highest quality scores
    2. Best turn/juxtaposition scores (if available)
    3. Most unusual or diverse imagery
    4. Representation across sections

    Args:
        haiku_analyses: List of analyzed haiku dicts (from anthology_editor)
        count: Number of poems to select

    Returns:
        List of selected haiku analysis dicts with section info
    """
    if len(haiku_analyses) <= count:
        return haiku_analyses

    # Score each haiku for "noteworthiness"
    scored = []
    for i, h in enumerate(haiku_analyses):
        noteworthy_score = 0.0

        # Quality score (primary factor)
        quality = h.get('quality', 7.0)
        noteworthy_score += quality * 2.0

        # Turn score bonus (if available from juxtaposition evaluator)
        turn_score = h.get('turn_score', 0)
        noteworthy_score += turn_score * 1.5

        # Sonic score bonus (if available)
        sonic_score = h.get('sonic_score', 0)
        noteworthy_score += sonic_score * 0.5

        # Unusual theme bonus
        theme = h.get('theme', '').lower()
        unusual_themes = {'technology', 'urban', 'experimental', 'cross-cultural'}
        if any(t in theme for t in unusual_themes):
            noteworthy_score += 2.0

        scored.append({
            **h,
            'noteworthy_score': noteworthy_score,
            'global_position': i + 1
        })

    # Sort by noteworthy score
    scored.sort(key=lambda x: x['noteworthy_score'], reverse=True)

    # Ensure section diversity — take top candidates but limit per section
    selected = []
    section_counts = {}
    max_per_section = max(count // 4, 2)

    for h in scored:
        section = h.get('season', 'timeless')
        if section_counts.get(section, 0) >= max_per_section:
            continue
        selected.append(h)
        section_counts[section] = section_counts.get(section, 0) + 1
        if len(selected) >= count:
            break

    # If we didn't get enough due to section limits, fill from remaining
    if len(selected) < count:
        selected_set = {h.get('haiku', h.get('text', '')) for h in selected}
        for h in scored:
            haiku_text = h.get('haiku', h.get('text', ''))
            if haiku_text not in selected_set:
                selected.append(h)
                if len(selected) >= count:
                    break

    return selected


def generate_scholarly_notes(
    standout_poems: List[Dict],
    persona: Dict,
    title: str
) -> Tuple[str, float]:
    """Generate scholarly notes section for standout poems.

    Args:
        standout_poems: List of selected haiku analysis dicts
        persona: Author persona dictionary
        title: Collection title

    Returns:
        Tuple of (formatted_notes_text, cost)
    """
    author = persona.get('name', 'the poet')
    characteristic = persona.get('characteristic', '')

    # Build haiku listing for the prompt
    haiku_listing = ""
    for i, h in enumerate(standout_poems):
        haiku_text = h.get('haiku', h.get('text', ''))
        season = h.get('season', 'timeless')
        theme = h.get('theme', '')
        position = h.get('global_position', i + 1)
        haiku_listing += f"[#{position}] Section: {season.title()}\n{haiku_text}\n"
        if theme:
            haiku_listing += f"Theme: {theme}\n"
        haiku_listing += "\n"

    prompt = f"""You are writing brief scholarly notes for a haiku anthology titled "{title}"
by {author} (characteristic style: {characteristic}).

For each of the following standout haiku from the collection, write a
50-100 word note that:
- Identifies and explains key imagery or seasonal references (kigo)
- Notes the technique of juxtaposition (kireji) and why it works
- Explains any cultural allusions or layered meanings
- Comments on what makes this particular haiku stand out
- Notes sonic or rhythmic qualities if relevant

Tone: Accessible and appreciative, not pedantic. Think margin notes from
a knowledgeable friend, not an academic paper. Write as if the reader is
intelligent but may not know haiku terminology.

Haiku to annotate:

{haiku_listing}

Format each note as:

### [Haiku number] "[first few words...]"

[note text]

Write all {len(standout_poems)} notes."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    notes_text = response.content[0].text.strip()

    # Wrap in section formatting
    formatted = "\n\n"
    formatted += "=" * 70 + "\n"
    formatted += "NOTES ON SELECTED POEMS".center(70) + "\n"
    formatted += "=" * 70 + "\n\n"
    formatted += notes_text
    formatted += "\n\n" + "-" * 70

    usage = response.usage
    cost = (usage.input_tokens * 3.00 / 1_000_000) + (usage.output_tokens * 15.00 / 1_000_000)

    return formatted, cost


if __name__ == "__main__":
    test_analyses = [
        {'haiku': "spring rain falling—\nthe earthworm crosses\na wet stone path", 'season': 'spring', 'theme': 'nature', 'quality': 9.0},
        {'haiku': "traffic light changing—\nthe cyclist's breath\nfogs the morning air", 'season': 'timeless', 'theme': 'urban', 'quality': 8.5},
        {'haiku': "construction crane lifting—\na butterfly crosses\nits shadow", 'season': 'summer', 'theme': 'urban', 'quality': 9.2},
    ]

    test_persona = {
        'name': 'Kenji Watanabe',
        'characteristic': 'finds poetry in the intersection of nature and industry',
    }

    print("Scholarly Notes Test")
    print("=" * 50)

    standouts = select_standout_poems(test_analyses, count=3)
    notes, cost = generate_scholarly_notes(standouts, test_persona, "Steel and Blossoms")

    print(notes)
    print(f"\nCost: ${cost:.4f}")
