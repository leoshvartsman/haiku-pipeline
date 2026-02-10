#!/usr/bin/env python3
"""
Feature 4: Voice Consistency Auditing

Evaluates all haiku against the persona description to ensure voice
consistency. Flags and removes outliers that break character — poems
that don't match the persona's voice, background, or characteristic style.
"""

import re
import json
import os
from typing import List, Dict, Tuple
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0, max_retries=5)
MODEL = "claude-haiku-4-5-20251001"  # Classification task — Haiku handles well


def generate_voice_profile(persona: Dict) -> str:
    """Generate a concise voice profile string from persona data."""
    name = persona.get('name', 'Unknown')
    age = persona.get('age', '')
    occupation = persona.get('occupation', '')
    characteristic = persona.get('characteristic', '')

    locations = persona.get('locations', {})
    childhood = locations.get('childhood', '')
    current = locations.get('current', '')

    languages = persona.get('languages', {})
    native = ', '.join(languages.get('native', []))
    fluent = ', '.join(languages.get('fluent', []))

    education = persona.get('poetry_education', {})
    institution = education.get('institution', '')
    edu_type = education.get('type', '')

    experience_years = persona.get('work_experience_years', 0)

    profile = f"""Poet: {name}, age {age}
Occupation: {occupation} ({experience_years} years experience)
Grew up in: {childhood}
Currently lives in: {current}
Native language(s): {native}
Fluent in: {fluent}
Poetry education: {edu_type} at {institution}
Characteristic style: {characteristic}"""

    return profile


def audit_voice_consistency(
    haiku_list: List[str],
    persona: Dict,
    batch_size: int = 50,
    threshold: float = 5.0
) -> Tuple[List[str], List[Dict], float]:
    """Audit haiku for voice consistency with persona.

    Args:
        haiku_list: List of haiku texts
        persona: Persona dictionary
        batch_size: Number of haiku per API call
        threshold: Minimum voice consistency score (0-10). Haiku below
            this score are flagged and removed.

    Returns:
        Tuple of (consistent_haiku, flagged_outliers, total_cost)
    """
    voice_profile = generate_voice_profile(persona)
    all_scores = {}  # text -> {score, consistent, reason}
    total_cost = 0.0

    for i in range(0, len(haiku_list), batch_size):
        batch = haiku_list[i:i + batch_size]
        formatted = "\n\n".join([f"[{j}]\n{h}" for j, h in enumerate(batch)])

        prompt = f"""You are auditing haiku for voice consistency. The poet is:

{voice_profile}

For each haiku below, evaluate whether it is CONSISTENT with this poet's
likely voice, perspective, and experience. A poem is inconsistent if:
- It uses vocabulary or references alien to the poet's background
- It adopts a tone fundamentally at odds with the poet's characteristic style
- It references experiences the poet would unlikely have
- It feels generic rather than reflecting a specific human perspective

Be reasonably generous — poets draw from imagination and observation,
not just personal experience. Only flag truly jarring mismatches.

Score each 0-10 for voice consistency:
- 8-10: Strongly consistent with this poet's voice
- 6-7: Generally consistent, minor concerns
- 4-5: Somewhat inconsistent, notable mismatch
- 0-3: Clearly inconsistent with this persona

Return ONLY a JSON array: [{{"index": 0, "voice_score": 8.0, "consistent": true, "reason": ""}}]

Haiku:

{formatted}"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                for score_data in scores:
                    idx = score_data.get('index', 0)
                    if idx < len(batch):
                        all_scores[batch[idx]] = {
                            'voice_score': score_data.get('voice_score', 7.0),
                            'consistent': score_data.get('consistent', True),
                            'reason': score_data.get('reason', '')
                        }
                # Fill missing
                for idx in range(len(batch)):
                    if batch[idx] not in all_scores:
                        all_scores[batch[idx]] = {
                            'voice_score': 7.0,
                            'consistent': True,
                            'reason': ''
                        }
        except (json.JSONDecodeError, AttributeError):
            for h in batch:
                all_scores[h] = {
                    'voice_score': 7.0,
                    'consistent': True,
                    'reason': ''
                }

        usage = response.usage
        cost = (usage.input_tokens * 3.00 / 1_000_000) + (usage.output_tokens * 15.00 / 1_000_000)
        total_cost += cost

        print(f"  Voice audit batch {i // batch_size + 1}/{(len(haiku_list) + batch_size - 1) // batch_size}: ${cost:.4f}")

    # Separate consistent from outliers
    consistent = []
    flagged = []

    for h in haiku_list:
        score_data = all_scores.get(h, {'voice_score': 7.0, 'consistent': True, 'reason': ''})
        if score_data['voice_score'] >= threshold:
            consistent.append(h)
        else:
            flagged.append({
                'text': h,
                'voice_score': score_data['voice_score'],
                'reason': score_data['reason']
            })

    return consistent, flagged, total_cost


if __name__ == "__main__":
    test_persona = {
        'name': 'Yuki Tanaka',
        'age': 72,
        'occupation': 'Retired fisherman',
        'work_experience_years': 45,
        'characteristic': 'writes about the sea with weathered simplicity',
        'locations': {
            'childhood': 'Hokkaido, Japan',
            'current': 'Monterey, California'
        },
        'languages': {
            'native': ['Japanese'],
            'fluent': ['Japanese', 'English']
        },
        'poetry_education': {
            'institution': 'Self-taught',
            'type': 'Autodidact'
        }
    }

    test_haiku = [
        "salt-crusted nets dry\nin the afternoon sun\na gull waits patiently",  # Consistent
        "the blockchain update\nrendering my portfolio\nin shades of red",  # Inconsistent
        "harbor fog rolling\nthe old boat rocks gently\nrope creaks at the dock",  # Consistent
    ]

    print("Voice Consistency Audit Test")
    print("=" * 50)

    consistent, flagged, cost = audit_voice_consistency(test_haiku, test_persona)

    print(f"\nConsistent: {len(consistent)}")
    for h in consistent:
        print(f"  {h.replace(chr(10), ' / ')}")

    print(f"\nFlagged: {len(flagged)}")
    for f in flagged:
        print(f"  {f['text'].replace(chr(10), ' / ')}")
        print(f"    Score: {f['voice_score']}, Reason: {f['reason']}")

    print(f"\nCost: ${cost:.4f}")
