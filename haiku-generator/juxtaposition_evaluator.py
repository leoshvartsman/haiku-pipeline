#!/usr/bin/env python3
"""
Feature 3: Juxtaposition / "Turn" Evaluation

Evaluates the kireji (cutting moment) in each haiku — the juxtaposition
between two images or ideas that creates meaning in the gap. This is
arguably the most essential element of haiku craft.
"""

import re
import json
import os
from typing import List, Dict, Tuple
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0)
MODEL = "claude-haiku-4-5-20251001"  # Scoring/classification task — Haiku handles well


def evaluate_turns(haiku_list: List[str], batch_size: int = 50) -> Tuple[List[Dict], float]:
    """Evaluate the kireji/turn quality of each haiku.

    Args:
        haiku_list: List of haiku texts
        batch_size: Number of haiku per API call

    Returns:
        Tuple of (results, total_cost)
        results: List of {text, turn_score, turn_type, analysis}
        total_cost: API cost in dollars
    """
    all_results = []
    total_cost = 0.0

    for i in range(0, len(haiku_list), batch_size):
        batch = haiku_list[i:i + batch_size]
        formatted = "\n\n".join([f"[{j}]\n{h}" for j, h in enumerate(batch)])

        prompt = f"""Evaluate the KIREJI (cutting/turn) quality of each haiku. The "turn" is the
juxtaposition between two images, ideas, or perspectives within the poem.

A great haiku creates a gap between two elements that the reader's mind must
bridge — this is what creates resonance and depth.

Score each 0-10 on TURN QUALITY:
- 9-10: Masterful juxtaposition. Two distinct images create a profound,
  unexpected resonance. The gap between them opens a universe of meaning.
- 7-8: Strong turn. Clear juxtaposition with meaningful resonance.
- 5-6: Adequate turn. Some juxtaposition present but predictable.
- 3-4: Weak turn. Images are connected too literally or logically.
- 0-2: No turn. The haiku reads as a single continuous description.

Return ONLY a JSON array: [{{"index": 0, "turn_score": 8.5, "turn_type": "image_contrast", "analysis": "brief note"}}]

Turn types: image_contrast, temporal_shift, scale_shift, emotional_pivot, sensory_switch, conceptual_leap

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
                        all_results.append({
                            'text': batch[idx],
                            'turn_score': score_data.get('turn_score', 5.0),
                            'turn_type': score_data.get('turn_type', 'unknown'),
                            'analysis': score_data.get('analysis', '')
                        })
                    # Fill in any missing indices
                scored_indices = {s.get('index') for s in scores}
                for idx in range(len(batch)):
                    if idx not in scored_indices:
                        all_results.append({
                            'text': batch[idx],
                            'turn_score': 5.0,
                            'turn_type': 'unknown',
                            'analysis': ''
                        })
        except (json.JSONDecodeError, AttributeError):
            for h in batch:
                all_results.append({
                    'text': h,
                    'turn_score': 5.0,
                    'turn_type': 'unknown',
                    'analysis': ''
                })

        usage = response.usage
        cost = (usage.input_tokens * 1.00 / 1_000_000) + (usage.output_tokens * 5.00 / 1_000_000)
        total_cost += cost

        print(f"  Turn evaluation batch {i // batch_size + 1}/{(len(haiku_list) + batch_size - 1) // batch_size}: ${cost:.4f}")

    return all_results, total_cost


def get_turn_scores_map(results: List[Dict]) -> Dict[str, float]:
    """Convert results list to a text->score mapping for quick lookup."""
    return {r['text']: r['turn_score'] for r in results}


if __name__ == "__main__":
    test_haiku = [
        "spring rain falling—\nthe earthworm crosses\na wet stone path",
        "traffic light changing—\nthe cyclist's breath\nfogs the morning air",
        "construction crane lifting—\na butterfly crosses\nits shadow",
    ]

    print("Turn/Juxtaposition Evaluation Test")
    print("=" * 50)

    results, cost = evaluate_turns(test_haiku)
    for r in results:
        print(f"\n{r['text']}")
        print(f"  Turn score: {r['turn_score']}")
        print(f"  Turn type:  {r['turn_type']}")
        print(f"  Analysis:   {r['analysis']}")
    print(f"\nTotal cost: ${cost:.4f}")
