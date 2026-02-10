#!/usr/bin/env python3
"""
Feature 5: Canonical Plagiarism / Echo Detection

Checks generated haiku against famous works by Basho, Buson, Issa, Shiki,
and English-language masters. Uses semantic similarity (sentence-transformers)
for fast detection, then LLM verification on flagged items.
"""

import json
import re
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0, max_retries=5)
MODEL = "claude-sonnet-4-20250514"

CANONICAL_FILE = Path(__file__).parent / "canonical_haiku.json"

# Cache for canonical embeddings
_canonical_cache = {
    'texts': None,
    'embeddings': None,
    'metadata': None
}


def load_canonical_haiku() -> List[Dict]:
    """Load canonical haiku database."""
    with open(CANONICAL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['haiku']


def _get_canonical_embeddings():
    """Get or compute embeddings for canonical haiku. Cached in memory."""
    if _canonical_cache['embeddings'] is not None:
        return _canonical_cache['texts'], _canonical_cache['embeddings'], _canonical_cache['metadata']

    canonical = load_canonical_haiku()
    texts = [h['text'] for h in canonical]
    metadata = [{'author': h['author'], 'era': h.get('era', '')} for h in canonical]

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=False)
    except ImportError:
        print("  Warning: sentence-transformers not installed. Skipping semantic plagiarism check.")
        return texts, None, metadata

    _canonical_cache['texts'] = texts
    _canonical_cache['embeddings'] = embeddings
    _canonical_cache['metadata'] = metadata

    return texts, embeddings, metadata


def detect_echoes_semantic(
    new_haiku: List[str],
    threshold: float = 0.70
) -> List[Dict]:
    """Use sentence-transformers embeddings to find haiku too close to canonical works.

    Args:
        new_haiku: List of generated haiku texts
        threshold: Similarity threshold (0-1). Default 0.70 flags items
            with 70%+ similarity to any canonical haiku.

    Returns:
        List of flagged items: {haiku, canonical_match, similarity, author, index}
    """
    canonical_texts, canonical_embeddings, metadata = _get_canonical_embeddings()

    if canonical_embeddings is None:
        return []

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        new_embeddings = model.encode(new_haiku, show_progress_bar=False)
    except ImportError:
        return []

    # Compute cosine similarity matrix: new_haiku x canonical_haiku
    # Normalize embeddings
    new_norm = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    can_norm = canonical_embeddings / np.linalg.norm(canonical_embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(new_norm, can_norm.T)

    flagged = []
    for i in range(len(new_haiku)):
        max_sim_idx = np.argmax(similarity_matrix[i])
        max_sim = similarity_matrix[i][max_sim_idx]

        if max_sim >= threshold:
            flagged.append({
                'haiku': new_haiku[i],
                'index': i,
                'canonical_match': canonical_texts[max_sim_idx],
                'similarity': float(max_sim),
                'author': metadata[max_sim_idx]['author']
            })

    return flagged


def detect_echoes_llm(
    flagged_haiku: List[Dict],
    batch_size: int = 25
) -> Tuple[List[Dict], float]:
    """LLM verification of semantically flagged haiku.

    Determines if similarity is genuine plagiarism, a noticeable echo,
    or coincidental.

    Returns:
        Tuple of (verdicts, cost)
        verdicts: List of {haiku, verdict, explanation, canonical_match, author}
        verdict is "plagiarism", "echo", or "coincidence"
    """
    if not flagged_haiku:
        return [], 0.0

    all_verdicts = []
    total_cost = 0.0

    for i in range(0, len(flagged_haiku), batch_size):
        batch = flagged_haiku[i:i + batch_size]

        pairs_text = ""
        for j, item in enumerate(batch):
            pairs_text += f"""[{j}]
GENERATED: {item['haiku']}
CANONICAL: {item['canonical_match']} — {item['author']}
SIMILARITY: {item['similarity']:.2f}

"""

        prompt = f"""These generated haiku have been flagged as potentially too similar to famous
canonical haiku. For each pair, determine if the generated haiku is:

A) PLAGIARISM: Too close to the original. Uses the same core image, structure,
   and meaning. Should be removed.
B) ECHO: Reminiscent but with enough originality to stand on its own.
   Borderline — flag for awareness but can be kept.
C) COINCIDENCE: Surface similarity only. Different in meaning and effect.
   Safe to keep.

{pairs_text}

Return ONLY a JSON array: [{{"index": 0, "verdict": "A", "explanation": "..."}}]"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group())
                for r in results:
                    idx = r.get('index', 0)
                    if idx < len(batch):
                        verdict_map = {'A': 'plagiarism', 'B': 'echo', 'C': 'coincidence'}
                        raw_verdict = r.get('verdict', 'C')
                        all_verdicts.append({
                            'haiku': batch[idx]['haiku'],
                            'verdict': verdict_map.get(raw_verdict, raw_verdict.lower()),
                            'explanation': r.get('explanation', ''),
                            'canonical_match': batch[idx]['canonical_match'],
                            'author': batch[idx]['author'],
                            'similarity': batch[idx]['similarity']
                        })
        except (json.JSONDecodeError, AttributeError):
            # Default to keeping on parse failure
            for item in batch:
                all_verdicts.append({
                    'haiku': item['haiku'],
                    'verdict': 'coincidence',
                    'explanation': 'Parse error, defaulting to keep',
                    'canonical_match': item['canonical_match'],
                    'author': item['author'],
                    'similarity': item['similarity']
                })

        usage = response.usage
        cost = (usage.input_tokens * 3.00 / 1_000_000) + (usage.output_tokens * 15.00 / 1_000_000)
        total_cost += cost

    return all_verdicts, total_cost


def filter_plagiaristic(
    haiku_list: List[str],
    threshold: float = 0.70
) -> Tuple[List[str], List[Dict], float]:
    """Full plagiarism detection pipeline: semantic check + LLM verification.

    Args:
        haiku_list: List of generated haiku texts
        threshold: Semantic similarity threshold for flagging

    Returns:
        Tuple of (clean_haiku, removed, total_cost)
        clean_haiku: Haiku that passed plagiarism check
        removed: List of removed haiku with details
        total_cost: API cost in dollars
    """
    # Phase 1: Semantic similarity check (free)
    flagged = detect_echoes_semantic(haiku_list, threshold)

    if not flagged:
        print(f"  No canonical echoes detected (threshold: {threshold})")
        return haiku_list, [], 0.0

    print(f"  Flagged {len(flagged)} potential echoes for LLM verification...")

    # Phase 2: LLM verification (costs money only for flagged items)
    verdicts, cost = detect_echoes_llm(flagged)

    # Remove only plagiarism verdicts
    plagiarism_texts = set()
    removed = []
    for v in verdicts:
        if v['verdict'] == 'plagiarism':
            plagiarism_texts.add(v['haiku'])
            removed.append(v)

    clean = [h for h in haiku_list if h not in plagiarism_texts]

    echo_count = sum(1 for v in verdicts if v['verdict'] == 'echo')
    coincidence_count = sum(1 for v in verdicts if v['verdict'] == 'coincidence')

    print(f"  Results: {len(removed)} plagiarism, {echo_count} echoes (kept), {coincidence_count} coincidences (kept)")

    return clean, removed, cost


if __name__ == "__main__":
    test_haiku = [
        "an old silent pond\na frog leaps into the water\nsplash breaks the silence",  # Very close to Basho
        "summer grasses grow\nwhere soldiers once dreamed their dreams\nnothing remains now",  # Echo of Basho
        "morning coffee steam\nrises from the blue ceramic\nlike a question mark",  # Original
        "the crow settles down\non a bare and leafless branch\nautumn dusk descends",  # Close to Basho
    ]

    print("Plagiarism Detection Test")
    print("=" * 50)

    clean, removed, cost = filter_plagiaristic(test_haiku, threshold=0.65)

    print(f"\nClean haiku: {len(clean)}")
    for h in clean:
        print(f"  {h.replace(chr(10), ' / ')}")

    print(f"\nRemoved: {len(removed)}")
    for r in removed:
        print(f"  {r['haiku'].replace(chr(10), ' / ')}")
        print(f"    Matched: {r['canonical_match'].replace(chr(10), ' / ')} — {r['author']}")
        print(f"    Similarity: {r['similarity']:.2f}")
        print(f"    Reason: {r['explanation']}")

    print(f"\nCost: ${cost:.4f}")
