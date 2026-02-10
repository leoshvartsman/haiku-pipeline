#!/usr/bin/env python3
"""
Quality Filtering System for Haiku Generation WITH PROMPT CACHING
Generates 2500 haiku, keeps the best 250
Uses caching to reduce costs by ~20%
"""

import anthropic
import os
import re
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

# Initialize client
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), max_retries=5)

# Configuration
HAIKU_TO_GENERATE = 2500
HAIKU_TO_KEEP = 250
FIRST_PASS_COUNT = 500
MODEL = "claude-3-haiku-20240307"

# Common clichés to avoid
CLICHES = [
    "lonely crow", "old pond", "frog jumps", "frog leaps",
    "cherry blossom falls", "cherry blossoms fall",
    "autumn leaf falls", "autumn leaves fall",
    "moonlight shines", "moonlight gleams",
    "winter wind blows", "winter winds blow",
    "snow is falling", "snow falls gently",
    "petals floating", "petals drift",
    "distant mountain", "mountains in mist",
    "temple bell rings", "temple bells ring"
]

# CACHED SYSTEM PROMPTS
# These will be reused across all evaluation calls
EVALUATION_SYSTEM_PROMPT = """You are a haiku evaluation expert. Your task is to score haiku based on traditional and contemporary quality standards.

Quality Criteria:

1. Traditional Qualities (0-10):
   - Mono no aware (poignancy, awareness of impermanence)
   - Kigo (seasonal resonance and awareness)
   - Kire (cutting word, juxtaposition)
   - Shasei (direct observation, not abstraction)
   - Karumi (lightness, simplicity)

2. Technical Excellence (0-10):
   - Structure appropriateness (5-7-5 or justified variation)
   - Concrete imagery (no vague abstractions)
   - Single moment captured
   - Two-part structure with juxtaposition
   - Present tense immediacy

3. Contemporary Merit (0-10):
   - Freshness (avoiding clichés)
   - Cultural relevance
   - Linguistic precision
   - Sensory vividness
   - Interpretive openness

Calculate composite score as average of all dimensions.
Return scores in JSON format."""

EVALUATION_EXAMPLES = """
Examples of scoring:

HIGH QUALITY (8.5-9.5):
spring rain falling—
the earthworm crosses
a wet stone path
Score: 9.2 (strong imagery, seasonal awareness, precise moment)

traffic light changing—
the cyclist's breath
fogs the morning air
Score: 8.8 (contemporary, precise observation, juxtaposition)

MEDIUM QUALITY (7.0-8.0):
winter morning light
through bare tree branches—
coffee steam rising
Score: 7.5 (good structure, clear imagery, but somewhat expected)

LOW QUALITY (below 7.0):
lonely moonlight shines
on the ancient temple bell
sadness fills my heart
Score: 5.2 (clichés, abstraction, no concrete moment)
"""

# ============================================================================
# SYLLABLE COUNTING & VALIDATION (same as before)
# ============================================================================

def count_syllables(word: str) -> int:
    """Approximate syllable counter"""
    word = word.lower()
    count = 0
    vowels = 'aeiouy'
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel

    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count = 1

    return count

def validate_structure(haiku: str) -> bool:
    """Validate haiku structure"""
    lines = [l.strip() for l in haiku.split('\n') if l.strip()]

    if len(lines) != 3:
        return False

    syllables = []
    for line in lines:
        words = re.findall(r'\b\w+\b', line)
        line_syllables = sum(count_syllables(w) for w in words)
        syllables.append(line_syllables)

    total = sum(syllables)

    if syllables in [(5,7,5), (3,5,3), (4,6,4), (5,6,5), (4,7,4), (6,7,6)]:
        return True

    if 10 <= total <= 19 and all(2 <= s <= 9 for s in syllables):
        return True

    return False

def has_cliche(haiku: str) -> bool:
    """Check for common clichés"""
    haiku_lower = haiku.lower()
    return any(cliche in haiku_lower for cliche in CLICHES)

def get_existing_hashes(output_dir: Path) -> set:
    """Get hashes of existing haiku from previous runs"""
    hashes = set()
    if output_dir.exists():
        for file in output_dir.glob("*.txt"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    haiku_blocks = re.findall(r'\d+\.\n(.*?)\n\n', content, re.DOTALL)
                    for haiku in haiku_blocks:
                        haiku_hash = hashlib.md5(haiku.encode()).hexdigest()
                        hashes.add(haiku_hash)
            except:
                pass
    return hashes

def is_duplicate(haiku: str, existing_hashes: set) -> bool:
    """Check if haiku is duplicate"""
    haiku_hash = hashlib.md5(haiku.encode()).hexdigest()
    return haiku_hash in existing_hashes

def apply_computational_filters(haiku_list: List[str], existing_hashes: set) -> List[str]:
    """Apply all computational filters"""
    filtered = []
    for haiku in haiku_list:
        if (validate_structure(haiku) and
            not has_cliche(haiku) and
            not is_duplicate(haiku, existing_hashes)):
            filtered.append(haiku)
    return filtered

# ============================================================================
# GENERATION
# ============================================================================

def generate_haiku_batch(count: int = 2500) -> Tuple[List[str], float]:
    """Generate haiku using Claude"""
    now = datetime.now()

    month = now.month
    if month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    elif month in [9, 10, 11]:
        season = "Autumn"
    else:
        season = "Winter"

    prompt = f"""Generate {count} excellent haiku.

Current context:
- Date: {now.strftime('%B %d, %Y')}
- Season: {season}

Requirements:
- Traditional 5-7-5 or justified variations
- Concrete imagery (no abstractions)
- Present tense immediacy
- Seasonal awareness where appropriate
- Mix of traditional and contemporary styles
- Diverse subjects: nature, urban, human moments, technology

Format: One haiku per entry, separated by blank lines.
Output only the haiku, no numbering or commentary."""

    print(f"Requesting {count} haiku from Claude...")

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.9,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    haiku_list = []

    blocks = text.strip().split('\n\n')
    for block in blocks:
        block = re.sub(r'^\d+[\.)]\s*', '', block, flags=re.MULTILINE)
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) == 3:
            haiku_list.append('\n'.join(lines))

    usage = response.usage
    cost = (usage.input_tokens * 0.80 / 1_000_000) + (usage.output_tokens * 4.00 / 1_000_000)

    return haiku_list, cost

def generate_multiple_batches(target_count: int = 2500) -> Tuple[List[str], float]:
    """Generate haiku in multiple batches to reach target"""
    all_haiku = []
    total_cost = 0.0

    batches_needed = (target_count // 250) + 1
    per_batch = target_count // batches_needed

    print(f"\nGenerating {target_count} haiku in {batches_needed} batches...")

    for i in range(batches_needed):
        batch_haiku, batch_cost = generate_haiku_batch(per_batch)
        all_haiku.extend(batch_haiku)
        total_cost += batch_cost
        print(f"  Batch {i+1}/{batches_needed}: {len(batch_haiku)} haiku (${batch_cost:.4f})")

        if len(all_haiku) >= target_count:
            break

    return all_haiku[:target_count + 500], total_cost

# ============================================================================
# EVALUATION WITH CACHING
# ============================================================================

def evaluate_haiku_batch_cached(haiku_list: List[str], target_count: int,
                                 pass_name: str = "evaluation") -> Tuple[List[Dict], float]:
    """
    Evaluate haiku and return top scoring ones
    Uses prompt caching to save money on repeated system prompts
    """

    batch_size = 100
    all_scored = []
    total_cost = 0.0
    total_cache_savings = 0.0

    # Create system messages with caching
    # These will be reused across all batches in this call
    system_messages = [
        {
            "type": "text",
            "text": EVALUATION_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}  # Cache this prompt
        },
        {
            "type": "text",
            "text": EVALUATION_EXAMPLES,
            "cache_control": {"type": "ephemeral"}  # Cache examples too
        }
    ]

    print(f"\n{pass_name} ({len(haiku_list)} haiku → top {target_count})...")

    for i in range(0, len(haiku_list), batch_size):
        batch = haiku_list[i:i+batch_size]
        formatted = "\n\n".join([f"[{j}]\n{h}" for j, h in enumerate(batch)])

        user_prompt = f"""Evaluate these haiku and score each 0-10.

Return JSON array: [{{"index": 0, "score": 8.5}}, ...]

Haiku to evaluate:

{formatted}"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0.3,
            system=system_messages,  # Include cached system messages
            messages=[{"role": "user", "content": user_prompt}]
        )

        # Parse scores
        try:
            scores_text = response.content[0].text
            json_match = re.search(r'\[.*\]', scores_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                for score_data in scores:
                    idx = score_data.get('index', 0)
                    if idx < len(batch):
                        all_scored.append({
                            'text': batch[idx],
                            'score': score_data.get('score', 7.0)
                        })
        except Exception as e:
            print(f"  Warning: Error parsing scores: {e}")
            for h in batch:
                all_scored.append({'text': h, 'score': 7.0})

        # Calculate cost with caching
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        # Check for cache usage
        cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)
        cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)

        # Calculate actual cost
        cost = (
            (input_tokens * 0.80 / 1_000_000) +          # Regular input
            (cache_creation_tokens * 1.00 / 1_000_000) + # Cache creation (1.25x)
            (cache_read_tokens * 0.10 / 1_000_000) +     # Cache read (0.125x)
            (output_tokens * 4.00 / 1_000_000)           # Output
        )

        # Calculate what it would have cost without caching
        uncached_cost = (
            ((input_tokens + cache_creation_tokens + cache_read_tokens) * 0.80 / 1_000_000) +
            (output_tokens * 4.00 / 1_000_000)
        )

        batch_savings = uncached_cost - cost
        total_cache_savings += batch_savings

        total_cost += cost

        # Show cache stats for first batch and when cache is being read
        if i == 0 or cache_read_tokens > 0:
            print(f"  Batch {i//batch_size + 1}: ${cost:.4f} " +
                  (f"(created cache: {cache_creation_tokens:,} tokens)" if cache_creation_tokens > 0 else "") +
                  (f"(cache read: {cache_read_tokens:,} tokens, saved ${batch_savings:.4f})" if cache_read_tokens > 0 else ""))
        else:
            print(f"  Batch {i//batch_size + 1}: ${cost:.4f}")

    # Sort by score and return top
    all_scored.sort(key=lambda x: x['score'], reverse=True)
    top_haiku = all_scored[:target_count]

    if total_cache_savings > 0:
        print(f"  Cache savings: ${total_cache_savings:.4f}")

    return top_haiku, total_cost

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_quality_filter():
    """Run the complete quality filtering process with caching"""
    print("="*70)
    print("HAIKU QUALITY FILTER - WITH PROMPT CACHING")
    print("="*70)
    print(f"Target generation: {HAIKU_TO_GENERATE}")
    print(f"Target output: {HAIKU_TO_KEEP}")
    print(f"Model: {MODEL}")
    print(f"Caching: ENABLED (saves ~20% on costs)")
    print("="*70)

    start_time = datetime.now()
    output_dir = Path("haiku_output")
    output_dir.mkdir(exist_ok=True)

    total_cost = 0.0

    # PHASE 1: GENERATION
    print("\nPHASE 1: GENERATION")
    print("-" * 70)
    generated, gen_cost = generate_multiple_batches(HAIKU_TO_GENERATE)
    total_cost += gen_cost
    print(f"\n✓ Generated {len(generated)} haiku")
    print(f"✓ Cost: ${gen_cost:.4f}")

    # PHASE 2: COMPUTATIONAL FILTERING
    print("\nPHASE 2: COMPUTATIONAL FILTERING")
    print("-" * 70)
    existing_hashes = get_existing_hashes(output_dir)
    print(f"Loaded {len(existing_hashes)} existing haiku hashes")

    filtered = apply_computational_filters(generated, existing_hashes)
    print(f"\n✓ Passed structure validation: {len([h for h in generated if validate_structure(h)])}")
    print(f"✓ Removed clichés: {len([h for h in generated if has_cliche(h)])}")
    print(f"✓ Filtered to: {len(filtered)} haiku")

    # PHASE 3: FIRST EVALUATION PASS (WITH CACHING)
    print("\nPHASE 3: FIRST EVALUATION PASS (CACHED)")
    print("-" * 70)
    first_pass, eval1_cost = evaluate_haiku_batch_cached(
        filtered, FIRST_PASS_COUNT, "First pass evaluation"
    )
    total_cost += eval1_cost
    print(f"\n✓ Selected top {len(first_pass)} haiku")
    print(f"✓ Score range: {min(h['score'] for h in first_pass):.2f} - {max(h['score'] for h in first_pass):.2f}")
    print(f"✓ Cost: ${eval1_cost:.4f}")

    # PHASE 4: DETAILED EVALUATION (WITH CACHING)
    print("\nPHASE 4: DETAILED EVALUATION (CACHED)")
    print("-" * 70)
    final_haiku, eval2_cost = evaluate_haiku_batch_cached(
        [h['text'] for h in first_pass], HAIKU_TO_KEEP, "Final evaluation"
    )
    total_cost += eval2_cost
    print(f"\n✓ Selected final {len(final_haiku)} haiku")
    print(f"✓ Average score: {sum(h['score'] for h in final_haiku) / len(final_haiku):.2f}")
    print(f"✓ Score range: {min(h['score'] for h in final_haiku):.2f} - {max(h['score'] for h in final_haiku):.2f}")
    print(f"✓ Cost: ${eval2_cost:.4f}")

    # PHASE 5: SAVE RESULTS
    print("\nPHASE 5: SAVING RESULTS")
    print("-" * 70)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save curated edition (top 250)
    output_file_curated = output_dir / f"haikus_curated_{timestamp}.txt"
    with open(output_file_curated, 'w', encoding='utf-8') as f:
        f.write(f"CURATED EDITION (Top {len(final_haiku)})\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Total generated: {len(generated)}\n")
        f.write(f"After filtering: {len(filtered)}\n")
        f.write(f"Final count: {len(final_haiku)}\n")
        f.write(f"Average score: {sum(h['score'] for h in final_haiku) / len(final_haiku):.2f}\n")
        f.write(f"Total cost: ${total_cost:.4f}\n")
        f.write(f"Caching: ENABLED\n")
        f.write("="*70 + "\n\n")

        for i, haiku_data in enumerate(final_haiku, 1):
            f.write(f"{i}.\n{haiku_data['text']}\n")
            f.write(f"Score: {haiku_data['score']:.2f}\n\n")

    print(f"✓ Saved curated edition to: {output_file_curated}")

    # Save extended edition (all filtered haiku)
    output_file_extended = output_dir / f"haikus_extended_{timestamp}.txt"
    with open(output_file_extended, 'w', encoding='utf-8') as f:
        f.write(f"EXTENDED EDITION (All {len(filtered)} Filtered Haiku)\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Total generated: {len(generated)}\n")
        f.write(f"After filtering: {len(filtered)}\n")
        f.write(f"Total cost: ${total_cost:.4f}\n")
        f.write(f"Caching: ENABLED\n")
        f.write("="*70 + "\n\n")

        for i, haiku_text in enumerate(filtered, 1):
            f.write(f"{i}.\n{haiku_text}\n\n")

    print(f"✓ Saved extended edition to: {output_file_extended}")

    # SUMMARY
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Generated:        {len(generated)} haiku")
    print(f"Filtered:         {len(filtered)} haiku")
    print(f"First pass:       {len(first_pass)} haiku")
    print(f"Final output:     {len(final_haiku)} haiku (curated)")
    print(f"Extended edition: {len(filtered)} haiku (all filtered)")
    print(f"Average score:    {sum(h['score'] for h in final_haiku) / len(final_haiku):.2f}")
    print(f"Total cost:       ${total_cost:.4f}")
    print(f"Time elapsed:     {elapsed:.1f}s")
    print(f"Caching:          ENABLED")
    print(f"\nOutput files:")
    print(f"  Curated:  {output_file_curated}")
    print(f"  Extended: {output_file_extended}")
    print("="*70)

    return output_file_curated, output_file_extended

if __name__ == "__main__":
    run_quality_filter()
