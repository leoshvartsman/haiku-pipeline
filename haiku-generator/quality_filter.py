#!/usr/bin/env python3
"""
Quality Filtering System for Haiku Generation
Generates 2500 haikus and keeps the best 250
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

# ============================================================================
# SYLLABLE COUNTING
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

    # Adjust for common patterns
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

    # Count syllables per line
    syllables = []
    for line in lines:
        words = re.findall(r'\b\w+\b', line)
        line_syllables = sum(count_syllables(w) for w in words)
        syllables.append(line_syllables)

    total = sum(syllables)

    # Accept various patterns
    if syllables in [(5,7,5), (3,5,3), (4,6,4), (5,6,5), (4,7,4), (6,7,6)]:
        return True

    # General brevity (allow flexibility)
    if 10 <= total <= 19 and all(2 <= s <= 9 for s in syllables):
        return True

    return False

# ============================================================================
# FILTERING
# ============================================================================

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
                    # Extract haiku (numbered format)
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
        # Must pass all checks
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

    # Determine season
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
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    # Parse haiku from response
    text = response.content[0].text
    haiku_list = []

    # Split by double newlines
    blocks = text.strip().split('\n\n')

    for block in blocks:
        # Remove any numbering
        block = re.sub(r'^\d+[\.)]\s*', '', block, flags=re.MULTILINE)

        lines = [l.strip() for l in block.split('\n') if l.strip()]

        # Valid haiku has 3 lines
        if len(lines) == 3:
            haiku_list.append('\n'.join(lines))

    # Calculate approximate cost (this is for one call, may need multiple)
    usage = response.usage
    cost = (usage.input_tokens * 0.80 / 1_000_000) + (usage.output_tokens * 4.00 / 1_000_000)

    return haiku_list, cost

def generate_multiple_batches(target_count: int = 2500) -> Tuple[List[str], float]:
    """Generate haiku in multiple batches to reach target"""
    all_haiku = []
    total_cost = 0.0

    # Generate in batches due to token limits
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

    return all_haiku[:target_count + 500], total_cost  # Get a bit extra for filtering

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_haiku_batch(haiku_list: List[str], target_count: int) -> Tuple[List[Dict], float]:
    """Evaluate haiku and return top scoring ones"""

    evaluation_prompt = f"""Evaluate these haiku and score each 0-10 based on:

Quality Criteria:
- Traditional haiku qualities (mono no aware, kigo, kire)
- Technical excellence (structure, imagery, moment)
- Contemporary merit (freshness, relevance, precision)

Return a JSON array with scores for each haiku.
Format: [{{"index": 0, "score": 8.5}}, ...]

Haiku to evaluate:

"""

    # Process in batches of 100
    batch_size = 100
    all_scored = []
    total_cost = 0.0

    print(f"\nEvaluating {len(haiku_list)} haiku...")

    for i in range(0, len(haiku_list), batch_size):
        batch = haiku_list[i:i+batch_size]

        # Format haiku with indices
        formatted = "\n\n".join([f"[{j}]\n{h}" for j, h in enumerate(batch)])

        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": evaluation_prompt + formatted
            }]
        )

        # Parse scores
        try:
            scores_text = response.content[0].text
            # Extract JSON array
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
            print(f"  Warning: Error parsing scores for batch, using fallback: {e}")
            # Fallback: assign neutral scores
            for h in batch:
                all_scored.append({'text': h, 'score': 7.0})

        # Calculate cost
        usage = response.usage
        cost = (usage.input_tokens * 0.80 / 1_000_000) + (usage.output_tokens * 4.00 / 1_000_000)
        total_cost += cost

        print(f"  Evaluated batch {i//batch_size + 1}: ${cost:.4f}")

    # Sort by score and return top
    all_scored.sort(key=lambda x: x['score'], reverse=True)
    top_haiku = all_scored[:target_count]

    return top_haiku, total_cost

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_quality_filter():
    """Run the complete quality filtering process"""
    print("="*70)
    print("HAIKU QUALITY FILTER - 2500 GENERATED, 250 KEPT")
    print("="*70)
    print(f"Target generation: {HAIKU_TO_GENERATE}")
    print(f"Target output: {HAIKU_TO_KEEP}")
    print(f"Model: {MODEL}")
    print("="*70)

    start_time = datetime.now()
    output_dir = Path("haiku_output")
    output_dir.mkdir(exist_ok=True)

    # Track costs
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
    print(f"✓ Removed duplicates: {len(generated) - len(filtered) - len([h for h in generated if not validate_structure(h) or has_cliche(h)])}")
    print(f"✓ Filtered to: {len(filtered)} haiku")

    # PHASE 3: FIRST EVALUATION PASS
    print("\nPHASE 3: FIRST EVALUATION PASS")
    print("-" * 70)
    first_pass, eval1_cost = evaluate_haiku_batch(filtered, FIRST_PASS_COUNT)
    total_cost += eval1_cost
    print(f"\n✓ Selected top {len(first_pass)} haiku")
    print(f"✓ Score range: {min(h['score'] for h in first_pass):.2f} - {max(h['score'] for h in first_pass):.2f}")
    print(f"✓ Cost: ${eval1_cost:.4f}")

    # PHASE 4: DETAILED EVALUATION
    print("\nPHASE 4: DETAILED EVALUATION")
    print("-" * 70)
    final_haiku, eval2_cost = evaluate_haiku_batch([h['text'] for h in first_pass], HAIKU_TO_KEEP)
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
    print(f"\nOutput files:")
    print(f"  Curated:  {output_file_curated}")
    print(f"  Extended: {output_file_extended}")
    print("="*70)

    return output_file_curated, output_file_extended

if __name__ == "__main__":
    run_quality_filter()
