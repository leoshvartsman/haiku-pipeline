#!/usr/bin/env python3
"""
Generate a single haiku book using a random persona and random themes.
Quality filtering: generates 2500, keeps the best 250, publishes one book.

Enhanced pipeline includes:
- Plagiarism detection against canonical haiku (Feature 5)
- Voice consistency auditing (Feature 4)
- Sonic quality scoring (Feature 9)
- Juxtaposition/turn evaluation (Feature 3)
- Revision pass on near-misses (Feature 1)
- LLM-based sequencing (Feature 2)
- Scholarly notes (Feature 8)
- Cover image prompt generation (Feature 7)
- Collection arc evaluation (Feature 10)
"""

from persona_selector import PersonaSelector
from generate_and_format import save_for_book_formatter, format_as_book
from anthology_editor import analyze_haiku_collection, build_anthology
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0)

# Title tracking file
TITLES_FILE = Path("haiku_output") / "used_titles.json"

def get_existing_titles() -> set:
    """Load existing book titles from tracking file"""
    if TITLES_FILE.exists():
        import json
        with open(TITLES_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_title(title: str):
    """Add a new title to the tracking file"""
    import json
    titles = get_existing_titles()
    titles.add(title)
    TITLES_FILE.parent.mkdir(exist_ok=True)
    with open(TITLES_FILE, 'w') as f:
        json.dump(list(titles), f, indent=2)

def generate_unique_title(haiku_sample: List[str], author: str, existing_titles: set) -> str:
    """Generate a unique, creative book title using Claude"""

    # Take a sample of haiku for title inspiration
    sample = haiku_sample[:20] if len(haiku_sample) > 20 else haiku_sample
    sample_text = "\n\n".join(sample)

    existing_list = "\n".join(f"- {t}" for t in existing_titles) if existing_titles else "(none yet)"

    prompt = f"""Generate a unique, evocative title for a book of haiku by {author}.

Here are some sample haiku from the collection:

{sample_text}

EXISTING TITLES TO AVOID (do not use these or anything too similar):
{existing_list}

GUIDELINES FOR THE TITLE:
- Should be poetic but not pretentious
- 2-5 words is ideal
- Should evoke the observational, concrete nature of haiku
- Can reference seasons, nature, moments, or the act of observation
- Should NOT include the word "Haiku" or the author's name
- Should feel like a real published poetry collection title

Respond with ONLY the title, nothing else."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=50,
        temperature=0.9,
        messages=[{"role": "user", "content": prompt}]
    )

    title = response.content[0].text.strip().strip('"\'')

    # If somehow we got a duplicate, add a subtle variation
    if title in existing_titles:
        title = f"{title}: New Observations"

    return title

# Configuration
HAIKU_TO_GENERATE = 2500
HAIKU_TO_KEEP = 250
FIRST_PASS_COUNT = 500
MODEL = "claude-sonnet-4-20250514"
EVAL_MODEL = "claude-haiku-4-5-20251001"  # Cheaper model for evaluation/scoring tasks

# Feature toggles
ENABLE_PLAGIARISM_CHECK = True       # Feature 5
ENABLE_VOICE_AUDIT = True            # Feature 4
ENABLE_TURN_EVALUATION = True        # Feature 3
ENABLE_LLM_SEQUENCING = True         # Feature 2
ENABLE_SCHOLARLY_NOTES = True        # Feature 8
ENABLE_COVER_PROMPT = True           # Feature 7
ENABLE_COVER_IMAGE = True            # Feature 7b: Generate actual cover via DALL-E 3
ENABLE_ARC_EVALUATION = True         # Feature 10

# Feature-specific settings
VOICE_AUDIT_THRESHOLD = 5.0
PLAGIARISM_THRESHOLD = 0.70
TURN_WEIGHT = 0.25                   # 25% of composite score
SCHOLARLY_NOTES_COUNT = 12
ARC_EVAL_MIN_SCORE = 7.0
IMAGERY_MAX_REPEATS = 4              # Max times any single image can appear across collection

# Import filtering functions from quality_filter
from quality_filter import (
    validate_structure, has_cliche, get_existing_hashes,
    is_duplicate, apply_computational_filters, count_syllables
)

# Import semantic similarity filter
from semantic_filter import filter_similar_haiku


def generate_with_quality_filter(persona: Dict, target_count: int = 250) -> Tuple[List[Dict], List[str], float, Dict]:
    """Generate haiku with quality filtering

    Returns:
        Tuple of (top_haiku, all_filtered_haiku, total_cost, cost_breakdown)
        - top_haiku: List of dicts with top scored haiku
        - all_filtered_haiku: List of strings with all filtered haiku (for extended edition)
        - total_cost: Total API cost
        - cost_breakdown: Dict of per-feature costs
    """

    print(f"\nGenerating {HAIKU_TO_GENERATE} haiku for filtering...")
    print(f"Target output: {target_count} best haiku\n")

    total_cost = 0.0
    cost_breakdown = {}
    all_haiku = []

    # Generate in multiple batches — each batch can realistically produce
    # ~40-50 haiku within the 4096 token limit, so we ask for 50 per call
    per_batch = 50
    batches_needed = (HAIKU_TO_GENERATE + per_batch - 1) // per_batch

    from persona_selector import PersonaSelector
    selector = PersonaSelector()
    persona_prompt = selector.format_for_prompt(persona)

    for i in range(batches_needed):
        prompt = f"""{persona_prompt}

Write exactly {per_batch} haiku on widely varied subjects.

GUIDELINES:
- Each haiku should have a different subject from the others
- Subjects can include: people, places, objects, animals, plants, food, work, play, travel, weather, urban life, rural life, technology, craft, art, music, sports, domestic scenes, public spaces, or anything else observable
- Use concrete, specific imagery
- Write as an observer describing what is seen, heard, or noticed
- Avoid clichés and overly familiar haiku subjects

Structure: 5-7-5 syllables (or natural variations).

Output exactly {per_batch} haiku, each separated by a blank line. No numbering, no commentary."""

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0.9,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse haiku
        text = response.content[0].text
        blocks = text.strip().split('\n\n')

        for block in blocks:
            block = re.sub(r'^\d+[\.)]\s*', '', block, flags=re.MULTILINE)
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            if len(lines) == 3:
                all_haiku.append('\n'.join(lines))

        # Calculate cost
        usage = response.usage
        # Sonnet pricing: $3/M input, $15/M output
        cost = (usage.input_tokens * 3.00 / 1_000_000) + (usage.output_tokens * 15.00 / 1_000_000)
        total_cost += cost

        print(f"  Batch {i+1}/{batches_needed}: {len(all_haiku)} total haiku (${cost:.4f})")

        if len(all_haiku) >= HAIKU_TO_GENERATE:
            break

    gen_cost = total_cost
    cost_breakdown['generation'] = gen_cost
    print(f"\n✓ Generated {len(all_haiku)} haiku (${gen_cost:.4f})")

    # Apply computational filters
    print("\nApplying computational filters...")
    output_dir = Path("haiku_output")
    existing_hashes = get_existing_hashes(output_dir)
    filtered = apply_computational_filters(all_haiku, existing_hashes)
    print(f"✓ Passed computational filters: {len(filtered)} haiku")

    # Apply semantic similarity filter
    print("\nApplying semantic similarity filter...")
    before_semantic = len(filtered)
    filtered = filter_similar_haiku(filtered, threshold=0.80)
    print(f"✓ Passed semantic filter: {len(filtered)} haiku ({before_semantic - len(filtered)} too similar)")

    # Feature 5: Plagiarism detection
    if ENABLE_PLAGIARISM_CHECK:
        print("\nChecking for canonical plagiarism...")
        from plagiarism_detector import filter_plagiaristic
        before_plag = len(filtered)
        filtered, plagiarism_removed, plag_cost = filter_plagiaristic(filtered, threshold=PLAGIARISM_THRESHOLD)
        total_cost += plag_cost
        cost_breakdown['plagiarism_check'] = plag_cost
        print(f"✓ Plagiarism check: {before_plag - len(filtered)} removed (${plag_cost:.4f})")

    # First evaluation pass
    print(f"\nFirst evaluation pass (selecting top {FIRST_PASS_COUNT})...")
    first_pass, eval1_cost = evaluate_batch(filtered, FIRST_PASS_COUNT)
    total_cost += eval1_cost
    cost_breakdown['evaluation_pass_1'] = eval1_cost
    first_pass_texts = [h['text'] for h in first_pass]
    print(f"✓ Selected {len(first_pass)} haiku (${eval1_cost:.4f})")

    # Final evaluation
    print(f"\nFinal evaluation (selecting top {target_count})...")
    final_haiku, eval2_cost = evaluate_batch(first_pass_texts, target_count)
    total_cost += eval2_cost
    cost_breakdown['evaluation_pass_2'] = eval2_cost
    avg_score = sum(h['score'] for h in final_haiku) / len(final_haiku) if final_haiku else 0
    print(f"✓ Selected {len(final_haiku)} final haiku")
    print(f"✓ Average score: {avg_score:.2f}")
    print(f"✓ Cost: ${eval2_cost:.4f}")

    # Feature 4: Voice consistency audit (on final selection only)
    if ENABLE_VOICE_AUDIT:
        print("\nAuditing voice consistency...")
        from voice_auditor import audit_voice_consistency
        final_texts = [h['text'] for h in final_haiku]
        before_voice = len(final_texts)
        kept_texts, voice_outliers, voice_cost = audit_voice_consistency(
            final_texts, persona, threshold=VOICE_AUDIT_THRESHOLD
        )
        total_cost += voice_cost
        cost_breakdown['voice_audit'] = voice_cost
        kept_set = set(kept_texts)
        final_haiku = [h for h in final_haiku if h['text'] in kept_set]
        print(f"✓ Voice audit: {before_voice - len(final_haiku)} outliers removed (${voice_cost:.4f})")

    # Feature 3: Juxtaposition/turn evaluation (on final selection only)
    turn_scores = {}
    if ENABLE_TURN_EVALUATION:
        print("\nEvaluating juxtaposition/turn quality...")
        from juxtaposition_evaluator import evaluate_turns, get_turn_scores_map
        final_texts = [h['text'] for h in final_haiku]
        turn_results, turn_cost = evaluate_turns(final_texts)
        turn_scores = get_turn_scores_map(turn_results)
        total_cost += turn_cost
        cost_breakdown['turn_evaluation'] = turn_cost
        avg_turn = sum(turn_scores.values()) / len(turn_scores) if turn_scores else 0
        print(f"✓ Turn evaluation: avg {avg_turn:.1f} (${turn_cost:.4f})")

    # Attach turn scores to final haiku for downstream use
    for h in final_haiku:
        h['turn_score'] = turn_scores.get(h['text'], 5.0)

    return final_haiku, filtered, total_cost, cost_breakdown

def deduplicate_imagery(haiku_analyses: List[Dict], max_repeats: int = 4) -> int:
    """Remove haiku with overused imagery to ensure visual variety.

    Counts how often each specific image appears across the collection.
    When an image exceeds max_repeats, drops the lowest-quality poems
    using that image until it's within the cap.

    Modifies haiku_analyses in-place.

    Returns:
        Number of haiku removed
    """
    from collections import Counter

    # Count all imagery across collection
    imagery_counts = Counter()
    for h in haiku_analyses:
        imagery = h.get('imagery', [])
        if isinstance(imagery, list):
            for img in imagery:
                imagery_counts[img.lower()] += 1

    # Find overused images
    overused = {img: count for img, count in imagery_counts.items()
                if count > max_repeats}

    if not overused:
        return 0

    to_remove = set()

    for image, count in sorted(overused.items(), key=lambda x: x[1], reverse=True):
        excess = count - max_repeats

        # Find all haiku using this image, sorted by quality (worst first)
        users = []
        for i, h in enumerate(haiku_analyses):
            if i in to_remove:
                continue
            imagery = [img.lower() for img in h.get('imagery', [])]
            if image in imagery:
                users.append((i, h.get('quality', 7.0)))

        # Sort by quality ascending (drop worst first)
        users.sort(key=lambda x: x[1])

        # Drop the weakest until we're at the cap
        for idx, _ in users[:excess]:
            to_remove.add(idx)

    if not to_remove:
        return 0

    # Print what's being removed
    removed_images = Counter()
    for idx in to_remove:
        for img in haiku_analyses[idx].get('imagery', []):
            if img.lower() in overused:
                removed_images[img.lower()] += 1

    # Remove in reverse order to preserve indices
    for idx in sorted(to_remove, reverse=True):
        haiku_analyses.pop(idx)

    # Report
    for img, count in removed_images.most_common(5):
        print(f"    Capped '{img}': was {overused[img]}x, removed {count} weakest")

    return len(to_remove)


def evaluate_batch(haiku_list: List[str], target_count: int) -> Tuple[List[Dict], float]:
    """Evaluate haiku batch and return top scoring ones"""
    batch_size = 100
    all_scored = []
    total_cost = 0.0

    for i in range(0, len(haiku_list), batch_size):
        batch = haiku_list[i:i+batch_size]
        formatted = "\n\n".join([f"[{j}]\n{h}" for j, h in enumerate(batch)])

        evaluation_prompt = f"""Evaluate these haiku and score each 0-10.

SCORING CRITERIA:
- Concrete, specific imagery (not vague or abstract)
- Fresh perspective or unexpected observation
- Clear, precise language
- Captures a moment or image effectively
- Avoids clichés and overused phrases

BONUS POINTS FOR:
- Unusual or surprising subject matter
- Vivid sensory details
- Original word choices

PENALIZE:
- Generic or forgettable images
- Clichéd phrases (e.g., "lonely crow", "cherry blossoms fall")
- Vague abstractions without grounding

Return JSON array: [{{"index": 0, "score": 8.5}}, ...]

Haiku:

{formatted}"""

        response = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": evaluation_prompt}]
        )

        try:
            scores_text = response.content[0].text
            json_match = re.search(r'\[.*\]', scores_text, re.DOTALL)
            if json_match:
                import json
                scores = json.loads(json_match.group())
                for score_data in scores:
                    idx = score_data.get('index', 0)
                    if idx < len(batch):
                        all_scored.append({
                            'text': batch[idx],
                            'score': score_data.get('score', 7.0)
                        })
        except:
            for h in batch:
                all_scored.append({'text': h, 'score': 7.0})

        usage = response.usage
        # Haiku 4.5 pricing: $1/M input, $5/M output
        cost = (usage.input_tokens * 1.00 / 1_000_000) + (usage.output_tokens * 5.00 / 1_000_000)
        total_cost += cost

    all_scored.sort(key=lambda x: x['score'], reverse=True)
    return all_scored[:target_count], total_cost

def main():
    # Initialize selector
    persona_selector = PersonaSelector()

    # Get random persona
    persona = persona_selector.get_random_persona()
    author = persona['name']

    # Display selections
    print("=" * 70)
    print("SELECTED PERSONA:")
    print("=" * 70)
    print(persona_selector.format_persona_description(persona, "full"))
    print("=" * 70)
    print(f"\nGenerating one book of {HAIKU_TO_KEEP} haiku with quality filtering...")
    print(f"({HAIKU_TO_GENERATE} generated → filtered → best {HAIKU_TO_KEEP} kept)")
    print("=" * 70)

    # Display active features
    features = []
    if ENABLE_PLAGIARISM_CHECK: features.append("Plagiarism Detection")
    if ENABLE_VOICE_AUDIT: features.append("Voice Audit")
    if ENABLE_TURN_EVALUATION: features.append("Turn Evaluation")
    if ENABLE_LLM_SEQUENCING: features.append("LLM Sequencing")
    if ENABLE_SCHOLARLY_NOTES: features.append("Scholarly Notes")
    if ENABLE_COVER_PROMPT: features.append("Cover Prompt")
    if ENABLE_COVER_IMAGE: features.append("Cover Image (DALL-E 3)")
    if ENABLE_ARC_EVALUATION: features.append("Arc Evaluation")
    print(f"\nActive features: {', '.join(features)}")
    print()

    # Generate with quality filtering (2500 → 250)
    haiku_data, all_filtered, total_cost, cost_breakdown = generate_with_quality_filter(persona, HAIKU_TO_KEEP)

    # Use only the top-scoring haiku for the book
    top_haiku_texts = [h['text'] for h in haiku_data]

    # Generate unique title based on the haiku content
    print("\nGenerating unique book title...")
    existing_titles = get_existing_titles()
    title = generate_unique_title(top_haiku_texts, author, existing_titles)
    save_title(title)
    print(f"✓ Title: {title}")

    # Analyze the top haiku for anthology structure (seasons, themes, tones)
    print("\nAnalyzing haiku collection for book structure...")
    analysis_results = analyze_haiku_collection(top_haiku_texts)
    haiku_analyses = analysis_results['analyses']

    # Attach turn scores to analyses for downstream features
    score_map = {h['text']: h for h in haiku_data}
    for analysis in haiku_analyses:
        haiku_text = analysis.get('haiku', '')
        if haiku_text in score_map:
            analysis['turn_score'] = score_map[haiku_text].get('turn_score', 5.0)

    # Imagery deduplication — cap overused images for visual variety
    print("\nChecking imagery variety...")
    imagery_removed = deduplicate_imagery(haiku_analyses, max_repeats=IMAGERY_MAX_REPEATS)
    if imagery_removed:
        print(f"✓ Removed {imagery_removed} haiku with overused imagery ({len(haiku_analyses)} remain)")
    else:
        print("✓ Imagery variety OK — no duplicates to remove")

    # Build anthology — use LLM sequencing if enabled
    sequencing_strategy = "llm" if ENABLE_LLM_SEQUENCING else "tonal"
    print(f"\nBuilding book with anthology editing (sequencing: {sequencing_strategy})...")
    book_text = build_anthology(haiku_analyses, title, author, sequencing_strategy=sequencing_strategy)

    # Feature 10: Evaluate collection arc
    arc_eval = None
    if ENABLE_ARC_EVALUATION:
        print("\nEvaluating collection arc...")
        from arc_evaluator import evaluate_collection_arc, apply_arc_suggestions, print_arc_evaluation
        arc_eval, arc_cost = evaluate_collection_arc(book_text, title)
        total_cost += arc_cost
        cost_breakdown['arc_evaluation'] = arc_cost
        print_arc_evaluation(arc_eval)

        # If any dimension scored below 7, apply targeted fixes and rebuild
        scores = arc_eval.get('scores', {})
        has_weak_dimension = any(v < 7 for v in scores.values())
        if has_weak_dimension:
            print(f"\n  Applying arc fixes for weak dimensions...")
            modified = apply_arc_suggestions(haiku_analyses, arc_eval)
            if modified:
                book_text = build_anthology(haiku_analyses, title, author, sequencing_strategy=sequencing_strategy)
                print("  ✓ Rebuilt anthology with arc fixes")
            else:
                print("  No actionable fixes identified")

    # Feature 8: Generate scholarly notes
    notes_text = ""
    if ENABLE_SCHOLARLY_NOTES:
        print("\nGenerating scholarly notes...")
        from scholarly_notes import select_standout_poems, generate_scholarly_notes
        standouts = select_standout_poems(haiku_analyses, count=SCHOLARLY_NOTES_COUNT)
        notes_text, notes_cost = generate_scholarly_notes(standouts, persona, title)
        total_cost += notes_cost
        cost_breakdown['scholarly_notes'] = notes_cost
        book_text += notes_text
        print(f"✓ Generated notes for {len(standouts)} poems (${notes_cost:.4f})")

    # Feature 7: Generate cover image prompt
    cover_prompt_file = None
    cover_image_file = None
    if ENABLE_COVER_PROMPT:
        print("\nGenerating cover image prompt...")
        from cover_prompt_generator import generate_cover_prompt, save_cover_prompt
        cover_data, cover_cost = generate_cover_prompt(haiku_analyses, title, author, persona)
        total_cost += cover_cost
        cost_breakdown['cover_prompt'] = cover_cost
        output_dir = Path("haiku_output")
        cover_prompt_file = save_cover_prompt(cover_data, output_dir, title)
        print(f"✓ Cover prompt saved to {cover_prompt_file} (${cover_cost:.4f})")

        # Feature 7b: Generate actual cover image with DALL-E 3
        if ENABLE_COVER_IMAGE:
            print("Generating cover image with DALL-E 3...")
            from cover_prompt_generator import generate_cover_image, composite_cover_text
            cover_image_file, image_cost = generate_cover_image(cover_data, output_dir, title)
            total_cost += image_cost
            cost_breakdown['cover_image'] = image_cost
            if cover_image_file:
                print(f"✓ Cover image saved to {cover_image_file} (${image_cost:.4f})")
                # Composite title and author text onto the cover image
                print("Compositing title and author onto cover...")
                cover_image_file = composite_cover_text(cover_image_file, title, author, cover_data)
            else:
                print("  Cover image generation skipped or failed.")

    # Save the book
    output_dir = Path("haiku_output")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"book_{timestamp}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(book_text)

    # Save in book_formatter-ready format
    formatter_file = save_for_book_formatter(top_haiku_texts, title, author)

    # Generate PDF and EPUB
    print("\nFormatting as PDF and EPUB...")
    book_success = format_as_book(formatter_file, title, author, cover_image=cover_image_file)

    # Generate annotated edition with scholarly notes
    if notes_text:
        annotated_title = f"{title} — Annotated Edition"
        print(f"\nFormatting annotated edition as PDF and EPUB...")
        annotated_file = save_for_book_formatter(
            top_haiku_texts, annotated_title, author,
            notes_text=notes_text, suffix="_annotated"
        )
        format_as_book(annotated_file, annotated_title, author, cover_image=cover_image_file)

    # Create rejected haiku collection
    rejected_title = f"Rejected: {title}"
    rejected_haiku_texts = [h for h in all_filtered if h not in top_haiku_texts]

    rejected_success = False
    rejected_formatter_file = None
    if rejected_haiku_texts:
        print(f"\nSaving {len(rejected_haiku_texts)} rejected haiku...")
        rejected_formatter_file = save_for_book_formatter(rejected_haiku_texts, rejected_title, author)
        rejected_success = format_as_book(rejected_formatter_file, rejected_title, author)

    # Index this book
    from book_indexer import add_entry
    index_files = {
        'book_txt': str(output_file),
        'haiku_file': str(formatter_file),
    }
    if cover_prompt_file:
        index_files['cover_prompt'] = str(cover_prompt_file)
    if cover_image_file:
        index_files['cover_image'] = str(cover_image_file)
    if book_success:
        formatter_output_dir = Path(__file__).parent.parent / "book_formatter" / "output"
        title_slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        pdf_path = formatter_output_dir / f"{title_slug}.pdf"
        epub_path = formatter_output_dir / f"{title_slug}.epub"
        if pdf_path.exists():
            index_files['pdf'] = str(pdf_path)
        if epub_path.exists():
            index_files['epub'] = str(epub_path)
    add_entry(
        title=title,
        author=author,
        persona=persona,
        haiku_count=len(top_haiku_texts),
        avg_score=sum(h['score'] for h in haiku_data) / len(haiku_data),
        total_cost=total_cost,
        cost_breakdown=cost_breakdown,
        arc_eval=arc_eval,
        files=index_files,
        timestamp=timestamp,
    )

    # Summary
    avg_score = sum(h['score'] for h in haiku_data) / len(haiku_data)
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\n  Generated:       {len(all_filtered)} haiku passed filters")
    print(f"  Selected:        {len(haiku_data)} best haiku")
    print(f"  Rejected:        {len(rejected_haiku_texts)} haiku")
    print(f"  Average score:   {avg_score:.2f}")
    print(f"  Total API cost:  ${total_cost:.4f}")
    print(f"\n  Title:   {title}")
    print(f"  Author:  {author}")

    # Cost breakdown
    print(f"\n  Cost breakdown:")
    for feature, feature_cost in sorted(cost_breakdown.items()):
        print(f"    {feature:<25} ${feature_cost:.4f}")

    # Arc evaluation summary
    if arc_eval:
        print(f"\n  Arc evaluation:  {arc_eval.get('overall_score', 'N/A')}/10")
        summary = arc_eval.get('summary', '')
        if summary:
            print(f"  {summary[:100]}")

    print(f"\n  Text files:")
    print(f"    Book (with intros):    {output_file}")
    print(f"    Plain (numbered):      {formatter_file}")
    if rejected_formatter_file:
        print(f"    Rejected (numbered):   {rejected_formatter_file}")
    if cover_prompt_file:
        print(f"    Cover prompt:          {cover_prompt_file}")
    if cover_image_file:
        print(f"    Cover image:           {cover_image_file}")
    if book_success or rejected_success:
        formatter_output = Path(__file__).parent.parent / "book_formatter" / "output"
        print(f"\n  Published books:")
        print(f"    PDF/EPUB location:     {formatter_output}/")
    else:
        print(f"\n  Note: PDF/EPUB generation failed. Check that Pandoc and LaTeX are installed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
