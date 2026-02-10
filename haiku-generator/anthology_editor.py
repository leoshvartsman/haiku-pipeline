#!/usr/bin/env python3
"""
Anthology Editor for Haiku Collections

Implements professional anthology editing principles:
- Sequencing: Order poems for optimal reading experience
- Sectioning: Organize into meaningful groups
- Contextualizing: Provide framework and guidance

Based on principles from "The Craft of Poetry Anthology Editing"
"""

import anthropic
import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    timeout=120.0,  # 2 minute timeout to avoid hanging on stalled API calls
    max_retries=5,  # Auto-retry on 429/529 (overloaded) with exponential backoff
)

MODEL = "claude-sonnet-4-20250514"  # Using best model for analysis
SEQUENCING_MODEL = "claude-haiku-4-5-20251001"  # Cheaper model for sequencing pair comparisons


def _api_call_with_retry(**kwargs):
    """Wrap client.messages.create with explicit retry for 529 Overloaded errors."""
    for attempt in range(6):
        try:
            return client.messages.create(**kwargs)
        except anthropic.OverloadedError:
            if attempt == 5:
                raise
            wait = 2 ** attempt  # 1, 2, 4, 8, 16, 32 seconds
            print(f"    API overloaded, retrying in {wait}s (attempt {attempt + 1}/6)...")
            time.sleep(wait)
        except anthropic.RateLimitError:
            if attempt == 5:
                raise
            wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32, 64 seconds
            print(f"    Rate limited, retrying in {wait}s (attempt {attempt + 1}/6)...")
            time.sleep(wait)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_haiku_collection(haiku_list: List[str]) -> Dict:
    """Analyze collection to detect themes, seasons, tones, imagery"""

    print("Analyzing haiku collection for anthology structure...")

    # Process in batches for analysis
    batch_size = 50
    all_analyses = []

    for i in range(0, len(haiku_list), batch_size):
        batch = haiku_list[i:i+batch_size]
        formatted = "\n\n".join([f"[{j}]\n{h}" for j, h in enumerate(batch)])

        analysis_prompt = f"""Analyze these haiku for anthology editing. For each haiku, identify:

1. Primary SEASON (spring/summer/autumn/winter/none)
2. Primary THEME (nature/urban/technology/human/spiritual/etc)
3. TONE (quiet/intense/joyful/melancholic/contemplative/etc)
4. Key IMAGERY (what images/objects appear)
5. QUALITY score 0-10

Return JSON array: [{{"index": 0, "season": "spring", "theme": "nature", "tone": "contemplative", "imagery": ["rain", "earthworm", "stone"], "quality": 8.5}}, ...]

Haiku:

{formatted}"""

        response = _api_call_with_retry(
            model=MODEL,
            max_tokens=4000,
            temperature=0.3,
            messages=[{"role": "user", "content": analysis_prompt}]
        )

        try:
            analysis_text = response.content[0].text
            json_match = re.search(r'\[.*\]', analysis_text, re.DOTALL)
            if json_match:
                analyses = json.loads(json_match.group())
                # Adjust indices for batch offset
                for analysis in analyses:
                    analysis['global_index'] = i + analysis['index']
                    analysis['haiku'] = batch[analysis['index']]
                all_analyses.extend(analyses)
        except Exception as e:
            print(f"  Warning: Error analyzing batch {i//batch_size + 1}: {e}")
            # Create fallback analyses
            for j, haiku in enumerate(batch):
                all_analyses.append({
                    'global_index': i + j,
                    'haiku': haiku,
                    'season': 'none',
                    'theme': 'general',
                    'tone': 'contemplative',
                    'imagery': [],
                    'quality': 7.0
                })

        print(f"  Analyzed batch {i//batch_size + 1}/{(len(haiku_list)-1)//batch_size + 1}")

    return {
        'analyses': all_analyses,
        'total': len(all_analyses)
    }


def create_seasonal_sections(analyses: List[Dict]) -> Dict[str, List[Dict]]:
    """Group haiku by season"""
    seasons = {
        'spring': [],
        'summer': [],
        'autumn': [],
        'winter': [],
        'timeless': []  # For non-seasonal haiku
    }

    for analysis in analyses:
        season = analysis.get('season', 'none').lower()
        if season in seasons:
            seasons[season].append(analysis)
        else:
            seasons['timeless'].append(analysis)

    # Sort each season by quality score
    for season in seasons:
        seasons[season].sort(key=lambda x: x.get('quality', 7.0), reverse=True)

    return seasons


def create_thematic_sections(analyses: List[Dict]) -> Dict[str, List[Dict]]:
    """Group haiku by theme"""
    themes = defaultdict(list)

    for analysis in analyses:
        theme = analysis.get('theme', 'general')
        themes[theme].append(analysis)

    # Sort each theme by quality score
    for theme in themes:
        themes[theme].sort(key=lambda x: x.get('quality', 7.0), reverse=True)

    return dict(themes)


def sequence_within_section(haiku_analyses: List[Dict], strategy: str = "tonal") -> List[Dict]:
    """Sequence haiku within a section using specified strategy

    Strategies:
    - tonal: Vary intensity (wavelike pattern)
    - quality: Best poems in middle
    - thematic: Group by sub-themes
    - chronological: Natural progression (for seasonal)
    """

    if strategy == "quality":
        # Place strongest poems in middle third
        sorted_haiku = sorted(haiku_analyses, key=lambda x: x.get('quality', 7.0), reverse=True)
        n = len(sorted_haiku)

        if n < 10:
            return sorted_haiku

        # Divide into thirds
        third = n // 3
        top_third = sorted_haiku[:third]
        middle_third = sorted_haiku[third:2*third]
        bottom_third = sorted_haiku[2*third:]

        # Arrange: medium start, best middle, medium end
        return bottom_third[:len(bottom_third)//2] + top_third + bottom_third[len(bottom_third)//2:] + middle_third

    elif strategy == "tonal":
        # Wavelike pattern: vary intensity
        # Group by tone intensity
        quiet = []
        moderate = []
        intense = []

        quiet_tones = {'quiet', 'contemplative', 'peaceful', 'gentle', 'serene'}
        intense_tones = {'intense', 'joyful', 'melancholic', 'powerful', 'dramatic'}

        for h in haiku_analyses:
            tone = h.get('tone', '').lower()
            if any(t in tone for t in quiet_tones):
                quiet.append(h)
            elif any(t in tone for t in intense_tones):
                intense.append(h)
            else:
                moderate.append(h)

        # Create wavelike sequence
        result = []
        while quiet or moderate or intense:
            if quiet:
                result.append(quiet.pop(0))
            if intense:
                result.append(intense.pop(0))
            if moderate:
                result.append(moderate.pop(0))

        return result

    elif strategy == "llm":
        # LLM-based sequencing (Feature 2)
        sequenced, _ = sequence_with_llm(haiku_analyses)
        return sequenced

    else:  # chronological/thematic (use as-is)
        return haiku_analyses


def sequence_with_llm(
    haiku_analyses: List[Dict],
    section_name: str = ""
) -> Tuple[List[Dict], float]:
    """LLM-based greedy sequencing for optimal adjacency.

    For each position, evaluates candidate next-poems against the current
    poem and picks the one that creates the best resonance, contrast,
    or progression.

    Args:
        haiku_analyses: List of analyzed haiku dicts to sequence
        section_name: Name of the section (for context)

    Returns:
        Tuple of (sequenced_list, cost)
    """
    if len(haiku_analyses) <= 3:
        return haiku_analyses, 0.0

    total_cost = 0.0
    sequenced = []
    remaining = list(haiku_analyses)

    # Step 1: Ask LLM to pick the best opener
    opener_idx, opener_cost = _pick_best_opener(remaining, section_name)
    total_cost += opener_cost
    sequenced.append(remaining.pop(opener_idx))

    # Step 2: Greedy — for each position, evaluate top candidates
    total_poems = len(remaining) + 1
    while remaining:
        if len(remaining) == 1:
            sequenced.append(remaining.pop(0))
            break

        # Pick up to 8 candidates (to keep costs reasonable)
        candidates = remaining[:8] if len(remaining) > 8 else remaining

        best_idx, pair_cost = _pick_best_next(
            sequenced[-1], candidates, section_name
        )
        total_cost += pair_cost

        # Find the actual index in remaining
        chosen = candidates[best_idx]
        actual_idx = remaining.index(chosen)
        sequenced.append(remaining.pop(actual_idx))

        # Progress indicator every 10 poems
        if len(sequenced) % 10 == 0:
            print(f"    Sequencing {section_name}: {len(sequenced)}/{total_poems} poems placed...")

    return sequenced, total_cost


def _pick_best_opener(
    haiku_analyses: List[Dict],
    section_name: str
) -> Tuple[int, float]:
    """Ask LLM which haiku makes the strongest section opener."""
    if len(haiku_analyses) <= 1:
        return 0, 0.0

    # Present top candidates (by quality) for opener selection
    candidates = sorted(haiku_analyses, key=lambda x: x.get('quality', 7.0), reverse=True)[:10]

    listing = ""
    for i, h in enumerate(candidates):
        listing += f"[{i}]\n{h.get('haiku', h.get('text', ''))}\n\n"

    section_ctx = f' for the "{section_name}" section' if section_name else ''

    prompt = f"""Which of these haiku would make the strongest OPENING poem{section_ctx}
of a poetry collection?

The opener should:
- Immediately engage the reader
- Set an inviting tone
- Be strong but not the absolute best (save that for the middle)
- Create curiosity about what follows

{listing}

Return ONLY a JSON object: {{"index": 0, "reason": "brief reason"}}"""

    response = _api_call_with_retry(
        model=SEQUENCING_MODEL,
        max_tokens=200,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            chosen_candidate_idx = result.get('index', 0)
            # Map back to original list
            if chosen_candidate_idx < len(candidates):
                chosen = candidates[chosen_candidate_idx]
                original_idx = haiku_analyses.index(chosen)
            else:
                original_idx = 0
        else:
            original_idx = 0
    except (json.JSONDecodeError, AttributeError, ValueError):
        original_idx = 0

    usage = response.usage
    cost = (usage.input_tokens * 1.00 / 1_000_000) + (usage.output_tokens * 5.00 / 1_000_000)

    return original_idx, cost


def _pick_best_next(
    current: Dict,
    candidates: List[Dict],
    section_name: str
) -> Tuple[int, float]:
    """Ask LLM which candidate follows best after the current poem."""
    current_text = current.get('haiku', current.get('text', ''))

    listing = ""
    for i, h in enumerate(candidates):
        listing += f"[{i}]\n{h.get('haiku', h.get('text', ''))}\n\n"

    prompt = f"""Given this haiku that was just read:

{current_text}

Which of the following haiku should come NEXT for the best reading experience?

Consider:
- RESONANCE: Does it echo or deepen the previous image?
- CONTRAST: Does it create productive tension or shift?
- PROGRESSION: Does it move the reader forward?
- VARIETY: Is it different enough to avoid monotony?

Candidates:

{listing}

Return ONLY a JSON object: {{"index": 0, "relationship": "resonance|contrast|progression|surprise"}}"""

    response = _api_call_with_retry(
        model=SEQUENCING_MODEL,
        max_tokens=150,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            idx = result.get('index', 0)
            if idx >= len(candidates):
                idx = 0
        else:
            idx = 0
    except (json.JSONDecodeError, AttributeError):
        idx = 0

    usage = response.usage
    cost = (usage.input_tokens * 1.00 / 1_000_000) + (usage.output_tokens * 5.00 / 1_000_000)

    return idx, cost


# ============================================================================
# ANTHOLOGY BUILDING
# ============================================================================

def build_anthology(haiku_analyses: List[Dict],
                   title: str,
                   author: str = None,
                   sequencing_strategy: str = "tonal") -> str:
    """Build complete anthology as a single unsectioned sequence.

    Args:
        haiku_analyses: List of analyzed haiku dicts
        title: Anthology title
        author: Author name (optional)
        sequencing_strategy: "tonal", "quality", "llm", or "chronological"

    Returns:
        Formatted anthology text
    """

    anthology = []

    # Title page
    anthology.append("=" * 70)
    anthology.append(title.center(70))
    if author:
        anthology.append(f"by {author}".center(70))
    anthology.append("=" * 70)
    anthology.append("\n\n")

    # Sequence all haiku as one continuous flow
    sequenced = sequence_within_section(haiku_analyses, strategy=sequencing_strategy)

    for i, haiku_data in enumerate(sequenced, 1):
        anthology.append(f"{i}.\n")
        anthology.append(f"{haiku_data['haiku']}\n\n")

    # Closing
    anthology.append("\n\n")
    anthology.append("=" * 70)
    anthology.append("\n")
    anthology.append(f"End of {title}".center(70))
    anthology.append("\n")
    anthology.append("=" * 70)

    return "".join(anthology)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def edit_anthology(input_file: Path,
                  title: str,
                  author: str = None,
                  output_dir: Path = None) -> Path:
    """Edit haiku collection into professional anthology

    Args:
        input_file: Path to haiku text file
        title: Anthology title
        author: Author name (optional)
        output_dir: Output directory (defaults to haiku_output/anthologies/)

    Returns:
        Path to created anthology file
    """

    print("\n" + "=" * 70)
    print("ANTHOLOGY EDITOR")
    print("=" * 70)
    print(f"Input: {input_file}")
    print(f"Title: {title}")
    if author:
        print(f"Author: {author}")
    print("=" * 70 + "\n")

    # Read haiku from input file
    print("Reading haiku collection...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract haiku (numbered format)
    haiku_list = []
    pattern = r'\d+\.\n(.*?)\n\n'
    matches = re.findall(pattern, content, re.DOTALL)
    for match in matches:
        lines = [l.strip() for l in match.split('\n') if l.strip() and not l.strip().startswith('Score:')]
        if len(lines) == 3:
            haiku_list.append('\n'.join(lines))

    print(f"✓ Found {len(haiku_list)} haiku\n")

    if len(haiku_list) == 0:
        print("Error: No haiku found in input file")
        return None

    # Analyze collection
    analysis_results = analyze_haiku_collection(haiku_list)
    print(f"✓ Analyzed {analysis_results['total']} haiku\n")

    # Build anthology
    print("Building anthology...")
    anthology_text = build_anthology(
        analysis_results['analyses'],
        title,
        author
    )
    print("✓ Anthology structure complete\n")

    # Save anthology
    if output_dir is None:
        output_dir = Path("haiku_output/anthologies")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"anthology_{timestamp}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(anthology_text)

    print(f"✓ Saved anthology to: {output_file}")
    print(f"✓ Total length: {len(anthology_text)} characters")

    return output_file


# ============================================================================
# MULTI-BOOK DIVISION
# ============================================================================

def ai_decide_book_divisions(haiku_analyses: List[Dict],
                            num_books: int = 8,
                            poet_info: Dict = None) -> Dict:
    """Have AI analyze collection and decide how to divide into books

    Args:
        haiku_analyses: List of analyzed haiku
        num_books: Number of books to create
        poet_info: Optional dict with poet name, career span info for career stage decisions

    Returns:
        Dictionary with book_plan containing themes/titles for each book
    """

    print(f"\nAsking anthology editor to devise {num_books}-book division strategy...")

    # Summarize the collection for the AI
    seasons = create_seasonal_sections(haiku_analyses)
    themes = create_thematic_sections(haiku_analyses)

    season_summary = {k: len(v) for k, v in seasons.items() if v}
    theme_summary = {k: len(v) for k, v in themes.items() if v}

    # Get sample haiku for context
    sample_haiku = [h['haiku'] for h in haiku_analyses[:20]]
    samples_text = "\n\n".join(sample_haiku)

    # Build poet context for career stage decisions
    poet_context = ""
    if poet_info:
        poet_context = f"""
POET INFORMATION:
- Name: {poet_info.get('name', 'Unknown')}
- Current age: {poet_info.get('current_age', 50)}
- Active since: {poet_info.get('active_since', 2000)}
- Total years as poet: {poet_info.get('total_years', 20)}
- Age when started writing: {poet_info.get('start_age', 25)}

CAREER STAGES AVAILABLE:
- emerging (0-20% of career): Finding voice, experimental, raw, earnest
- developing (20-40%): Building technique, growing confidence
- established (40-60%): Mature voice, recognized style, refined
- mature (60-80%): Deep wisdom, distilled expression, quiet authority
- master (80-100%): Transcendent simplicity, effortless depth

You may assign different career stages to different books to show the poet's evolution,
or focus on a particular period. This affects the voice and themes of each book.
"""

    prompt = f"""You are a professional anthology editor. You have a collection of {len(haiku_analyses)} haiku to organize into {num_books} distinct books, plus a combined anthology.

COLLECTION STATISTICS:
- Seasonal distribution: {json.dumps(season_summary, indent=2)}
- Thematic distribution: {json.dumps(theme_summary, indent=2)}
{poet_context}
SAMPLE HAIKU FROM COLLECTION:
{samples_text}

Your task: Design {num_books} distinct books that together form a cohesive series. Each book should have:
1. A unique thematic focus or organizing principle
2. A compelling title
3. Clear criteria for which haiku belong in it
4. A career stage (if poet info provided) - books can span the poet's journey

Consider creative organizing principles beyond just seasons/themes:
- Emotional journeys (grief to hope, solitude to connection)
- Times of day (dawn, midday, dusk, night)
- Human experiences (love, loss, work, play, aging)
- Natural elements (water, earth, fire, air)
- Sensory focus (visual, auditory, tactile)
- Philosophical concepts (impermanence, presence, emptiness)
- Life stages or moments
- The poet's evolution over their career

Return a JSON object with this structure:
{{
    "series_concept": "Brief description of the overall series concept",
    "books": [
        {{
            "number": 1,
            "title": "Book Title",
            "theme": "primary_theme_key",
            "criteria": "Description of what haiku belong here",
            "career_stage": "emerging",  // optional: emerging, developing, established, mature, or master
            "career_percentage": 15,     // optional: specific percentage through career (0-100)
            "seasons": ["spring", "summer"],  // optional: limit to specific seasons
            "themes": ["nature", "urban"],    // optional: limit to specific themes
            "tones": ["contemplative", "quiet"],  // optional: prefer certain tones
            "keywords": ["rain", "morning"]   // optional: keywords to look for in imagery
        }},
        ...
    ]
}}

Be creative and thoughtful. Create books that readers would want to explore individually."""

    response = _api_call_with_retry(
        model=MODEL,
        max_tokens=3000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        response_text = response.content[0].text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            book_plan = json.loads(json_match.group())
            print(f"✓ Anthology editor devised series: \"{book_plan.get('series_concept', 'Untitled Series')}\"")
            return book_plan
    except Exception as e:
        print(f"  Warning: Could not parse AI response, using fallback: {e}")

    # Fallback plan
    return {
        "series_concept": "A journey through seasons and themes",
        "books": [
            {"number": 1, "title": "Spring Awakening", "theme": "spring", "seasons": ["spring"]},
            {"number": 2, "title": "Summer Light", "theme": "summer", "seasons": ["summer"]},
            {"number": 3, "title": "Autumn Shadows", "theme": "autumn", "seasons": ["autumn"]},
            {"number": 4, "title": "Winter Silence", "theme": "winter", "seasons": ["winter"]},
            {"number": 5, "title": "Natural World", "theme": "nature", "themes": ["nature"]},
            {"number": 6, "title": "Urban Moments", "theme": "urban", "themes": ["urban"]},
            {"number": 7, "title": "Human Touch", "theme": "human", "themes": ["human"]},
            {"number": 8, "title": "Timeless", "theme": "timeless", "themes": ["spiritual", "general"]}
        ]
    }


def assign_haiku_to_books(haiku_analyses: List[Dict], book_plan: Dict) -> List[Dict]:
    """Assign haiku to books based on the AI-generated plan

    Returns:
        List of book dictionaries with assigned haiku
    """

    books = []
    assigned_indices = set()

    for book_spec in book_plan.get('books', []):
        book_haiku = []

        for analysis in haiku_analyses:
            if analysis['global_index'] in assigned_indices:
                continue

            # Check if haiku matches book criteria
            matches = True

            # Season filter
            if 'seasons' in book_spec and book_spec['seasons']:
                if analysis.get('season', '').lower() not in [s.lower() for s in book_spec['seasons']]:
                    matches = False

            # Theme filter
            if matches and 'themes' in book_spec and book_spec['themes']:
                if analysis.get('theme', '').lower() not in [t.lower() for t in book_spec['themes']]:
                    matches = False

            # Tone filter
            if matches and 'tones' in book_spec and book_spec['tones']:
                haiku_tone = analysis.get('tone', '').lower()
                if not any(t.lower() in haiku_tone for t in book_spec['tones']):
                    matches = False

            # Keyword filter (check imagery)
            if matches and 'keywords' in book_spec and book_spec['keywords']:
                imagery = [img.lower() for img in analysis.get('imagery', [])]
                haiku_text = analysis.get('haiku', '').lower()
                if not any(kw.lower() in haiku_text or kw.lower() in ' '.join(imagery)
                          for kw in book_spec['keywords']):
                    matches = False

            if matches:
                book_haiku.append(analysis)
                assigned_indices.add(analysis['global_index'])

        books.append({
            'number': book_spec.get('number', len(books) + 1),
            'title': book_spec.get('title', f"Book {len(books) + 1}"),
            'theme': book_spec.get('theme', 'general'),
            'criteria': book_spec.get('criteria', ''),
            'career_stage': book_spec.get('career_stage'),
            'career_percentage': book_spec.get('career_percentage'),
            'haiku': book_haiku,
            'organization': 'ai_curated'
        })

    # Assign remaining haiku to the book with fewest entries
    remaining = [a for a in haiku_analyses if a['global_index'] not in assigned_indices]
    if remaining and books:
        # Distribute remaining haiku evenly
        for i, analysis in enumerate(remaining):
            smallest_book = min(books, key=lambda b: len(b['haiku']))
            smallest_book['haiku'].append(analysis)

    return books


def divide_into_books(haiku_analyses: List[Dict],
                     num_books: int = 8,
                     title_prefix: str = "Haiku Collection",
                     author: str = None,
                     use_ai_division: bool = True,
                     poet_info: Dict = None) -> List[Dict]:
    """Intelligently divide haiku collection into multiple books

    Args:
        haiku_analyses: List of analyzed haiku
        num_books: Number of books to create (default 8)
        title_prefix: Prefix for book titles
        author: Author name
        use_ai_division: If True, let AI decide book themes (default True)
        poet_info: Optional dict with poet career info for career stage decisions

    Returns:
        List of book dictionaries with title, haiku, and metadata
    """

    print(f"\nDividing {len(haiku_analyses)} haiku into {num_books} books...")

    if use_ai_division:
        # Let AI decide how to divide the books
        book_plan = ai_decide_book_divisions(haiku_analyses, num_books, poet_info)
        books = assign_haiku_to_books(haiku_analyses, book_plan)

        # Update titles with prefix
        for book in books:
            if title_prefix and not book['title'].startswith(title_prefix):
                book['full_title'] = f"{title_prefix}: {book['title']}"
            else:
                book['full_title'] = book['title']
            book['subtitle'] = f"Book {book['number']} of {num_books}"

        print(f"\n✓ AI-curated {len(books)} book divisions:")
        for book in books:
            print(f"  Book {book['number']}: {book['title']} ({len(book['haiku'])} haiku)")

        return books

    # Fallback: Original static division logic
    seasons = create_seasonal_sections(haiku_analyses)
    themes = create_thematic_sections(haiku_analyses)

    books = []

    # Seasonal books (Books 1-4)
    seasonal_order = ['spring', 'summer', 'autumn', 'winter']
    for i, season in enumerate(seasonal_order, 1):
        season_haiku = seasons.get(season, [])
        if season_haiku:
            books.append({
                'number': i,
                'title': f"{title_prefix}: {season.capitalize()}",
                'full_title': f"{title_prefix}: {season.capitalize()}",
                'subtitle': f"Book {i} of {num_books}",
                'haiku': season_haiku,
                'organization': 'seasonal',
                'focus': season
            })

    # Thematic books (Books 5-8)
    theme_counts = [(theme, len(haiku)) for theme, haiku in themes.items()]
    theme_counts.sort(key=lambda x: x[1], reverse=True)
    top_themes = theme_counts[:4]

    for i, (theme, count) in enumerate(top_themes, 5):
        theme_haiku = themes[theme]
        theme_title = theme.replace('_', ' ').title()
        books.append({
            'number': i,
            'title': f"{title_prefix}: {theme_title}",
            'full_title': f"{title_prefix}: {theme_title}",
            'subtitle': f"Book {i} of {num_books}",
            'haiku': theme_haiku,
            'organization': 'thematic',
            'focus': theme
        })

    # Pad if we don't have 8 books
    while len(books) < num_books:
        remaining = [h for h in haiku_analyses if not any(h in book['haiku'] for book in books)]
        if remaining:
            books.append({
                'number': len(books) + 1,
                'title': f"{title_prefix}: Mixed Selections",
                'full_title': f"{title_prefix}: Mixed Selections",
                'subtitle': f"Book {len(books) + 1} of {num_books}",
                'haiku': remaining,
                'organization': 'quality',
                'focus': 'mixed'
            })
        else:
            break

    print(f"✓ Created {len(books)} book divisions")
    for book in books:
        print(f"  Book {book['number']}: {book['title']} ({len(book['haiku'])} haiku)")

    return books


def build_single_book(book: Dict, author: str = None, poet_info: Dict = None) -> str:
    """Build a single book with anthology editing

    Args:
        book: Book dictionary with title, haiku, career_stage, etc.
        author: Author name
        poet_info: Optional dict with poet career info

    Returns:
        Formatted book text
    """

    title = book.get('full_title', book.get('title', 'Untitled'))
    haiku_analyses = book.get('haiku', [])

    if not haiku_analyses:
        return ""

    anthology = []

    # Get career stage info
    career_stage = book.get('career_stage')
    career_percentage = book.get('career_percentage')

    # Title page
    anthology.append("=" * 70)
    anthology.append(title.center(70))
    if book.get('subtitle'):
        anthology.append(book['subtitle'].center(70))
    if author:
        anthology.append(f"by {author}".center(70))

    # Add career stage subtitle if present
    if career_stage and poet_info:
        start_age = poet_info.get('start_age', 25)
        total_years = poet_info.get('total_years', 20)
        pct = career_percentage if career_percentage else {'emerging': 10, 'developing': 30, 'established': 50, 'mature': 70, 'master': 90}.get(career_stage, 50)
        years_in = int(total_years * pct / 100)
        poet_age = start_age + years_in
        anthology.append(f"({career_stage.title()} Period, Age {poet_age})".center(70))

    anthology.append("=" * 70)
    anthology.append("\n\n")

    # Generate book-specific introduction
    print(f"  Generating introduction for {title}...")
    sample_haiku = [h['haiku'] for h in haiku_analyses[:5]]
    criteria = book.get('criteria', f"Haiku exploring {book.get('theme', 'various themes')}")

    # Build career stage context for the intro
    career_context = ""
    if career_stage:
        stage_descriptions = {
            'emerging': "These poems come from the poet's early years, when they were still finding their voice. The work is marked by experimentation, raw emotion, and a sense of discovery.",
            'developing': "This collection represents the poet's developing period, when their technique was maturing and their distinctive style was beginning to emerge.",
            'established': "These haiku come from the poet's established period, when their voice had fully matured and their mastery of the form was evident.",
            'mature': "This volume contains work from the poet's mature years, characterized by deep wisdom, economy of expression, and profound simplicity.",
            'master': "These are late works from the poet's mastery period, marked by transcendent simplicity and effortless depth."
        }
        career_context = f"\n\nCareer context: {stage_descriptions.get(career_stage, '')}"

    intro_prompt = f"""Write a 200-300 word introduction for a haiku book titled "{title}".

This book contains {len(haiku_analyses)} haiku with this focus: {criteria}
{career_context}

Sample haiku from this collection:

{chr(10).join(sample_haiku)}

The introduction should:
1. Welcome readers to this specific volume
2. Explain what makes this collection distinctive
3. If this represents a specific period in the poet's career, acknowledge where these poems fall in their artistic journey
4. Suggest how to approach reading these haiku
5. Set the mood for the journey ahead

Tone: Warm, accessible, contemplative."""

    response = _api_call_with_retry(
        model=MODEL,
        max_tokens=800,
        temperature=0.7,
        messages=[{"role": "user", "content": intro_prompt}]
    )

    intro = response.content[0].text.strip()
    anthology.append("# INTRODUCTION\n\n")
    anthology.append(intro)
    anthology.append("\n\n")
    anthology.append("-" * 70)
    anthology.append("\n\n")

    # Sequence the haiku
    sequenced = sequence_within_section(haiku_analyses, strategy="tonal")

    # Add haiku
    for i, haiku_data in enumerate(sequenced, 1):
        anthology.append(f"{i}.\n")
        anthology.append(f"{haiku_data['haiku']}\n\n")

    # Closing
    anthology.append("\n")
    anthology.append("=" * 70)
    anthology.append("\n")
    anthology.append(f"End of {title}".center(70))
    anthology.append("\n")
    anthology.append("=" * 70)

    return "".join(anthology)


def build_combined_anthology(books: List[Dict],
                            series_title: str,
                            author: str = None,
                            poet_info: Dict = None) -> str:
    """Build combined anthology from all books

    Args:
        books: List of book dictionaries
        series_title: Title for the combined anthology
        author: Author name
        poet_info: Optional dict with poet career info

    Returns:
        Formatted combined anthology text
    """

    total_haiku = sum(len(book.get('haiku', [])) for book in books)

    anthology = []

    # Title page
    anthology.append("=" * 70)
    anthology.append(series_title.center(70))
    anthology.append("The Complete Collection".center(70))
    if author:
        anthology.append(f"by {author}".center(70))
    anthology.append("=" * 70)
    anthology.append("\n\n")

    # Generate series introduction
    print(f"Generating introduction for combined anthology...")

    # Build book descriptions with career stages
    book_descriptions = []
    for book in books:
        desc = book.get('title', f"Book {book['number']}")
        if book.get('career_stage'):
            desc += f" ({book['career_stage']} period)"
        book_descriptions.append(desc)

    # Check if books span career stages
    career_stages_used = [b.get('career_stage') for b in books if b.get('career_stage')]
    career_journey_context = ""
    if career_stages_used:
        unique_stages = list(dict.fromkeys(career_stages_used))  # Preserve order, remove duplicates
        if len(unique_stages) > 1:
            career_journey_context = f"\n\nThis anthology spans the poet's artistic journey from their {unique_stages[0]} period through their {unique_stages[-1]} years, allowing readers to witness the evolution of a poetic voice."

    intro_prompt = f"""Write a 400-500 word introduction for a complete haiku anthology that combines {len(books)} individual volumes into one collection.

Series title: {series_title}
Total haiku: {total_haiku}
Individual volumes: {', '.join(book_descriptions)}
{f"Author: {author}" if author else ""}
{career_journey_context}

The introduction should:
1. Welcome readers to this comprehensive collection
2. Explain the structure (divided into {len(books)} thematic sections)
3. If the books span different career periods, describe the arc of the poet's development
4. Describe the journey readers will take through the volumes
5. Offer guidance on how to read - straight through or exploring sections
6. Reflect on the achievement of this complete collection

Tone: Celebratory but accessible, honoring the tradition while welcoming new readers."""

    response = _api_call_with_retry(
        model=MODEL,
        max_tokens=1200,
        temperature=0.7,
        messages=[{"role": "user", "content": intro_prompt}]
    )

    intro = response.content[0].text.strip()
    anthology.append("# INTRODUCTION\n\n")
    anthology.append(intro)
    anthology.append("\n\n")

    # Table of contents
    anthology.append("# CONTENTS\n\n")
    for book in books:
        title = book.get('title', f"Book {book['number']}")
        count = len(book.get('haiku', []))
        stage_note = ""
        if book.get('career_stage'):
            stage_note = f" [{book['career_stage']}]"
        anthology.append(f"  {book['number']}. {title}{stage_note} ({count} haiku)\n")
    anthology.append("\n")
    anthology.append("=" * 70)
    anthology.append("\n\n")

    # Include each book as a section
    for book in books:
        title = book.get('title', f"Book {book['number']}")
        haiku_analyses = book.get('haiku', [])

        if not haiku_analyses:
            continue

        # Section header
        anthology.append(f"\n\n{'#' * 70}\n")
        anthology.append(f"# BOOK {book['number']}: {title.upper()}\n")
        if book.get('career_stage'):
            anthology.append(f"# {book['career_stage'].upper()} PERIOD\n")
        anthology.append(f"{'#' * 70}\n\n")

        # Brief section intro
        criteria = book.get('criteria', '')
        if criteria:
            anthology.append(f"_{criteria}_\n\n")

        anthology.append("-" * 70)
        anthology.append("\n\n")

        # Sequence and add haiku
        sequenced = sequence_within_section(haiku_analyses, strategy="tonal")

        for i, haiku_data in enumerate(sequenced, 1):
            anthology.append(f"{i}.\n")
            anthology.append(f"{haiku_data['haiku']}\n\n")

    # Closing
    anthology.append("\n\n")
    anthology.append("=" * 70)
    anthology.append("\n")
    anthology.append(f"End of {series_title}".center(70))
    anthology.append(f"{total_haiku} haiku across {len(books)} volumes".center(70))
    anthology.append("\n")
    anthology.append("=" * 70)

    return "".join(anthology)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI for anthology editor"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python anthology_editor.py <input_file> <title> [author]")
        print()
        print("Arguments:")
        print("  input_file    Path to haiku collection file")
        print("  title         Anthology title")
        print("  author        Author name (optional)")
        print()
        print("Example:")
        print("  python anthology_editor.py haiku_output/haikus_curated_20240101.txt \"Seasonal Haiku\" \"Jane Doe\"")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    title = sys.argv[2]
    author = sys.argv[3] if len(sys.argv) > 3 else None

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    edit_anthology(input_file, title, author)


if __name__ == "__main__":
    main()
