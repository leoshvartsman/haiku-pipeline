#!/usr/bin/env python3
"""
Retrofit titles and covers onto older books that were generated
before the title/cover pipeline existed.

Reads each untitled book's haiku, generates a proper title,
creates a cover prompt + DALL-E cover image, re-formats PDF/EPUB,
and updates the book index.

Usage:
    cd /Users/leo/shmindle/haiku-generator
    python3 retrofit_titles.py
"""

import re
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

# Reuse pipeline components
from generate_with_persona_and_theme import (
    generate_unique_title, get_existing_titles, save_title
)
from cover_prompt_generator import (
    generate_cover_prompt, save_cover_prompt,
    generate_cover_image, composite_cover_text
)
from generate_and_format import save_for_book_formatter, format_as_book
from book_indexer import load_index, save_index

HAIKU_OUTPUT = Path("haiku_output")
FORMATTER_OUTPUT = Path(__file__).resolve().parent.parent / "book_formatter" / "output"


def parse_haiku_from_file(filepath: Path) -> Tuple[List[str], str]:
    """Extract individual haiku texts and author from a haikus_*.txt file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract author
    author = ''
    for line in content.splitlines():
        if line.startswith('Author:'):
            author = line.replace('Author:', '').strip()
            break

    # Extract haiku (numbered: "1.\n<haiku text>\n\n2.\n...")
    haiku_list = []
    blocks = re.split(r'\n\d+\.\n', content)
    for block in blocks[1:]:  # skip header
        haiku = block.strip().split('\n\n')[0].strip()
        if haiku and len(haiku.splitlines()) >= 2:
            haiku_list.append(haiku)

    return haiku_list, author


def build_mock_persona(author: str) -> Dict:
    """Build a minimal persona dict from just the author name."""
    return {
        'name': author,
        'age': '',
        'occupation': '',
        'characteristic': '',
        'locations': {'current': ''},
    }


def build_mock_analyses(haiku_list: List[str]) -> List[Dict]:
    """Build minimal analysis dicts for cover prompt generation."""
    return [
        {
            'haiku': h,
            'text': h,
            'quality': 7.5,
            'season': 'timeless',
            'theme': 'nature',
            'tone': 'contemplative',
            'imagery': [],
        }
        for h in haiku_list
    ]


def find_untitled_books() -> List[Dict]:
    """Find books in the index that have generic 'Haikus by' titles."""
    index = load_index()
    untitled = []
    for entry in index:
        title = entry.get('title', '')
        if title.startswith('Haikus by '):
            untitled.append(entry)
    return untitled


def retrofit_book(entry: Dict) -> Dict:
    """Generate title, cover, and re-format a single untitled book.

    Returns updated entry dict and total cost.
    """
    old_title = entry['title']
    author = entry['author']
    book_id = entry['id']
    total_cost = 0.0

    # Find the haiku source file
    haiku_file = entry.get('files', {}).get('haiku_file', '')
    if not haiku_file:
        # Try to find it by timestamp
        haiku_file = str(HAIKU_OUTPUT / f"haikus_{book_id}.txt")
    haiku_path = Path(haiku_file)

    if not haiku_path.exists():
        print(f"  SKIP: {old_title} — haiku file not found: {haiku_path}")
        return entry

    # Parse haiku
    haiku_list, parsed_author = parse_haiku_from_file(haiku_path)
    if not haiku_list:
        print(f"  SKIP: {old_title} — no haiku found in file")
        return entry
    author = parsed_author or author

    print(f"\n{'─' * 60}")
    print(f"  Retrofitting: {old_title}")
    print(f"  Author: {author}  |  Haiku: {len(haiku_list)}")
    print(f"{'─' * 60}")

    # 1. Generate title
    print(f"  Generating title...")
    existing_titles = get_existing_titles()
    title = generate_unique_title(haiku_list, author, existing_titles)
    save_title(title)
    print(f"  ✓ Title: {title}")

    # 2. Generate cover prompt
    print(f"  Generating cover prompt...")
    persona = build_mock_persona(author)
    analyses = build_mock_analyses(haiku_list)
    cover_data, cover_cost = generate_cover_prompt(analyses, title, author, persona)
    total_cost += cover_cost
    cover_prompt_file = save_cover_prompt(cover_data, HAIKU_OUTPUT, title)
    print(f"  ✓ Cover prompt (${cover_cost:.4f})")

    # 3. Generate cover image
    cover_image_file = None
    print(f"  Generating cover image...")
    cover_image_file, image_cost = generate_cover_image(cover_data, HAIKU_OUTPUT, title)
    total_cost += image_cost
    if cover_image_file:
        print(f"  ✓ Cover image (${image_cost:.4f})")
        # Composite text
        print(f"  Compositing text...")
        cover_image_file = composite_cover_text(cover_image_file, title, author, cover_data)
    else:
        print(f"  ✗ Cover image failed or skipped")

    # 4. Re-save with new title and format as PDF/EPUB
    print(f"  Formatting PDF/EPUB...")
    formatter_file = save_for_book_formatter(haiku_list, title, author)
    book_success = format_as_book(formatter_file, title, author, cover_image=cover_image_file)

    # 5. Update entry
    entry['title'] = title
    entry['old_title'] = old_title

    files = entry.get('files', {})
    files['cover_prompt'] = str(cover_prompt_file)
    if cover_image_file:
        files['cover_image'] = str(cover_image_file)
    files['haiku_file'] = str(formatter_file)

    if book_success:
        title_slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        pdf_path = FORMATTER_OUTPUT / f"{title_slug}.pdf"
        epub_path = FORMATTER_OUTPUT / f"{title_slug}.epub"
        if pdf_path.exists():
            files['pdf'] = str(pdf_path)
        if epub_path.exists():
            files['epub'] = str(epub_path)
        print(f"  ✓ PDF/EPUB created")
    else:
        print(f"  ✗ PDF/EPUB formatting failed")

    entry['files'] = files
    entry['retrofit_cost'] = round(total_cost, 4)
    entry['retrofitted_at'] = datetime.now().isoformat()

    print(f"  Total cost: ${total_cost:.4f}")
    return entry


def main():
    untitled = find_untitled_books()

    if not untitled:
        print("No untitled books found. All books already have proper titles.")
        return

    print(f"\nFound {len(untitled)} untitled books to retrofit:")
    for entry in untitled:
        print(f"  - {entry['title']} ({entry['id']})")

    total_cost = 0.0
    index = load_index()

    # Build a map from id to index position
    id_map = {e['id']: i for i, e in enumerate(index)}

    for entry in untitled:
        updated = retrofit_book(entry)
        total_cost += updated.get('retrofit_cost', 0)

        # Update in the full index
        idx = id_map.get(updated['id'])
        if idx is not None:
            index[idx] = updated

    save_index(index)

    print(f"\n{'=' * 60}")
    print(f"  RETROFIT COMPLETE")
    print(f"  Books updated: {len(untitled)}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Index saved: {len(index)} books")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
