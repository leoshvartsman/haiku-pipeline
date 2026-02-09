#!/usr/bin/env python3
"""
Retrofit covers onto books that were generated before the new minimalist
cover style, or that have no covers at all.

Reads each book's haiku, generates a cover prompt + DALL-E cover image
with the new minimalist aesthetic, re-formats PDF/EPUB, and updates
the book index.

Usage:
    cd /Users/leo/shmindle/haiku-generator
    python3 retrofit_covers.py               # Process books 8-18
    python3 retrofit_covers.py --all         # Process all books without new covers
"""

import re
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

from retrofit_titles import parse_haiku_from_file, build_mock_persona, build_mock_analyses
from cover_prompt_generator import (
    generate_cover_prompt, save_cover_prompt,
    generate_cover_image, composite_cover_text
)
from generate_and_format import format_as_book
from book_indexer import load_index, save_index

HAIKU_OUTPUT = Path("haiku_output")
FORMATTER_OUTPUT = Path(__file__).resolve().parent.parent / "book_formatter" / "output"

# Books 8-18 by their index IDs
TARGET_IDS = [
    "20260206_062627",  # 8  Salt Stains and Marble Steps
    "20260206_071249",  # 9  Between Departures
    "20260206_075052",  # 10 Small Resurrections
    "20260206_083325",  # 11 Fluorescent Cathedrals
    "20260206_115755",  # 12 Dust Motes Dancing
    "20260206_131520",  # 13 Amber Morse and Cedar
    "20260206_143020",  # 14 Grease and Grace Notes
    "20260207_061049",  # 15 Threading the Monsoon
    "20260207_071331",  # 16 Thimbles and Wehrmacht Boots
    "20260207_091835",  # 17 Tuk-Tuk Saints
    "20260207_091959",  # 18 Spine Cracked at Hope
]


def retrofit_cover(entry: Dict) -> Dict:
    """Generate a new cover and re-format a single book.

    Returns updated entry dict.
    """
    title = entry['title']
    author = entry['author']
    total_cost = 0.0

    # Find the haiku source file
    haiku_file = entry.get('files', {}).get('haiku_file', '')
    if not haiku_file:
        haiku_file = str(HAIKU_OUTPUT / f"haikus_{entry['id']}.txt")
    haiku_path = Path(haiku_file)

    if not haiku_path.exists():
        print(f"  SKIP: {title} — haiku file not found: {haiku_path}")
        return entry

    # Parse haiku
    haiku_list, parsed_author = parse_haiku_from_file(haiku_path)
    if not haiku_list:
        print(f"  SKIP: {title} — no haiku found in file")
        return entry

    print(f"\n{'─' * 60}")
    print(f"  Retrofitting cover: {title}")
    print(f"  Author: {author}  |  Haiku: {len(haiku_list)}")
    print(f"{'─' * 60}")

    # 1. Generate cover prompt
    print(f"  Generating cover prompt...")
    persona = build_mock_persona(author)
    analyses = build_mock_analyses(haiku_list)
    cover_data, cover_cost = generate_cover_prompt(analyses, title, author, persona)
    total_cost += cover_cost
    cover_prompt_file = save_cover_prompt(cover_data, HAIKU_OUTPUT, title)
    print(f"  ✓ Cover prompt (${cover_cost:.4f})")

    # 2. Generate cover image
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

    # 3. Re-format PDF/EPUB with new cover
    print(f"  Formatting PDF/EPUB...")
    # Use existing haiku file for formatting
    book_success = format_as_book(haiku_path, title, author, cover_image=cover_image_file)

    # 4. Update entry
    files = entry.get('files', {})
    files['cover_prompt'] = str(cover_prompt_file)
    if cover_image_file:
        files['cover_image'] = str(cover_image_file)

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
    entry['cover_retrofit_cost'] = round(total_cost, 4)
    entry['cover_retrofitted_at'] = datetime.now().isoformat()

    print(f"  Total cost: ${total_cost:.4f}")
    return entry


def main():
    index = load_index()
    id_map = {e['id']: i for i, e in enumerate(index)}

    # Determine which books to process
    if '--all' in sys.argv:
        targets = [e for e in index if 'cover_image' not in e.get('files', {})]
    else:
        targets = [e for e in index if e['id'] in TARGET_IDS]

    if not targets:
        print("No books to process.")
        return

    print(f"\nFound {len(targets)} books to retrofit covers:")
    for entry in targets:
        has_cover = 'cover_image' in entry.get('files', {})
        status = "old cover" if has_cover else "no cover"
        print(f"  - {entry['title']} ({status})")

    total_cost = 0.0

    for entry in targets:
        updated = retrofit_cover(entry)
        total_cost += updated.get('cover_retrofit_cost', 0)

        # Update in the full index
        idx = id_map.get(updated['id'])
        if idx is not None:
            index[idx] = updated

    save_index(index)

    print(f"\n{'=' * 60}")
    print(f"  COVER RETROFIT COMPLETE")
    print(f"  Books updated: {len(targets)}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Index saved: {len(index)} books")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
