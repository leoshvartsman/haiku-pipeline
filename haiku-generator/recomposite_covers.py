#!/usr/bin/env python3
"""
Re-composite text on all book covers with updated text style.

Regenerates DALL-E cover images using the saved prompts from each
book's cover_prompt JSON, then composites the new text style
(smaller, top-right aligned) onto the fresh images.

Usage:
    cd /Users/leo/shmindle/haiku-generator
    python3 recomposite_covers.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from book_indexer import load_index
from cover_prompt_generator import generate_cover_image, composite_cover_text
from generate_and_format import format_as_book

HAIKU_OUTPUT = Path("haiku_output")


def main():
    index = load_index()

    processed = 0
    total_cost = 0.0
    for i, entry in enumerate(index, 1):
        files = entry.get('files', {})
        cover_prompt_path = files.get('cover_prompt')

        if not cover_prompt_path or not Path(cover_prompt_path).exists():
            print(f"  SKIP #{i}: {entry['title']} — no cover prompt JSON")
            continue

        title = entry['title']
        author = entry['author']

        print(f"\n{'─' * 60}")
        print(f"  #{i} Regenerating cover: {title}")
        print(f"{'─' * 60}")

        # Load cover prompt data (contains the DALL-E prompt)
        with open(cover_prompt_path, 'r') as f:
            cover_data = json.load(f)

        if not cover_data.get('dalle_prompt'):
            print(f"  SKIP — no dalle_prompt in JSON")
            continue

        # Delete old raw backup so composite_cover_text creates a fresh one
        cover_image_path = files.get('cover_image', '')
        if cover_image_path:
            raw_path = Path(str(cover_image_path).replace('.png', '_raw.png'))
            if raw_path.exists():
                raw_path.unlink()

        # Regenerate DALL-E image (fresh, no text)
        print(f"  Generating fresh DALL-E image...")
        cover_path, image_cost = generate_cover_image(cover_data, HAIKU_OUTPUT, title)
        total_cost += image_cost
        if not cover_path:
            print(f"  ✗ DALL-E generation failed")
            continue
        print(f"  ✓ Cover image (${image_cost:.3f})")

        # Composite new text (smaller, top-right)
        print(f"  Compositing text...")
        composite_cover_text(cover_path, title, author, cover_data)

        # Regenerate PDF/EPUB
        haiku_file = files.get('haiku_file', '')
        if haiku_file and Path(haiku_file).exists():
            print(f"  Formatting PDF/EPUB...")
            book_success = format_as_book(Path(haiku_file), title, author, cover_image=cover_path)
            if book_success:
                print(f"  ✓ PDF/EPUB regenerated")
            else:
                print(f"  ✗ PDF/EPUB formatting failed")

            # Regenerate annotated edition if it exists
            book_id = entry.get('id', '')
            annotated_file = files.get('annotated_file', '')
            if not annotated_file:
                annotated_file = str(HAIKU_OUTPUT / f"haikus_{book_id}_annotated.txt")
            if Path(annotated_file).exists():
                annotated_title = f"{title} — Annotated Edition"
                print(f"  Formatting annotated edition...")
                format_as_book(Path(annotated_file), annotated_title, author, cover_image=cover_path)
                print(f"  ✓ Annotated edition regenerated")
        else:
            print(f"  SKIP PDF — haiku file not found: {haiku_file}")

        processed += 1

    print(f"\n{'=' * 60}")
    print(f"  RE-COMPOSITE COMPLETE: {processed} covers updated")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
