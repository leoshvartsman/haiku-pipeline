#!/usr/bin/env python3
"""
Book Indexer — retroactive and ongoing catalog of generated haiku books.

Scans haiku_output/ for existing book files, extracts metadata, and builds
a book_index.json catalog. Also provides an `add_entry()` function for the
pipeline to call after each new run.

Usage:
    python3 book_indexer.py          # Rebuild index from existing files
    python3 book_indexer.py --print   # Print the current index
"""

import json
import re
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


INDEX_FILE = Path("haiku_output/book_index.json")
HAIKU_OUTPUT = Path("haiku_output")
FORMATTER_OUTPUT = Path(__file__).resolve().parent.parent / "book_formatter" / "output"


def load_index() -> List[Dict]:
    """Load existing index, or return empty list."""
    if INDEX_FILE.exists():
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_index(entries: List[Dict]):
    """Save the index to disk."""
    HAIKU_OUTPUT.mkdir(exist_ok=True)
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False, default=str)


def _parse_book_header(filepath: Path) -> Dict:
    """Extract title and author from a book_*.txt header line."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
    except (OSError, UnicodeDecodeError):
        return {'title': '', 'author': ''}

    # Header format: ===... Title  by Author  ===...
    cleaned = header.replace('=', '').strip()
    # Split on "by " — last occurrence
    parts = cleaned.rsplit(' by ', 1)
    if len(parts) == 2:
        title = parts[0].strip()
        author = parts[1].strip()
    else:
        title = cleaned
        author = ''

    return {'title': title, 'author': author}


def _count_haiku(filepath: Path) -> int:
    """Count numbered haiku in a book file."""
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if re.match(r'^\d+\.\s*$', line.strip()):
                    count += 1
    except (OSError, UnicodeDecodeError):
        pass
    return count


def _timestamp_from_filename(filename: str) -> Optional[str]:
    """Extract timestamp from book_YYYYMMDD_HHMMSS.txt pattern."""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        ts = match.group(1)
        try:
            dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
            return dt.isoformat()
        except ValueError:
            pass
    return None


def _find_related_files(timestamp: str, title: str) -> Dict:
    """Find cover prompt, cover image, formatter files, PDFs/EPUBs
    related to a given book by timestamp and title."""
    related = {}

    # Cover prompt JSON — match by title slug
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:50]
    cover_prompt = HAIKU_OUTPUT / f"cover_prompt_{safe_title}.json"
    if cover_prompt.exists():
        related['cover_prompt'] = str(cover_prompt)

    # Cover image
    cover_image = HAIKU_OUTPUT / f"cover_{safe_title}.png"
    if cover_image.exists():
        related['cover_image'] = str(cover_image)

    # Formatter-ready haiku file
    haiku_file = HAIKU_OUTPUT / f"haikus_{timestamp}.txt"
    if haiku_file.exists():
        related['haiku_file'] = str(haiku_file)

    # Annotated edition
    for suffix in ['_annotated']:
        annotated = HAIKU_OUTPUT / f"haikus_{timestamp}{suffix}.txt"
        if annotated.exists():
            related['annotated_file'] = str(annotated)

    # PDF and EPUB — slugified title in book_formatter/output
    if FORMATTER_OUTPUT.exists():
        title_slug = title.lower()
        title_slug = re.sub(r'[^a-z0-9]+', '-', title_slug).strip('-')

        pdf = FORMATTER_OUTPUT / f"{title_slug}.pdf"
        epub = FORMATTER_OUTPUT / f"{title_slug}.epub"
        if pdf.exists():
            related['pdf'] = str(pdf)
        if epub.exists():
            related['epub'] = str(epub)

        # Annotated edition
        ann_slug = f"{title_slug}-annotated-edition"
        ann_pdf = FORMATTER_OUTPUT / f"{ann_slug}.pdf"
        ann_epub = FORMATTER_OUTPUT / f"{ann_slug}.epub"
        if ann_pdf.exists():
            related['annotated_pdf'] = str(ann_pdf)
        if ann_epub.exists():
            related['annotated_epub'] = str(ann_epub)

    return related


def _load_cover_prompt_metadata(cover_prompt_path: str) -> Dict:
    """Extract metadata from a cover prompt JSON."""
    try:
        with open(cover_prompt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summary = data.get('collection_summary', {})
        return {
            'themes': summary.get('themes', {}),
            'tones': summary.get('tones', {}),
            'top_imagery': summary.get('top_imagery', [])[:5],
            'color_palette': data.get('color_palette', []),
        }
    except (OSError, json.JSONDecodeError):
        return {}


def build_retroactive_index() -> List[Dict]:
    """Scan haiku_output/ for book_*.txt files and build index entries."""
    entries = []
    book_files = sorted(HAIKU_OUTPUT.glob("book_*.txt"))

    for book_file in book_files:
        filename = book_file.name
        timestamp = filename.replace('book_', '').replace('.txt', '')
        iso_timestamp = _timestamp_from_filename(filename)

        header = _parse_book_header(book_file)
        title = header['title']
        author = header['author']
        haiku_count = _count_haiku(book_file)

        related = _find_related_files(timestamp, title)

        entry = {
            'id': timestamp,
            'title': title,
            'author': author,
            'generated_at': iso_timestamp,
            'haiku_count': haiku_count,
            'files': {
                'book_txt': str(book_file),
                **related,
            },
        }

        # Enrich from cover prompt if available
        if 'cover_prompt' in related:
            cover_meta = _load_cover_prompt_metadata(related['cover_prompt'])
            if cover_meta:
                entry['themes'] = cover_meta.get('themes', {})
                entry['tones'] = cover_meta.get('tones', {})
                entry['top_imagery'] = cover_meta.get('top_imagery', [])
                entry['color_palette'] = cover_meta.get('color_palette', [])

        entries.append(entry)

    return entries


def add_entry(
    title: str,
    author: str,
    persona: Dict,
    haiku_count: int,
    avg_score: float,
    total_cost: float,
    cost_breakdown: Dict,
    arc_eval: Optional[Dict],
    files: Dict,
    timestamp: Optional[str] = None,
):
    """Add a new entry to the book index. Called by the pipeline after each run.

    Args:
        title: Book title
        author: Author name
        persona: Full persona dict
        haiku_count: Number of haiku in the final book
        avg_score: Average quality score
        total_cost: Total API cost
        cost_breakdown: Per-feature cost dict
        arc_eval: Arc evaluation results (or None)
        files: Dict of file paths (book_txt, haiku_file, pdf, epub, etc.)
        timestamp: ISO timestamp string (auto-generated if None)
    """
    entries = load_index()

    ts = timestamp or datetime.now().isoformat()
    ts_short = datetime.now().strftime('%Y%m%d_%H%M%S')

    entry = {
        'id': ts_short,
        'title': title,
        'author': author,
        'generated_at': ts,
        'haiku_count': haiku_count,
        'avg_score': round(avg_score, 2),
        'total_cost': round(total_cost, 4),
        'cost_breakdown': {k: round(v, 4) for k, v in cost_breakdown.items()},
        'persona': {
            'name': persona.get('name', ''),
            'age': persona.get('age', ''),
            'occupation': persona.get('occupation', ''),
            'characteristic': persona.get('characteristic', ''),
            'location': persona.get('locations', {}).get('current', ''),
        },
        'files': {k: str(v) for k, v in files.items() if v},
    }

    if arc_eval:
        entry['arc_scores'] = arc_eval.get('scores', {})
        entry['arc_overall'] = arc_eval.get('overall_score', 0)
        entry['arc_summary'] = arc_eval.get('summary', '')

    entries.append(entry)
    save_index(entries)
    print(f"  Index updated: {len(entries)} books cataloged")
    return entry


def print_index():
    """Pretty-print the book index."""
    entries = load_index()
    if not entries:
        print("No books indexed yet.")
        return

    print(f"\n{'=' * 70}")
    print(f"  BOOK CATALOG — {len(entries)} books")
    print(f"{'=' * 70}\n")

    for i, entry in enumerate(entries, 1):
        title = entry.get('title', 'Untitled')
        author = entry.get('author', 'Unknown')
        date = entry.get('generated_at', '?')
        if 'T' in str(date):
            date = date.split('T')[0]
        count = entry.get('haiku_count', '?')
        score = entry.get('avg_score', '')
        cost = entry.get('total_cost', '')
        arc = entry.get('arc_overall', '')

        print(f"  {i:>3}. {title}")
        print(f"       by {author}")
        print(f"       Date: {date}  |  Haiku: {count}", end="")
        if score:
            print(f"  |  Avg score: {score}", end="")
        if cost:
            print(f"  |  Cost: ${cost}", end="")
        if arc:
            print(f"  |  Arc: {arc}/10", end="")
        print()

        # Show available formats
        files = entry.get('files', {})
        formats = []
        if 'pdf' in files:
            formats.append('PDF')
        if 'epub' in files:
            formats.append('EPUB')
        if 'cover_image' in files:
            formats.append('Cover')
        if 'annotated_pdf' in files:
            formats.append('Annotated')
        if formats:
            print(f"       Formats: {', '.join(formats)}")

        # Top themes
        themes = entry.get('themes', {})
        if themes:
            top = list(themes.keys())[:3]
            print(f"       Themes: {', '.join(top)}")

        print()


if __name__ == "__main__":
    if '--print' in sys.argv:
        print_index()
    else:
        print("Building retroactive book index...")
        entries = build_retroactive_index()
        save_index(entries)
        print(f"Indexed {len(entries)} books → {INDEX_FILE}")
        print()
        print_index()
