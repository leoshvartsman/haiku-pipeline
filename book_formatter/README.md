# Book Formatter

This tool converts a plain text file into a formatted book (PDF and EPUB), using Pandoc for conversions.

## Prerequisites (macOS)

- Install Pandoc: https://pandoc.org/installing.html
- Install a LaTeX engine for PDF (e.g., MacTeX or BasicTeX)
- (Optional) Install Calibre if you want MOBI: `brew install --cask calibre` (or from https://calibre-ebook.com/)
- Python 3.8+ and pip

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Basic:

```bash
python book_formatter.py examples/sample.txt --title "My Book" --author "Author Name" --out-dir ./out
```

Generate MOBI as well (requires `ebook-convert` from Calibre):

```bash
python book_formatter.py examples/sample.txt --formats pdf,epub,mobi --out-dir ./out
```

Custom templates directory (optional):

```bash
python book_formatter.py examples/sample.txt --templates ./templates --out-dir ./out
```

## How it works

- The script uses simple heuristics to detect chapter boundaries.
- It produces a temporary Markdown file with YAML metadata for Pandoc.
- Pandoc produces EPUB and PDF using the provided templates and CSS.

If you want me to add features like: cover image embedding, advanced chapter detection, TOC customization, or a GUI, tell me which one and I'll implement it.
