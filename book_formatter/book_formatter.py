#!/usr/bin/env python3
"""
Simple book formatter that converts a plain text file into a nicely formatted
Markdown file and then calls Pandoc to produce PDF and EPUB (and optionally
MOBI via `ebook-convert` from Calibre).

Usage examples are in README.md. This script uses simple heuristics to split
chapters and produces a Pandoc-friendly Markdown file with YAML metadata.
"""

import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import click


def find_executable(name):
    return shutil.which(name)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_poetry_headers(text: str) -> str:
    """Remove Author: and Title: headers from poetry/haiku files."""
    lines = text.splitlines()
    cleaned_lines = []
    skip_until_content = True

    for line in lines:
        if skip_until_content:
            # Skip Author: and Title: lines at the beginning
            if line.strip().startswith(('Author:', 'Title:')):
                continue
            elif line.strip() == '':
                continue
            else:
                skip_until_content = False
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def detect_chapters(text: str):
    """Return list of (title, body) for each detected chapter.

    Heuristics used (in order):
    - Lines that start with 'CHAPTER' or 'Chapter' followed by number/title
    - Lines in ALL CAPS of reasonable length (<= 60 chars) followed by blank line
    - Otherwise split by long sequences of blank lines (3+)
    """
    lines = text.splitlines()
    chapter_indices = []

    chapter_title_pattern = re.compile(r"^(CHAPTER|Chapter)\b.*")
    for i, line in enumerate(lines):
        if chapter_title_pattern.match(line):
            chapter_indices.append((i, line.strip()))
            continue
        stripped = line.strip()
        if (
            stripped
            and stripped.upper() == stripped
            and 1 <= len(stripped) <= 60
            and i + 1 < len(lines)
            and lines[i + 1].strip() == ""
        ):
            chapter_indices.append((i, stripped))

    if chapter_indices:
        chapters = []
        for idx, (start_i, title) in enumerate(chapter_indices):
            end_i = chapter_indices[idx + 1][0] if idx + 1 < len(chapter_indices) else len(lines)
            body = "\n".join(lines[start_i + 1 : end_i]).strip()
            chapters.append((title, body))
        return chapters

    # fallback: split on 3+ newlines
    parts = re.split(r"\n{3,}", text)
    chapters = []
    for i, part in enumerate(parts):
        title = f"Chapter {i+1}"
        chapters.append((title, part.strip()))
    return chapters


def build_markdown(title: str, author: str, chapters, cover_path=None, is_poetry=False):
    meta = []
    meta.append("---")
    if title:
        meta.append(f"title: \"{title}\"")
    if author:
        meta.append(f"author: \"{author}\"")
    if cover_path:
        meta.append(f"cover-image: \"{cover_path}\"")
    meta.append("lang: en")
    meta.append("---\n")

    md = "\n".join(meta)

    # Add title page for poetry books (only when no cover image, since cover has text)
    if is_poetry and title and not cover_path:
        md += "\n<div class='title-page'>\n\n"
        md += "<hr class='title-rule-top' />\n\n"
        md += f"# {title}\n\n"
        md += "<hr class='title-rule-middle' />\n\n"
        if author:
            md += f"<p class='author-name'>{author}</p>\n\n"
        md += "<hr class='title-rule-bottom' />\n\n"
        md += "</div>\n\n"
        md += "<div style='page-break-after: always;'></div>\n\n"

    for chap_title, body in chapters:
        if not is_poetry:  # Skip chapter titles for poetry
            md += f"\n# {chap_title}\n\n"

        if is_poetry:
            # For poetry, preserve line breaks within stanzas
            # Split by double newlines (separate poems/stanzas)
            stanzas = body.split('\n\n')
            processed_stanzas = []
            for stanza in stanzas:
                # Replace single newlines with markdown line breaks (two spaces + newline)
                processed_stanza = stanza.replace('\n', '  \n')
                processed_stanzas.append(processed_stanza)
            md += '\n\n'.join(processed_stanzas) + "\n"
        else:
            # normalize body: ensure paragraphs separated by blank line
            md += re.sub(r"\n{2,}", "\n\n", body) + "\n"
    return md


def write_temp_md(md_text: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_file = out_dir / f"book_temp_{uuid.uuid4().hex[:8]}.md"
    md_file.write_text(md_text, encoding="utf-8")
    return md_file


def is_poetry_book(title: str, input_path: Path) -> bool:
    """Detect if this is a poetry/haiku book based on title or filename."""
    poetry_keywords = ['haiku', 'poem', 'poetry', 'verse', 'sonnet']
    title_lower = title.lower() if title else ""
    filename_lower = input_path.stem.lower()

    return any(keyword in title_lower or keyword in filename_lower
               for keyword in poetry_keywords)


def run_pandoc(md_file: Path, out_dir: Path, title: str, author: str, formats, template_dir: Path, page_size: str, font_size: str, enable_toc: bool = True, cover_path=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Map user-friendly page sizes to LaTeX geometry names
    papersize_map = {"letter": "letterpaper", "a4": "a4paper", "a5": "a5paper"}
    latex_papersize = papersize_map.get(page_size.lower(), page_size)

    # Select templates based on poetry detection
    css_file = "ebook-poetry.css" if not enable_toc else "ebook.css"
    latex_template = "latex-poetry.tex" if not enable_toc else "latex-template.tex"

    # EPUB
    if "epub" in formats:
        epub_out = out_dir / f"{slugify(title or md_file.stem)}.epub"
        cmd = [
            "pandoc",
            str(md_file),
            "-o",
            str(epub_out),
        ]
        if enable_toc:
            cmd.append("--toc")
        cmd.extend([
            "--metadata",
            f"title={title}",
            "--css",
            str(template_dir / css_file),
        ])
        # Add cover image for EPUB
        if cover_path:
            cmd.extend(["--epub-cover-image", str(cover_path)])
        # Ensure pandoc can find CSS, images and font files inside the repo/templates
        resource_paths = [".", str(template_dir)]
        if cover_path:
            resource_paths.append(str(Path(cover_path).parent))
        cmd.extend(["--resource-path", ":".join(resource_paths)])
        # Embed font files so EPUBs render consistently across readers
        fonts_dir = template_dir / "fonts"
        if fonts_dir.is_dir():
            for font_file in sorted(fonts_dir.glob("*.ttf")) + sorted(fonts_dir.glob("*.otf")):
                cmd.extend(["--epub-embed-font", str(font_file)])
        if run_cmd(cmd, exit_on_fail=False):
            outputs['epub'] = epub_out
        else:
            click.echo("Warning: EPUB generation failed, continuing...")

    # PDF (via LaTeX)
    if "pdf" in formats:
        pdf_out = out_dir / f"{slugify(title or md_file.stem)}.pdf"
        cmd = [
            "pandoc",
            str(md_file),
            "-o",
            str(pdf_out),
            "--template",
            str(template_dir / latex_template),
            "--pdf-engine=xelatex",
            "-V", f"papersize:{latex_papersize}",
            "-V", f"fontsize={font_size}",
            "-V", f"fonts-dir={str(template_dir / 'fonts')}",
        ]
        if enable_toc:
            cmd.append("--toc")
        # Ensure pandoc/xelatex can resolve any images or font files referenced in templates
        resource_paths = [".", str(template_dir)]
        if cover_path:
            resource_paths.append(str(Path(cover_path).parent))
        cmd.extend(["--resource-path", ":".join(resource_paths)])
        if run_cmd(cmd, exit_on_fail=False):
            outputs['pdf'] = pdf_out
        else:
            click.echo("Warning: PDF generation failed, continuing...")

    # MOBI via ebook-convert (Calibre)
    if "mobi" in formats:
        if 'epub' not in outputs:
            # need an EPUB first
            epub_temp = out_dir / f"{slugify(title or md_file.stem)}.epub"
            cmd = [
                "pandoc",
                str(md_file),
                "-o",
                str(epub_temp),
            ]
            if enable_toc:
                cmd.append("--toc")
            cmd.extend([
                "--metadata",
                f"title={title}",
                "--css",
                str(template_dir / css_file),
            ])
            # Make sure resources referenced from templates are found for the temporary EPUB
            cmd.extend(["--resource-path", f".:{str(template_dir)}"])
            # Embed font files so EPUBs render consistently across readers
            fonts_dir = template_dir / "fonts"
            if fonts_dir.is_dir():
                for font_file in sorted(fonts_dir.glob("*.ttf")) + sorted(fonts_dir.glob("*.otf")):
                    cmd.extend(["--epub-embed-font", str(font_file)])
            run_cmd(cmd)
            outputs['epub'] = epub_temp
        mobi_out = out_dir / f"{slugify(title or md_file.stem)}.mobi"
        converter = find_executable('ebook-convert')
        if not converter:
            click.echo("Warning: 'ebook-convert' not found; install Calibre to enable MOBI conversion.")
        else:
            cmd = [converter, str(outputs['epub']), str(mobi_out)]
            run_cmd(cmd)
            outputs['mobi'] = mobi_out

    return outputs


def run_cmd(cmd, exit_on_fail=True):
    click.echo("Running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        click.echo(f"Command failed (exit code {e.returncode}):")
        if e.stderr:
            # Show last portion of stderr for debugging
            stderr_lines = e.stderr.strip().splitlines()
            for line in stderr_lines[-20:]:
                click.echo(f"  {line}")
        if exit_on_fail:
            sys.exit(1)
        return False


def slugify(s: str) -> str:
    if not s:
        return "book"
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "book"


@click.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--title", "title", default=None, help="Title of the book")
@click.option("--author", "author", default=None, help="Author name")
@click.option("--cover", "cover", default=None, help="Path to a cover image")
@click.option("--out-dir", "out_dir", default="./out", help="Output directory")
@click.option("--formats", "formats", default="pdf,epub", help="Comma-separated output formats: pdf,epub,mobi")
@click.option("--page-size", "page_size", default="letter", help="Paper size for PDF (e.g., letter, a4)")
@click.option("--font-size", "font_size", default="12pt", help="Base font size for PDF")
@click.option("--templates", "templates", default=None, help="Path to templates directory (css, latex template)")
def main(input_file, title, author, cover, out_dir, formats, page_size, font_size, templates):
    input_path = Path(input_file)
    out_dir = Path(out_dir)
    template_dir = Path(templates) if templates else Path(__file__).resolve().parent / "templates"

    text = read_text(input_path)

    if not title:
        title = input_path.stem

    # Auto-detect poetry books and disable TOC
    enable_toc = not is_poetry_book(title, input_path)

    # Strip Author:/Title: headers from poetry files
    if not enable_toc:
        text = strip_poetry_headers(text)

    chapters = detect_chapters(text)
    if not enable_toc:
        click.echo("Poetry/Haiku book detected - table of contents disabled")

    md = build_markdown(title, author, chapters, cover_path=cover, is_poetry=not enable_toc)
    md_file = write_temp_md(md, out_dir)

    fmt_set = set([f.strip().lower() for f in formats.split(',') if f.strip()])

    if not find_executable('pandoc'):
        click.echo("Error: 'pandoc' not found. Install Pandoc (https://pandoc.org/) before running this script.")
        sys.exit(1)

    outputs = run_pandoc(md_file, out_dir, title, author, fmt_set, template_dir, page_size, font_size, enable_toc, cover_path=cover)

    click.echo("Done. Outputs:")
    for k, v in outputs.items():
        click.echo(f" - {k}: {v}")


if __name__ == "__main__":
    main()
