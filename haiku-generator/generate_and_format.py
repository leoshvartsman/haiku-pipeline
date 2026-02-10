#!/usr/bin/env python3
"""
Generate Haiku and Format as Book
Integrates haiku generator with book formatter
"""

import argparse
import anthropic
import os
import sys
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Initialize client
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def generate_haiku_batch(count=250, theme="", persona=None, themes_list=None):
    """Generate a batch of haiku

    Args:
        count: Number of haiku to generate
        theme: Single theme string (legacy parameter)
        persona: Persona dictionary from persona_selector
        themes_list: List of themes from theme_selector
    """

    print(f"Generating {count} haiku...")

    # Build prompt based on what's provided
    if persona and themes_list:
        # Use persona and themes
        from persona_selector import PersonaSelector
        selector = PersonaSelector()
        persona_prompt = selector.format_for_prompt(persona)
        themes_str = ", ".join(themes_list)

        print(f"Persona: {persona['name']}")
        print(f"Themes: {themes_str}\n")

        prompt = f"""{persona_prompt}

Generate {count} excellent haiku exploring these themes: {themes_str}

Requirements:
- Traditional 5-7-5 or justified variations
- Concrete imagery (no abstractions)
- Present tense immediacy
- Draw from the persona's unique perspective and experiences
- Mix of traditional and contemporary styles

Format: One haiku per entry, separated by blank lines.
Output only the haiku, no numbering."""
    else:
        # Legacy mode
        print(f"Theme: {theme if theme else 'General'}\n")

        prompt = f"""Generate {count} excellent haiku.

{'Theme: ' + theme if theme else 'Mix of themes: nature, urban life, seasons, technology, human moments.'}

Requirements:
- Traditional 5-7-5 or justified variations
- Concrete imagery (no abstractions)
- Present tense immediacy
- Seasonal awareness where appropriate
- Mix of traditional and contemporary styles

Format: One haiku per entry, separated by blank lines.
Output only the haiku, no numbering."""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        temperature=0.9,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    # Parse haiku
    text = response.content[0].text
    haiku_list = []

    # Split by double newlines
    import re
    blocks = text.strip().split('\n\n')

    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) == 3:
            haiku_list.append('\n'.join(lines))

    print(f"✓ Generated {len(haiku_list)} haiku\n")
    return haiku_list

def generate_haiku(persona=None, themes=None, num_haiku=50):
    """Wrapper function for generating haiku with persona and themes

    Args:
        persona: Persona dictionary from persona_selector
        themes: List of themes from theme_selector
        num_haiku: Number of haiku to generate
    """
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        exit(1)

    # Generate haiku
    haiku_list = generate_haiku_batch(
        count=num_haiku,
        persona=persona,
        themes_list=themes
    )

    if not haiku_list:
        print("Error: No haiku generated")
        return

    # Create title and author from persona
    if persona:
        author = persona['name']
        title = f"Haikus by {author}"
    else:
        author = "Generated Collection"
        title = "Haikus"

    # Save and format
    input_file = save_for_book_formatter(haiku_list, title, author)
    success = format_as_book(input_file, title, author)

    if success:
        print()
        print("="*70)
        print("COMPLETE!")
        print(f"✓ Generated {len(haiku_list)} haiku")
        print(f"✓ Created PDF and EPUB")
        print(f"✓ Title: {title}")
        print(f"✓ Author: {author}")
        print("="*70)

def save_for_book_formatter(haiku_list, title, author, notes_text="", suffix=""):
    """Save haiku in format ready for book formatter"""

    output_dir = Path("haiku_output")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"haikus_{timestamp}{suffix}.txt"

    with open(filename, 'w') as f:
        f.write(f"Author: {author}\n")
        f.write(f"Title: {title}\n\n")

        for i, haiku in enumerate(haiku_list, 1):
            f.write(f"{i}.\n{haiku}\n\n")

        if notes_text:
            f.write(notes_text)

    print(f"✓ Saved {len(haiku_list)} haiku to: {filename}\n")
    return filename

def format_as_book(input_file, title, author, cover_image=None, formats="epub"):
    """Use book formatter to create PDF and/or EPUB"""

    print("Formatting as book...")

    # Path to book formatter
    formatter_dir = Path(__file__).parent.parent / "book_formatter"
    formatter_script = formatter_dir / "book_formatter.py"

    if not formatter_script.exists():
        print(f"Error: Book formatter not found at {formatter_script}")
        return False

    # Run book formatter
    cmd = [
        "python3",
        str(formatter_script),
        str(input_file),
        "--title", title,
        "--author", author,
        "--out-dir", str(formatter_dir / "output"),
        "--formats", formats,
    ]

    if cover_image:
        cmd.extend(["--cover", str(cover_image)])

    # Add PATH for LaTeX (macOS BasicTeX path + Linux texlive)
    env = os.environ.copy()
    env['PATH'] = f"/Library/TeX/texbin:/usr/local/bin:{env['PATH']}"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Book formatted successfully!")
        print(f"\nOutput location: {formatter_dir / 'output'}/")
        return True
    else:
        print("Error formatting book:")
        print(result.stderr)
        return False

def generate_cover(haiku_list, title, author, output_dir):
    """Generate a DALL-E cover image and composite title/author text.

    Returns the path to the cover image, or None on failure.
    """
    # Build minimal haiku analyses for the cover prompt generator
    analyses = [
        {"haiku": h, "season": "timeless", "theme": "general",
         "tone": "contemplative", "quality": 8.0, "imagery": []}
        for h in haiku_list
    ]
    persona = {
        "name": author,
        "characteristic": "finds poetry in everyday moments",
        "locations": {"current": ""},
    }

    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from cover_prompt_generator import (
        generate_cover_prompt, save_cover_prompt,
        generate_cover_image, composite_cover_text,
    )

    print("Generating cover prompt...")
    cover_data, prompt_cost = generate_cover_prompt(analyses, title, author, persona)
    save_cover_prompt(cover_data, output_dir, title)
    print(f"  Cover prompt cost: ${prompt_cost:.4f}")

    print("Generating cover image with DALL-E 3...")
    image_path, image_cost = generate_cover_image(cover_data, output_dir, title)
    if image_path:
        print(f"  Image cost: ${image_cost:.3f}")
        composite_cover_text(image_path, title, author, cover_data)
        return image_path
    else:
        print("  Cover image generation skipped (no OPENAI_API_KEY?)")
        return None


def main():
    """Main execution — supports both CLI args and interactive mode."""

    parser = argparse.ArgumentParser(description="Generate haiku and format as book")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 5 haiku, auto title/author, non-interactive")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of haiku to generate")
    parser.add_argument("--title", default=None, help="Book title")
    parser.add_argument("--author", default=None, help="Author name")
    parser.add_argument("--theme", default="", help="Theme for haiku generation")
    parser.add_argument("--cover", action="store_true",
                        help="Generate a DALL-E cover image (requires OPENAI_API_KEY)")
    parser.add_argument("--formats", default="epub",
                        help="Output formats: pdf,epub (default: epub)")
    args = parser.parse_args()

    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        print("Create a .env file with your API key")
        exit(1)

    # Test mode: small defaults, non-interactive
    if args.test:
        count = args.count or 5
        title = args.title or "Test Haiku Collection"
        author = args.author or "Test Author"
        theme = args.theme or "nature, seasons"
        make_cover = args.cover
    elif args.count is not None or args.title is not None:
        # CLI mode (any explicit arg skips interactive)
        count = args.count or 250
        title = args.title or "Haikus"
        author = args.author or "Generated Collection"
        theme = args.theme
        make_cover = args.cover
    else:
        # Interactive mode (original behavior)
        print("="*70)
        print("HAIKU GENERATOR + BOOK FORMATTER")
        print("="*70)
        print()

        count_input = input("How many haiku to generate? (default 250): ").strip()
        count = int(count_input) if count_input.isdigit() else 250
        theme = input("Theme (press Enter for mixed themes): ").strip()
        title_input = input("Book title (default: Haikus): ").strip()
        title = title_input if title_input else "Haikus"
        author_input = input("Author name (default: Generated Collection): ").strip()
        author = author_input if author_input else "Generated Collection"
        make_cover = False

        print()
        print("="*70)
        print()

    print(f"Generating {count} haiku | Title: \"{title}\" | Author: {author}")
    if make_cover:
        print("Cover generation: enabled")
    print()

    # Generate haiku
    haiku_list = generate_haiku_batch(count, theme)

    if not haiku_list:
        print("Error: No haiku generated")
        return

    # Save for formatter
    input_file = save_for_book_formatter(haiku_list, title, author)

    # Generate cover if requested
    cover_image = None
    output_dir = Path("haiku_output")
    if make_cover:
        cover_image = generate_cover(haiku_list, title, author, output_dir)

    # Format as book
    formats = args.formats if hasattr(args, 'formats') else "epub"
    success = format_as_book(input_file, title, author, cover_image=cover_image, formats=formats)

    if success:
        print()
        print("="*70)
        print("COMPLETE!")
        print(f"  {len(haiku_list)} haiku generated")
        print(f"  Title: {title}")
        print(f"  Author: {author}")
        if cover_image:
            print(f"  Cover: {cover_image}")
        print("="*70)


if __name__ == "__main__":
    main()
