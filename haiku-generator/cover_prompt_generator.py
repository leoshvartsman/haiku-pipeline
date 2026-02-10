#!/usr/bin/env python3
"""
Feature 7: Image Generation Prompts and Cover Images

Synthesizes the collection's themes, mood, and imagery into detailed
prompts for DALL-E and Midjourney to create unique, thematically
appropriate book covers. Optionally generates the cover image via DALL-E 3.
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0)
MODEL = "claude-sonnet-4-20250514"


def _summarize_collection(haiku_analyses: List[Dict]) -> Dict:
    """Extract summary statistics from haiku analyses for the prompt."""
    seasons = Counter()
    themes = Counter()
    tones = Counter()
    all_imagery = []

    for h in haiku_analyses:
        seasons[h.get('season', 'timeless')] += 1
        themes[h.get('theme', 'general')] += 1
        tones[h.get('tone', 'contemplative')] += 1
        imagery = h.get('imagery', [])
        if isinstance(imagery, list):
            all_imagery.extend(imagery)
        elif isinstance(imagery, str):
            all_imagery.append(imagery)

    # Get top imagery items
    imagery_counts = Counter(all_imagery)
    top_imagery = [item for item, _ in imagery_counts.most_common(15)]

    return {
        'seasons': dict(seasons.most_common()),
        'themes': dict(themes.most_common(8)),
        'tones': dict(tones.most_common(5)),
        'top_imagery': top_imagery
    }


def generate_cover_prompt(
    haiku_analyses: List[Dict],
    title: str,
    author: str,
    persona: Dict
) -> Tuple[Dict, float]:
    """Generate cover image prompt based on collection analysis.

    Args:
        haiku_analyses: List of analyzed haiku dicts
        title: Book title
        author: Author name
        persona: Author persona dict

    Returns:
        Tuple of (cover_data_dict, cost)
        cover_data includes dalle_prompt, midjourney_prompt, style_notes, etc.
    """
    summary = _summarize_collection(haiku_analyses)

    # Select a few standout haiku as inspiration
    sorted_haiku = sorted(haiku_analyses, key=lambda x: x.get('quality', 7.0), reverse=True)
    sample_haiku = "\n\n".join([h.get('haiku', h.get('text', '')) for h in sorted_haiku[:8]])

    characteristic = persona.get('characteristic', '')
    locations = persona.get('locations', {})
    current_location = locations.get('current', '')

    prompt = f"""You are a minimalist art director creating a clean, modern book cover
for a haiku collection. Your aesthetic is rooted in contemporary design: generous
white/negative space, restrained color palettes, precise geometric or organic forms,
and a sense of calm sophistication.

Title: "{title}"
Author: {author}
Poet's style: {characteristic}
Poet's location: {current_location}

COLLECTION ANALYSIS:
- Season distribution: {json.dumps(summary['seasons'])}
- Primary themes: {json.dumps(summary['themes'])}
- Dominant tones: {json.dumps(summary['tones'])}
- Most frequent imagery: {', '.join(summary['top_imagery'][:10])}

Sample standout haiku from the collection:

{sample_haiku}

STYLE DIRECTION — MANDATORY:
The image must be CLEAN, MINIMALIST, and MODERN. Think of the aesthetic of:
- Scandinavian design posters
- Japanese wabi-sabi minimalism
- Contemporary fine art photography with muted tones
- Modern editorial illustration with bold negative space

Specific requirements:
1. SIMPLICITY: One or two focal elements maximum. No clutter, no busy backgrounds.
   Let negative space do most of the visual work.
2. MUTED PALETTE: Limit to 2-3 colors plus neutrals. Avoid saturated or garish tones.
   Prefer dusty, muted, or tonal ranges (soft sage, warm grey, ink black, parchment).
3. COMPOSITION: The image must have a clear area of flat or minimal texture — either
   the upper 20% or lower 20% — where text can sit naturally without overlays. Design
   the composition so the title area feels intentional, not forced.
4. NO TEXT in the image itself. The text will be composited separately.
5. THEMATIC: Capture the essence of THIS collection through a single distilled visual
   metaphor, not a literal scene. Abstract or semi-abstract interpretations encouraged.
6. MODERN PRINT QUALITY: Should look at home on the shelf of a contemporary bookshop
   alongside publishers like Graywolf Press, Copper Canyon Press, or NYRB Classics.

Return ONLY a JSON object:
{{
  "dalle_prompt": "detailed prompt for DALL-E 3 (200-300 words). Must explicitly request: minimalist composition, clean negative space, muted color palette, contemporary fine art aesthetic, no text. Describe the single focal image and how it relates to the collection. Specify the area left clear for text placement (top or bottom band).",
  "midjourney_prompt": "detailed Midjourney prompt with parameters --ar 9:16 --style raw --v 6 --s 250. Emphasize minimalism and negative space.",
  "style_notes": "description of the desired artistic style — must reference minimalism, negative space, and modern design",
  "color_palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5"],
  "text_placement": "top or bottom — which area of the image has been left clear for text",
  "composition_notes": "detailed description of where the focal element sits and where text should go, including whether text should be light-on-dark or dark-on-light based on the background in the text area",
  "negative_prompt": "busy backgrounds, clutter, multiple focal points, saturated colors, text, letters, words, cliche imagery, stock photo aesthetic, ornate borders, gradients"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}]
    )

    cover_data = {}
    try:
        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            cover_data = json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError):
        cover_data = {
            'dalle_prompt': response.content[0].text,
            'midjourney_prompt': '',
            'style_notes': '',
            'color_palette': [],
            'composition_notes': '',
            'negative_prompt': ''
        }

    # Add metadata
    cover_data['title'] = title
    cover_data['author'] = author
    cover_data['collection_summary'] = summary

    usage = response.usage
    cost = (usage.input_tokens * 3.00 / 1_000_000) + (usage.output_tokens * 15.00 / 1_000_000)

    return cover_data, cost


def save_cover_prompt(
    cover_data: Dict,
    output_dir: Path,
    title: str
) -> Path:
    """Save cover prompt data as a JSON file.

    Args:
        cover_data: Cover prompt data dict
        output_dir: Output directory path
        title: Book title (used in filename)

    Returns:
        Path to the saved JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create safe filename
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:50]
    filename = output_dir / f"cover_prompt_{safe_title}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(cover_data, f, indent=2, ensure_ascii=False)

    return filename


def generate_cover_image(
    cover_data: Dict,
    output_dir: Path,
    title: str
) -> Tuple[Optional[Path], float]:
    """Generate a cover image using DALL-E 3.

    Uses the dalle_prompt from cover_data to generate a portrait-ratio
    book cover image. Requires OPENAI_API_KEY in environment.

    Args:
        cover_data: Cover prompt data dict (must contain 'dalle_prompt')
        output_dir: Directory to save the image
        title: Book title (used in filename)

    Returns:
        Tuple of (image_path or None, cost)
        Cost is ~$0.080 for dall-e-3 at 1024x1792 (HD quality)
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  Warning: OPENAI_API_KEY not set. Skipping cover image generation.")
        return None, 0.0

    dalle_prompt = cover_data.get('dalle_prompt', '')
    if not dalle_prompt:
        print("  Warning: No DALL-E prompt available. Skipping image generation.")
        return None, 0.0

    try:
        from openai import OpenAI
    except ImportError:
        print("  Warning: openai package not installed. Run: pip install openai")
        return None, 0.0

    openai_client = OpenAI(api_key=api_key, timeout=120.0)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:50]

    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1792",
            quality="hd",
            n=1,
        )

        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt

        # Download the image
        import urllib.request
        image_path = output_dir / f"cover_{safe_title}.png"
        urllib.request.urlretrieve(image_url, str(image_path))

        # Save the revised prompt for reference
        if revised_prompt:
            meta_path = output_dir / f"cover_{safe_title}_revised_prompt.txt"
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(revised_prompt)

        # DALL-E 3 HD at 1024x1792 costs $0.120
        cost = 0.120

        return image_path, cost

    except Exception as e:
        print(f"  Warning: DALL-E image generation failed: {e}")
        return None, 0.0


def _sample_region_color(img, x0, y0, x1, y1):
    """Sample the average color of a rectangular region of the image."""
    region = img.crop((x0, y0, x1, y1))
    # Resize to 1x1 to get average color
    avg = region.resize((1, 1)).getpixel((0, 0))
    return avg[:3]  # RGB only


def _perceived_brightness(r, g, b):
    """Calculate perceived brightness (0-255) using luminance formula."""
    return (r * 299 + g * 587 + b * 114) / 1000


def composite_cover_text(
    image_path: Path,
    title: str,
    author: str,
    cover_data: Dict
) -> Path:
    """Overlay title and author text onto the cover image.

    Uses a modern, minimalist approach: text is placed directly on the
    image's natural negative space (top or bottom area as specified by
    the LLM prompt). Text color adapts to the background — dark text
    on light areas, light text on dark areas. A thin accent line
    separates title from author for a clean editorial look.

    Args:
        image_path: Path to the raw DALL-E cover image
        title: Book title
        author: Author name
        cover_data: Cover prompt data (uses color_palette, text_placement,
                    composition_notes)

    Returns:
        Path to the composited cover image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
    except ImportError:
        print("  Warning: Pillow not installed. Run: pip install Pillow")
        return image_path

    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    # Save raw backup (textless version) if one doesn't exist
    raw_path = Path(str(image_path).replace('.png', '_raw.png'))
    if not raw_path.exists():
        img.save(str(raw_path))

    # Create overlay
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # --- Find fonts ---
    def load_font(name_hints, size):
        """Try to load a font from system paths."""
        search_dirs = [
            Path("/System/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
            Path("/Library/Fonts"),
        ]
        for font_dir in search_dirs:
            if not font_dir.exists():
                continue
            for f in font_dir.iterdir():
                fname = f.name.lower()
                if any(hint in fname for hint in name_hints):
                    try:
                        return ImageFont.truetype(str(f), size)
                    except (OSError, IOError):
                        continue
        return ImageFont.load_default()

    # Title font: bold and prominent
    title_font_size = width // 8
    title_font = load_font(["avenir next.ttc", "avenir.ttc", "helvetica neue", "helvetica"], title_font_size)

    # Author font: readable but secondary
    author_font_size = width // 16
    author_font = load_font(["avenir next.ttc", "avenir.ttc", "helvetica neue", "helvetica"], author_font_size)

    # --- Determine text color based on top-right background ---
    sample = _sample_region_color(
        img, int(width * 0.5), 0, width, int(height * 0.20)
    )
    bg_brightness = _perceived_brightness(*sample)

    if bg_brightness > 140:
        text_color = (30, 30, 30, 255)
        accent_color = (80, 80, 80, 200)
    else:
        text_color = (245, 245, 245, 255)
        accent_color = (200, 200, 200, 180)

    # Override with palette accent if good contrast
    palette = cover_data.get('color_palette', [])
    if palette:
        def hex_to_rgb(hex_color):
            h = hex_color.lstrip('#')
            if len(h) != 6:
                return None
            return (int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16))

        for color_hex in palette:
            rgb = hex_to_rgb(color_hex)
            if rgb is None:
                continue
            color_brightness = _perceived_brightness(*rgb)
            if abs(color_brightness - bg_brightness) > 80:
                accent_color = (*rgb, 220)
                break

    # --- Layout text (top-right aligned) ---
    title_upper = title.upper()
    right_margin = int(width * 0.08)
    top_margin = int(height * 0.06)

    # Word-wrap title
    max_text_width = int(width * 0.75)
    title_lines = []
    words = title_upper.split()
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=title_font)
        if bbox[2] - bbox[0] > max_text_width and current_line:
            title_lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        title_lines.append(current_line)

    # Dimensions
    line_spacing = int(title_font_size * 0.2)
    accent_line_width = int(width * 0.15)
    accent_line_thickness = 3
    spacing_after_title = int(title_font_size * 0.5)
    spacing_after_accent = int(author_font_size * 0.5)

    author_bbox = draw.textbbox((0, 0), author, font=author_font)
    author_w = author_bbox[2] - author_bbox[0]

    # --- Draw text (right-aligned from top) ---
    y = top_margin

    # Title lines — right-aligned
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        text_w = bbox[2] - bbox[0]
        x = width - right_margin - text_w
        draw.text((x, y), line, font=title_font, fill=text_color)
        y += title_font_size + line_spacing

    # Accent line — right-aligned
    y += spacing_after_title - line_spacing
    accent_x_right = width - right_margin
    accent_x_left = accent_x_right - accent_line_width
    draw.line(
        [(accent_x_left, y), (accent_x_right, y)],
        fill=accent_color, width=accent_line_thickness
    )

    # Author — right-aligned
    y += accent_line_thickness + spacing_after_accent
    author_x = width - right_margin - author_w
    draw.text((author_x, y), author, font=author_font, fill=text_color)

    # Composite
    result = Image.alpha_composite(img, overlay)
    result = result.convert("RGB")

    result.save(str(image_path), quality=95)
    print(f"  ✓ Composited title and author onto cover image")
    return image_path


if __name__ == "__main__":
    test_analyses = [
        {'haiku': "spring rain falling—\nthe earthworm crosses\na wet stone path", 'season': 'spring', 'theme': 'nature', 'tone': 'quiet', 'quality': 9.0, 'imagery': ['rain', 'earthworm', 'stone']},
        {'haiku': "traffic light changing—\nthe cyclist's breath\nfogs the morning air", 'season': 'timeless', 'theme': 'urban', 'tone': 'contemplative', 'quality': 8.5, 'imagery': ['traffic light', 'breath', 'fog']},
    ]

    test_persona = {
        'name': 'Kenji Watanabe',
        'characteristic': 'finds poetry in the intersection of nature and industry',
        'locations': {'current': 'Osaka, Japan'}
    }

    print("Cover Prompt Generator Test")
    print("=" * 50)

    cover_data, cost = generate_cover_prompt(test_analyses, "Steel and Blossoms", "Kenji Watanabe", test_persona)

    print(f"\nDALL-E prompt:\n{cover_data.get('dalle_prompt', 'N/A')[:200]}...")
    print(f"\nMidjourney prompt:\n{cover_data.get('midjourney_prompt', 'N/A')[:200]}...")
    print(f"\nStyle: {cover_data.get('style_notes', 'N/A')}")
    print(f"Colors: {cover_data.get('color_palette', [])}")
    print(f"\nCost: ${cost:.4f}")
