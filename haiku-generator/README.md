# Haiku Generator + Book Formatter Integration

Automated haiku generation using Claude, with quality filtering and professional book formatting.

## Features

- **Quality Filtering**: Generates 2500 haiku, keeps best 250
- **Persona System**: 500 unique author personas with diverse backgrounds
- **Theme Database**: 1,033 themes from global literary canon
- **Professional Formatting**: Award-winning PDF and EPUB templates

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key:**
   Create a `.env` file (copy from `.env.example`):
   ```bash
   ANTHROPIC_API_KEY=your-actual-key-here
   ```

3. **Test generation:**
   ```bash
   python simple_test.py
   ```

## Usage

### Option 1: Quality Filtered Generation (Recommended)

Generates 2500 haiku, keeps the best 250:

```bash
python quality_filter.py
```

**Process:**
1. Generates 2500 haiku
2. Filters for structure, clichés, duplicates
3. First evaluation pass (selects top 500)
4. Final evaluation (selects best 250)
5. Saves TWO editions:
   - **Curated Edition**: Top 250 haiku with quality scores
   - **Extended Edition**: All filtered haiku (~1,900-2,000)

**Output:**
- `haiku_output/haikus_curated_TIMESTAMP.txt`
- `haiku_output/haikus_extended_TIMESTAMP.txt`

**Cost:** ~$2-3 per run

### Option 2: Book with Persona & Themes

Generates complete books using random persona and themes:

```bash
python generate_with_persona_and_theme.py
```

**Process:**
1. Selects random persona from 500 options
2. Selects random themes (mix of traditional and contemporary)
3. Generates 2500 haiku in that persona's voice
4. Filters to best 250
5. Creates TWO formatted PDF and EPUB books:
   - **Curated Edition**: Top 250 haiku
   - **Extended Edition**: All filtered haiku (~1,900-2,000)

**Cost:** ~$2-3 per run
**Output:** `../book_formatter/output/`

### Option 3: Simple Generation (No Filtering)

Quick generation without quality filtering:

```bash
python generate_and_format.py
```

**Cost:** ~$0.50 per run

## Anthology Editing

The system includes professional anthology editing that implements established literary principles for poetry collections.

**Features:**
- **Sequencing**: Orders poems for optimal reading experience using tonal/quality-based strategies
- **Sectioning**: Organizes into seasonal or thematic groups with introductions
- **Contextualizing**: AI-generated introductions and section commentary
- **Analysis**: Detects seasons, themes, tones, and imagery patterns

**Automatic Integration:**
When using `generate_with_persona_and_theme.py`, anthology editions are created automatically alongside PDF/EPUB editions.

**Manual Usage:**
```bash
python anthology_editor.py <input_file> <title> [author] [organization]
```

**Example:**
```bash
python anthology_editor.py haiku_output/haikus_curated_20240101.txt "Seasonal Haiku" "Jane Doe" seasonal
```

**Organization Types:**
- `seasonal` - Groups by spring/summer/autumn/winter (default for haiku)
- `thematic` - Groups by detected themes (nature, urban, technology, etc.)

**Output Format:**
Professional anthology with:
- Title page
- Collection introduction
- Seasonal/thematic sections with introductions
- Strategically sequenced haiku
- Quality-based placement (best poems in middle)

## Files

- `config.py` - Configuration settings
- `simple_test.py` - Quick test script (generates small batch)
- `generate_and_format.py` - Integration with book formatter
- `anthology_editor.py` - Professional anthology editing system
- `theme_selector.py` - Theme selection utility
- `persona_selector.py` - Persona selection and management
- `haiku_themes.json` - Database of 1,077 themes from global literary canon
- `haiku_personas.json` - Database of 500 unique author personas
- `haiku_output/` - Generated haiku text files
- `haiku_output/anthologies/` - Professionally edited anthology versions
- Output books appear in `../book_formatter/output/`

## Theme Database

The system includes a comprehensive database of **1,033 themes** from the global haiku literary canon:

- **317 Traditional Seasonal (Kigo)** - Classical Japanese seasonal words organized by season and category
- **620 Contemporary Themes** - Modern subjects including urban life, technology, environment, social issues
- **96 Universal Themes** - Timeless human experiences and natural phenomena

### Using Themes

**View theme statistics:**
```bash
python theme_selector.py
```

**Get random themes programmatically:**
```python
from theme_selector import ThemeSelector

selector = ThemeSelector()

# Get 5 random themes
themes = selector.get_random_themes(5)

# Get spring themes
spring = selector.get_seasonal_themes('spring', 5)

# Get contemporary themes
modern = selector.get_contemporary_themes(5)

# Get mixed traditional + contemporary
mixed = selector.get_mixed_themes(traditional=3, contemporary=2)
```

**Theme Categories Include:**
- Traditional Seasonal (Spring, Summer, Autumn, Winter, New Year)
- Urban & Contemporary Life
- Technology & Digital Age
- Environmental & Ecological
- Social & Political
- Personal & Emotional
- Mental Health & Wellness
- Cross-Cultural & Identity
- And 15+ more categories...

## Persona Database

The system includes **500 unique haiku author personas** with diverse backgrounds:

**Persona Characteristics:**
- **Names:** Culturally appropriate names based on birth location and time period
- **Name changes:** 30% of personas over age 30 have changed their names (marriage, personal choice, cultural adaptation)
- **Age range:** 18-100 years (average: 58)
- **Work experience:** 0-60 years (average: 18)
- **Languages:** Average 3.5 languages per persona, 32 unique languages represented
- **Education:** 35+ leading poetry institutions including Iowa Writers' Workshop, Stanford, Oxford, Tokyo University
- **Locations:** 101+ global cities and regions across 6 continents
- **Diverse backgrounds:** 80+ occupations from teachers to engineers to farmers
- **Writing styles:** 20+ unique characteristics (urban landscapes, nature walks, bilingual writing, etc.)

### Using Personas

**View persona statistics:**
```bash
python persona_selector.py
```

**Select personas programmatically:**
```python
from persona_selector import PersonaSelector

selector = PersonaSelector()

# Get random persona
persona = selector.get_random_persona()

# Get diverse sample
diverse = selector.get_diverse_sample(5)

# Filter by age
young_poets = selector.filter_by_age(18, 30)

# Format for prompt
prompt_text = selector.format_for_prompt(persona)
```

**Example Persona:**
- **Name:** Seo Park (born Seo Smith)
- **Age:** 87, born 1939 in Busan, South Korea
- **Occupation:** Spiritual Guide (31 years experience)
- **Education:** Johns Hopkins University (MA)
- **Locations:** Childhood in Busan, worked in Boston, educated in Baltimore, now in Rural New Zealand
- **Languages:** English, Korean (native)
- **Style:** Influenced by jazz rhythms
- **Active since:** 2018

## Book Formatter Integration

The generated haiku are automatically:
- Formatted with award-winning PDF and EPUB templates
- Centered on pages
- No table of contents
- Professional title page with decorative rules
- Haikus never split across pages
