"""
Configuration for Claude Haiku Generator
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings"""

    # API Configuration
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    MODEL = "claude-haiku-4-20250514"

    # Generation Settings
    HAIKU_TO_GENERATE = 2000
    HAIKU_TO_PUBLISH = 250
    GENERATION_INTERVAL = 300  # seconds (5 minutes)

    # Quality Thresholds
    MIN_SCORE_FIRST_PASS = 7.0
    MIN_SCORE_FINAL = 7.8
    TARGET_AVG_SCORE = 8.2

    # Storage Paths
    DATABASE_PATH = Path("haiku_collection.db")
    OUTPUT_DIR = Path("haiku_output")
    BACKUP_DIR = Path("backups")
    LOG_DIR = Path("logs")

    # Create directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    BACKUP_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    # Prompt Caching
    USE_PROMPT_CACHING = True

    # System Prompts (cached)
    SYSTEM_PROMPT = """You are a haiku generation system specializing in high-quality, publication-worthy haiku.

Requirements:
- Traditional 5-7-5 syllable structure or justified variations (3-5-3, 4-6-4)
- Seasonal awareness (kigo) where appropriate
- Cutting word (kireji) creating juxtaposition
- Direct observation (shasei)
- Present tense immediacy
- Concrete imagery, no abstractions
- Mono no aware (awareness of impermanence)

Styles to include:
- Traditional (nature-focused, seasonal, 5-7-5)
- Contemporary American (urban, modern, flexible form)
- Experimental (boundary-pushing while respecting essence)

Quality bar: Every haiku should be publication-worthy."""

    EXAMPLES = """
Examples of excellent haiku:

spring rain falling—
the earthworm crosses
a wet stone path

traffic light changing—
the cyclist's breath
fogs the morning air

construction crane lifting—
a butterfly crosses
its shadow

subway doors closing—
a stranger's perfume
lingers in the air

harvest moon rising—
the hospital window
frames nothing else

algorithm waking—
to process another spring
without sensing it

first snow melting—
on the parking meter
yesterday's receipt

city park at dusk—
one pigeon watching
the joggers pass
"""

    EVALUATION_CRITERIA = """Score each haiku 0-10 on these dimensions:

Traditional Qualities:
- Mono no aware (poignancy, impermanence) 0-10
- Kigo (seasonal resonance) 0-10
- Kire (cutting, juxtaposition) 0-10
- Shasei (direct observation) 0-10
- Karumi (lightness, simplicity) 0-10

Technical Quality:
- Syllable appropriateness 0-10
- Concrete imagery (no abstractions) 0-10
- Single moment captured 0-10
- Two-part structure 0-10
- Present tense immediacy 0-10

Contemporary Merit:
- Freshness (avoiding clichés) 0-10
- Cultural relevance 0-10
- Linguistic precision 0-10
- Sensory vividness 0-10
- Interpretive openness 0-10

Calculate composite score as average of all dimensions.
Return JSON array with scores."""

    DETAILED_EVALUATION_PROMPT = """Evaluate each haiku with comprehensive scoring.

Provide detailed scores 0-10 for:

1. Traditional Qualities:
   - Mono no aware
   - Seasonal resonance
   - Cutting/juxtaposition
   - Direct observation
   - Lightness/simplicity

2. Technical Excellence:
   - Structure appropriateness
   - Imagery concreteness
   - Moment singularity
   - Two-part balance
   - Tense consistency

3. Contemporary Merit:
   - Originality/freshness
   - Cultural relevance
   - Word precision
   - Sensory impact
   - Reader interpretation space

Return JSON with composite_score and dimension breakdowns."""

    METADATA_PROMPT = """Generate comprehensive metadata for each haiku:

For each haiku provide:

1. Seasonal Context:
   - Identify kigo (seasonal word) if present
   - Explain seasonal significance
   - Note seasonal appropriateness

2. Craft Analysis:
   - Identify kire (cutting point)
   - Explain juxtaposition
   - Note technical strengths

3. Cultural Notes:
   - Relevant traditions
   - Contemporary context
   - Cross-cultural elements

4. Tags:
   - Season: Spring/Summer/Autumn/Winter/Universal
   - Theme: Nature/Urban/Technology/Human/Abstract
   - Style: Traditional/Contemporary/Experimental

Return JSON array with all metadata."""

    # Cliché Database
    CLICHES = [
        "lonely crow", "old pond", "frog jumps", "frog leaps",
        "cherry blossom falls", "cherry blossoms fall",
        "autumn leaf falls", "autumn leaves fall",
        "moonlight shines", "moonlight gleams",
        "winter wind blows", "winter winds blow",
        "snow is falling", "snow falls gently",
        "petals floating", "petals drift",
        "distant mountain", "mountains in mist",
        "temple bell rings", "temple bells ring"
    ]

    # Enhanced Pipeline Feature Toggles
    ENABLE_PLAGIARISM_CHECK = True       # Feature 5: Canonical echo detection
    ENABLE_VOICE_AUDIT = True            # Feature 4: Persona voice consistency
    ENABLE_SONIC_SCORING = True          # Feature 9: Phonetic quality scoring
    ENABLE_TURN_EVALUATION = True        # Feature 3: Kireji/juxtaposition evaluation
    ENABLE_REVISION_PASS = True          # Feature 1: Revise near-miss haiku
    ENABLE_LLM_SEQUENCING = True         # Feature 2: LLM-based poem ordering
    ENABLE_SCHOLARLY_NOTES = True        # Feature 8: Scholarly notes on standouts
    ENABLE_COVER_PROMPT = True           # Feature 7: Cover image prompt generation
    ENABLE_COVER_IMAGE = True            # Feature 7b: Generate actual cover via DALL-E 3
    ENABLE_ARC_EVALUATION = True         # Feature 10: Collection arc evaluation

    # Feature-Specific Settings
    REVISION_SCORE_MIN = 6.0             # Min score for revision candidates
    REVISION_SCORE_MAX = 7.5             # Max score for revision candidates
    REVISION_MAX_COUNT = 100             # Max haiku to attempt revising
    VOICE_AUDIT_THRESHOLD = 5.0          # Min voice consistency score to keep
    PLAGIARISM_THRESHOLD = 0.70          # Semantic similarity threshold for flagging
    SONIC_WEIGHT = 0.10                  # Sonic score weight in composite (10%)
    TURN_WEIGHT = 0.25                   # Turn score weight in composite (25%)
    SCHOLARLY_NOTES_COUNT = 12           # Number of poems to annotate
    ARC_EVAL_MIN_SCORE = 7.0             # Below this triggers rebuild
