#!/usr/bin/env python3
"""
Feature 9: Sound & Sonic Quality Scoring

Evaluates haiku for phonetic qualities — assonance, consonance,
alliteration, euphony, and rhythm — using the CMU Pronouncing Dictionary.
Pure computational scoring, no API calls needed.
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

try:
    import pronouncing
except ImportError:
    pronouncing = None


# Phoneme categories
VOWEL_PHONEMES = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
    'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
}

# Soft consonants (euphonic)
SOFT_CONSONANTS = {'L', 'M', 'N', 'R', 'W', 'Y'}
# Hard consonants
HARD_CONSONANTS = {'K', 'T', 'P', 'G', 'D', 'B'}

# All consonant phonemes
CONSONANT_PHONEMES = {
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
    'W', 'Y', 'Z', 'ZH'
}


def _get_phonemes(word: str) -> Optional[List[str]]:
    """Get phoneme list for a word from CMU dict."""
    if pronouncing is None:
        return None
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return None
    # Use first pronunciation, strip stress markers
    return [re.sub(r'\d', '', p) for p in phones[0].split()]


def _get_stresses(word: str) -> Optional[str]:
    """Get stress pattern for a word (0=unstressed, 1=primary, 2=secondary)."""
    if pronouncing is None:
        return None
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return None
    return pronouncing.stresses(phones[0])


def _extract_words(haiku: str) -> List[str]:
    """Extract words from haiku text."""
    return re.findall(r"[a-zA-Z']+", haiku.lower())


def score_assonance(haiku: str) -> float:
    """Score vowel sound repetition across lines (0-10).
    Higher scores mean more pleasing vowel patterns."""
    words = _extract_words(haiku)
    if not words:
        return 5.0

    vowels_by_line = []
    lines = haiku.strip().split('\n')

    for line in lines:
        line_words = _extract_words(line)
        line_vowels = []
        for w in line_words:
            phonemes = _get_phonemes(w)
            if phonemes:
                line_vowels.extend([p for p in phonemes if p in VOWEL_PHONEMES])
        vowels_by_line.append(line_vowels)

    if not any(vowels_by_line):
        return 5.0

    # Count vowel repetitions across lines (cross-line assonance is most valued)
    all_vowels = []
    for vl in vowels_by_line:
        all_vowels.extend(vl)

    if len(all_vowels) < 3:
        return 5.0

    vowel_counts = Counter(all_vowels)
    # Score based on how many vowels repeat and how often
    repeated = sum(c for c in vowel_counts.values() if c > 1)
    total = sum(vowel_counts.values())
    repetition_ratio = repeated / total if total > 0 else 0

    # Cross-line bonus: check if the same vowel appears in multiple lines
    cross_line_count = 0
    vowel_sets = [set(vl) for vl in vowels_by_line if vl]
    if len(vowel_sets) >= 2:
        for v in VOWEL_PHONEMES:
            lines_with_v = sum(1 for vs in vowel_sets if v in vs)
            if lines_with_v >= 2:
                cross_line_count += 1

    # Scale: 0.3 repetition + some cross-line = ~7-8 score
    score = 3.0 + (repetition_ratio * 4.0) + min(cross_line_count * 0.6, 3.0)
    return min(max(score, 0.0), 10.0)


def score_consonance(haiku: str) -> float:
    """Score consonant sound patterns (0-10)."""
    words = _extract_words(haiku)
    if not words:
        return 5.0

    all_consonants = []
    for w in words:
        phonemes = _get_phonemes(w)
        if phonemes:
            all_consonants.extend([p for p in phonemes if p in CONSONANT_PHONEMES])

    if len(all_consonants) < 3:
        return 5.0

    consonant_counts = Counter(all_consonants)
    repeated = sum(c for c in consonant_counts.values() if c > 1)
    total = sum(consonant_counts.values())
    repetition_ratio = repeated / total if total > 0 else 0

    score = 3.0 + (repetition_ratio * 5.0)
    return min(max(score, 0.0), 10.0)


def score_alliteration(haiku: str) -> float:
    """Score initial-sound repetition (0-10)."""
    lines = haiku.strip().split('\n')
    alliterations = 0
    total_pairs = 0

    for line in lines:
        words = _extract_words(line)
        initials = []
        for w in words:
            phonemes = _get_phonemes(w)
            if phonemes:
                initials.append(phonemes[0])

        # Check consecutive pairs
        for i in range(len(initials) - 1):
            total_pairs += 1
            if initials[i] == initials[i + 1]:
                alliterations += 1

    if total_pairs == 0:
        return 5.0

    ratio = alliterations / total_pairs
    # Subtle alliteration is better than overwhelming
    # Ideal: 1-2 alliterative pairs per haiku
    if alliterations == 0:
        score = 4.0
    elif alliterations <= 2:
        score = 6.0 + ratio * 6.0
    else:
        # Too much alliteration can feel forced
        score = 7.0 - (alliterations - 2) * 0.5

    return min(max(score, 0.0), 10.0)


def score_euphony(haiku: str) -> float:
    """Score the ratio of soft to hard consonant sounds (0-10).
    Higher = more euphonic (pleasing to the ear)."""
    words = _extract_words(haiku)
    if not words:
        return 5.0

    soft_count = 0
    hard_count = 0

    for w in words:
        phonemes = _get_phonemes(w)
        if phonemes:
            for p in phonemes:
                base = re.sub(r'\d', '', p)
                if base in SOFT_CONSONANTS:
                    soft_count += 1
                elif base in HARD_CONSONANTS:
                    hard_count += 1

    total = soft_count + hard_count
    if total == 0:
        return 5.0

    soft_ratio = soft_count / total
    # A good mix leans soft but has some hard consonants for texture
    # Ideal ratio: 55-70% soft
    if 0.5 <= soft_ratio <= 0.75:
        score = 7.0 + (1.0 - abs(soft_ratio - 0.625) / 0.125) * 2.0
    elif soft_ratio > 0.75:
        score = 6.5  # Too soft, lacks texture
    else:
        score = 4.0 + soft_ratio * 4.0

    return min(max(score, 0.0), 10.0)


def score_rhythm(haiku: str) -> float:
    """Score stress pattern variety and flow (0-10)."""
    lines = haiku.strip().split('\n')
    all_stresses = []

    for line in lines:
        words = _extract_words(line)
        line_stress = ""
        for w in words:
            s = _get_stresses(w)
            if s:
                line_stress += s
        if line_stress:
            all_stresses.append(line_stress)

    if not all_stresses:
        return 5.0

    # Score based on stress variation (monotone = bad, varied = good)
    total_score = 0.0
    for stress in all_stresses:
        if len(stress) < 2:
            total_score += 5.0
            continue

        # Count transitions between stressed and unstressed
        transitions = 0
        for i in range(len(stress) - 1):
            if stress[i] != stress[i + 1]:
                transitions += 1

        transition_ratio = transitions / (len(stress) - 1) if len(stress) > 1 else 0
        # Good rhythm has ~50-70% transitions (not monotone, not chaotic)
        if 0.4 <= transition_ratio <= 0.8:
            total_score += 7.0 + (1.0 - abs(transition_ratio - 0.6) / 0.2) * 2.0
        elif transition_ratio > 0.8:
            total_score += 6.0  # Too choppy
        else:
            total_score += 4.0 + transition_ratio * 5.0

    return min(max(total_score / len(all_stresses), 0.0), 10.0)


def score_sonic_quality(haiku: str) -> float:
    """Compute composite sonic quality score for a single haiku (0-10).

    Weights:
    - Assonance: 25%
    - Consonance: 20%
    - Alliteration: 15%
    - Euphony: 20%
    - Rhythm: 20%
    """
    if pronouncing is None:
        return 5.0  # Neutral score if library unavailable

    assonance = score_assonance(haiku)
    consonance = score_consonance(haiku)
    alliteration = score_alliteration(haiku)
    euphony = score_euphony(haiku)
    rhythm = score_rhythm(haiku)

    composite = (
        assonance * 0.25 +
        consonance * 0.20 +
        alliteration * 0.15 +
        euphony * 0.20 +
        rhythm * 0.20
    )
    return round(composite, 2)


def score_sonic_batch(haiku_list: List[str]) -> List[float]:
    """Score a batch of haiku for sonic quality. Pure computation, no API cost.

    Args:
        haiku_list: List of haiku texts

    Returns:
        List of sonic scores (0-10), one per haiku
    """
    return [score_sonic_quality(h) for h in haiku_list]


def score_sonic_batch_detailed(haiku_list: List[str]) -> List[Dict]:
    """Score a batch with detailed breakdowns.

    Returns:
        List of dicts with all sub-scores and composite
    """
    results = []
    for h in haiku_list:
        results.append({
            'text': h,
            'assonance': score_assonance(h),
            'consonance': score_consonance(h),
            'alliteration': score_alliteration(h),
            'euphony': score_euphony(h),
            'rhythm': score_rhythm(h),
            'sonic_score': score_sonic_quality(h)
        })
    return results


if __name__ == "__main__":
    test_haiku = [
        "spring rain falling—\nthe earthworm crosses\na wet stone path",
        "traffic light changing—\nthe cyclist's breath\nfogs the morning air",
        "construction crane lifting—\na butterfly crosses\nits shadow",
        "subway doors closing—\na stranger's perfume\nlingers in the air",
    ]

    print("Sonic Quality Scoring Test")
    print("=" * 50)

    for h in test_haiku:
        details = score_sonic_batch_detailed([h])[0]
        print(f"\n{h}")
        print(f"  Assonance:    {details['assonance']:.1f}")
        print(f"  Consonance:   {details['consonance']:.1f}")
        print(f"  Alliteration: {details['alliteration']:.1f}")
        print(f"  Euphony:      {details['euphony']:.1f}")
        print(f"  Rhythm:       {details['rhythm']:.1f}")
        print(f"  COMPOSITE:    {details['sonic_score']:.1f}")
