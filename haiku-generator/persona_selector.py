#!/usr/bin/env python3
"""
Persona Selector for Haiku Generation
Reads from haiku_personas.json and provides persona selection utilities
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


# Career stage definitions with percentage ranges and characteristics
CAREER_STAGES = {
    'emerging': {
        'range': (0, 20),
        'description': 'Finding their voice, experimenting with form',
        'voice_traits': ['searching', 'experimental', 'raw', 'earnest', 'questioning'],
        'theme_tendencies': ['identity', 'discovery', 'wonder', 'uncertainty', 'first experiences'],
    },
    'developing': {
        'range': (20, 40),
        'description': 'Building technique, gaining confidence',
        'voice_traits': ['growing confidence', 'developing style', 'more precise', 'exploring depth'],
        'theme_tendencies': ['relationships', 'place', 'craft', 'observation', 'early wisdom'],
    },
    'established': {
        'range': (40, 60),
        'description': 'Mature voice, recognized style',
        'voice_traits': ['confident', 'distinctive', 'refined', 'assured', 'masterful technique'],
        'theme_tendencies': ['complexity', 'nuance', 'interconnection', 'subtle observation'],
    },
    'mature': {
        'range': (60, 80),
        'description': 'Deep wisdom, distilled expression',
        'voice_traits': ['distilled', 'wise', 'economical', 'profound simplicity', 'quiet authority'],
        'theme_tendencies': ['impermanence', 'memory', 'acceptance', 'legacy', 'cycles'],
    },
    'master': {
        'range': (80, 100),
        'description': 'Transcendent simplicity, effortless depth',
        'voice_traits': ['effortless', 'transcendent', 'luminous', 'spare', 'beyond technique'],
        'theme_tendencies': ['essence', 'emptiness', 'presence', 'timelessness', 'unity'],
    }
}


class PersonaSelector:
    def __init__(self, personas_file: str = "haiku_personas.json"):
        """Load personas database"""
        self.personas_path = Path(__file__).parent / personas_file
        with open(self.personas_path, 'r') as f:
            self.personas_data = json.load(f)
        self.personas = self.personas_data['personas']

    def get_random_persona(self) -> Dict[str, Any]:
        """Get a random persona"""
        return random.choice(self.personas)

    def get_persona_by_id(self, persona_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific persona by ID"""
        for persona in self.personas:
            if persona['id'] == persona_id:
                return persona
        return None

    def filter_by_age(self, min_age: int = 18, max_age: int = 100) -> List[Dict[str, Any]]:
        """Get personas within age range"""
        return [p for p in self.personas if min_age <= p['age'] <= max_age]

    def filter_by_experience(self, min_exp: int = 0, max_exp: int = 60) -> List[Dict[str, Any]]:
        """Get personas within experience range"""
        return [p for p in self.personas if min_exp <= p['work_experience_years'] <= max_exp]

    def filter_by_location(self, location_type: str, location: str) -> List[Dict[str, Any]]:
        """Filter by location type ('childhood', 'occupation', 'current', 'poetry_education')"""
        return [p for p in self.personas if location in p['locations'].get(location_type, '')]

    def filter_by_education_type(self, edu_type: str) -> List[Dict[str, Any]]:
        """Filter by education type (e.g., 'MFA', 'Workshop', 'Mentorship')"""
        return [p for p in self.personas if p['poetry_education']['type'] == edu_type]

    def filter_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Filter personas who are fluent in a specific language"""
        return [p for p in self.personas if language in p.get('languages', {}).get('fluent', [])]

    def filter_multilingual(self, min_languages: int = 3) -> List[Dict[str, Any]]:
        """Filter personas who speak a minimum number of languages"""
        return [p for p in self.personas
                if p.get('languages', {}).get('total_languages', 1) >= min_languages]

    def get_diverse_sample(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get a diverse sample ensuring variety in age, location, and background"""
        if count >= len(self.personas):
            return random.sample(self.personas, len(self.personas))

        # Try to get diverse sample
        selected = []
        remaining = self.personas.copy()

        # Get one from each age bracket
        age_brackets = [
            (18, 30),   # Young
            (31, 50),   # Mid-career
            (51, 70),   # Senior
            (71, 100)   # Elder
        ]

        for min_age, max_age in age_brackets:
            bracket_personas = [p for p in remaining if min_age <= p['age'] <= max_age]
            if bracket_personas and len(selected) < count:
                chosen = random.choice(bracket_personas)
                selected.append(chosen)
                remaining.remove(chosen)

        # Fill remaining with random choices
        while len(selected) < count and remaining:
            chosen = random.choice(remaining)
            selected.append(chosen)
            remaining.remove(chosen)

        return selected

    def format_persona_description(self, persona: Dict[str, Any], detail_level: str = "brief") -> str:
        """Format persona into readable description

        Args:
            persona: Persona dictionary
            detail_level: 'brief', 'medium', or 'full'
        """
        lang_info = persona.get('languages', {})
        fluent_langs = ', '.join(lang_info.get('fluent', ['English']))
        name = persona.get('name', 'Unknown')
        name_details = persona.get('name_details', {})

        if detail_level == "brief":
            return (f"{name}: {persona['age']}-year-old {persona['occupation']}, "
                   f"studied at {persona['poetry_education']['institution']}, "
                   f"speaks {fluent_langs}, "
                   f"{persona['characteristic']}")

        elif detail_level == "medium":
            return (f"{name} - Age {persona['age']}, {persona['work_experience_years']} years work experience as {persona['occupation']}. "
                   f"Studied poetry at {persona['poetry_education']['institution']} in {persona['poetry_education']['location']}. "
                   f"Currently based in {persona['locations']['current']}. "
                   f"Languages: {fluent_langs}. "
                   f"Known for: {persona['characteristic']}.")

        else:  # full
            native_langs = ', '.join(lang_info.get('native', ['English']))
            total_langs = lang_info.get('total_languages', 1)

            name_info = f"  Name: {name}"
            if name_details.get('name_changed'):
                name_info += f" (born {name_details.get('birth_name')})"

            return (f"Poet Profile:\n"
                   f"{name_info}\n"
                   f"  Age: {persona['age']}\n"
                   f"  Occupation: {persona['occupation']} ({persona['work_experience_years']} years experience)\n"
                   f"  Education: {persona['poetry_education']['institution']} ({persona['poetry_education']['type']})\n"
                   f"  Childhood: {persona['locations']['childhood']}\n"
                   f"  Education: {persona['locations']['poetry_education']}\n"
                   f"  Work: {persona['locations']['occupation']}\n"
                   f"  Current: {persona['locations']['current']}\n"
                   f"  Languages: {fluent_langs} (Native: {native_langs}, Total: {total_langs})\n"
                   f"  Style: {persona['characteristic']}\n"
                   f"  Active since: {persona['active_since']}")

    def format_for_prompt(self, persona: Dict[str, Any], career_stage: Dict[str, Any] = None) -> str:
        """Format persona into a prompt for haiku generation

        Args:
            persona: Persona dictionary
            career_stage: Optional career stage info from get_persona_at_career_stage()
        """
        lang_info = persona.get('languages', {})
        fluent_langs = ', '.join(lang_info.get('fluent', ['English']))
        name = persona.get('name', 'the poet')

        if career_stage:
            # Use career-stage adjusted persona
            age = career_stage['adjusted_age']
            years_writing = career_stage['years_writing']
            stage_name = career_stage['stage_name']
            stage_info = CAREER_STAGES[stage_name]

            voice_traits = ', '.join(stage_info['voice_traits'][:3])
            theme_tendencies = ', '.join(stage_info['theme_tendencies'][:3])

            # Limit work experience to what they'd have at this point
            adjusted_work_exp = min(
                persona['work_experience_years'],
                max(0, age - 22)  # Assume started working around 22
            )

            prompt = (
                f"Write from the perspective of {name} at age {age}, "
                f"a poet {years_writing} years into their writing journey. "
                f"This is their {stage_name} period as a poet. "
                f"\n\nAt this stage, their voice is: {voice_traits}. "
                f"Their themes tend toward: {theme_tendencies}. "
                f"\n\n{stage_info['description']}."
            )

            # Add biographical context appropriate to the career stage
            if career_stage['percentage'] < 30:
                # Early career: mention education, early influences
                prompt += (
                    f"\n\nThey recently studied at {persona['poetry_education']['institution']}. "
                    f"They grew up in {persona['locations']['childhood']} and this shapes their early work."
                )
            elif career_stage['percentage'] < 60:
                # Mid-career: mention current work, developed style
                prompt += (
                    f"\n\nThey work as a {persona['occupation']} and have been writing for {years_writing} years. "
                    f"Their style is becoming known for: {persona['characteristic']}."
                )
            else:
                # Late career: full wisdom, all experiences
                prompt += (
                    f"\n\nAfter {years_writing} years of writing, they have developed a distinctive voice. "
                    f"Their work reflects a lifetime of experience as a {persona['occupation']}, "
                    f"journeys from {persona['locations']['childhood']} to {persona['locations']['current']}, "
                    f"and mastery of their craft. They are known for: {persona['characteristic']}."
                )

            prompt += f"\n\nThey are fluent in: {fluent_langs}."
            return prompt
        else:
            # Full biography format — all details shape the haiku
            locations = persona.get('locations', {})
            childhood = locations.get('childhood', '')
            occupation_loc = locations.get('occupation', '')
            edu_loc = locations.get('poetry_education', '')
            current = locations.get('current', '')

            edu = persona.get('poetry_education', {})
            edu_type = edu.get('type', '')
            edu_institution = edu.get('institution', '')
            edu_location = edu.get('location', '')

            birth_year = persona.get('birth_year', '')
            active_since = persona.get('active_since', '')
            years_writing = (datetime.now().year - active_since) if active_since else 0

            name_details = persona.get('name_details', {})

            prompt = f"Write from the perspective of {name}, born {birth_year}"
            if childhood:
                prompt += f", raised in {childhood}"
            prompt += f". They are now {persona['age']} years old"
            if current and current != childhood:
                prompt += f", living in {current}"
            prompt += ".\n\n"

            # Name change — indicates cultural transition, marriage, reinvention
            if name_details.get('name_changed') and name_details.get('birth_name'):
                prompt += f"They were born {name_details['birth_name']} and later took the name {name}. "

            # Education background shapes poetic sensibility
            prompt += f"POETIC FORMATION: "
            if edu_type and edu_type != 'Autodidact':
                prompt += f"Trained through a {edu_type} at {edu_institution}"
                if edu_location and edu_location != 'Various':
                    prompt += f" in {edu_location}"
                prompt += ". "
            elif edu_type == 'Autodidact':
                prompt += f"Self-taught poet ({edu_institution}). "
            else:
                prompt += f"Studied at {edu_institution}. "

            if active_since and years_writing > 0:
                prompt += f"Writing since {active_since} ({years_writing} years). "

            # Work life — occupational imagery and rhythms
            prompt += f"\n\nLIFE EXPERIENCE: {persona['work_experience_years']} years as a {persona['occupation']}"
            if occupation_loc and occupation_loc != current:
                prompt += f" (worked in {occupation_loc})"
            prompt += ". "

            prompt += f"Known for: {persona['characteristic']}.\n\n"

            # Geographic layers — each place contributes sensory memory
            places = []
            if childhood:
                places.append(f"childhood in {childhood}")
            if edu_location and edu_location not in ('Various', childhood):
                places.append(f"studies in {edu_location}")
            if occupation_loc and occupation_loc not in (childhood, current, 'Various'):
                places.append(f"work in {occupation_loc}")
            if current and current != childhood:
                places.append(f"home now in {current}")

            if len(places) > 1:
                prompt += f"LANDSCAPES THEY CARRY: {', '.join(places)}. Draw on the sensory details, weather, flora, fauna, and daily rhythms of these places.\n\n"

            prompt += f"Languages: {fluent_langs}."

            return prompt

    def calculate_career_span(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the career span for a persona

        Returns dict with:
            - start_year: When they became active
            - current_year: Current year
            - total_years: Total years as a poet
            - start_age: Age when they started
            - current_age: Current age
        """
        current_year = datetime.now().year
        active_since = persona.get('active_since', current_year - 10)
        current_age = persona['age']

        total_years = current_year - active_since
        start_age = current_age - total_years

        return {
            'start_year': active_since,
            'current_year': current_year,
            'total_years': max(1, total_years),  # At least 1 year
            'start_age': max(15, start_age),  # Assume started at least at 15
            'current_age': current_age
        }

    def get_stage_for_percentage(self, percentage: float) -> str:
        """Get the career stage name for a given percentage through career"""
        for stage_name, stage_info in CAREER_STAGES.items():
            min_pct, max_pct = stage_info['range']
            if min_pct <= percentage < max_pct:
                return stage_name
        return 'master'  # 100% = master

    def get_persona_at_career_stage(self, persona: Dict[str, Any],
                                    percentage: float = None,
                                    stage_name: str = None) -> Dict[str, Any]:
        """Get a persona adjusted to a specific point in their career

        Args:
            persona: The base persona
            percentage: Percentage through career (0-100), or
            stage_name: Name of stage ('emerging', 'developing', 'established', 'mature', 'master')

        Returns:
            Dict with adjusted persona info and career stage details
        """
        career = self.calculate_career_span(persona)

        # Determine percentage
        if percentage is not None:
            pct = max(0, min(100, percentage))
        elif stage_name:
            # Use middle of the stage's range
            stage_info = CAREER_STAGES.get(stage_name, CAREER_STAGES['established'])
            pct = (stage_info['range'][0] + stage_info['range'][1]) / 2
        else:
            # Random percentage
            pct = random.uniform(10, 90)

        # Calculate adjusted values
        years_into_career = int(career['total_years'] * (pct / 100))
        adjusted_age = career['start_age'] + years_into_career
        stage = self.get_stage_for_percentage(pct)

        return {
            'persona': persona,
            'percentage': pct,
            'stage_name': stage,
            'stage_info': CAREER_STAGES[stage],
            'years_writing': years_into_career,
            'adjusted_age': adjusted_age,
            'career_span': career,
            'year': career['start_year'] + years_into_career
        }

    def get_random_career_stage(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Get persona at a random career stage"""
        return self.get_persona_at_career_stage(persona)

    def format_career_stage_description(self, career_stage: Dict[str, Any]) -> str:
        """Format career stage info for display"""
        persona = career_stage['persona']
        return (
            f"{persona['name']} at age {career_stage['adjusted_age']} "
            f"({career_stage['stage_name'].upper()} period, "
            f"{career_stage['years_writing']} years into their poetry career, "
            f"circa {career_stage['year']})"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the persona database"""
        ages = [p['age'] for p in self.personas]
        experiences = [p['work_experience_years'] for p in self.personas]

        education_types = {}
        for p in self.personas:
            edu_type = p['poetry_education']['type']
            education_types[edu_type] = education_types.get(edu_type, 0) + 1

        current_locations = {}
        for p in self.personas:
            loc = p['locations']['current']
            current_locations[loc] = current_locations.get(loc, 0) + 1

        # Language statistics
        all_languages = set()
        language_counts = []
        multilingual_count = 0
        for p in self.personas:
            lang_info = p.get('languages', {})
            fluent = lang_info.get('fluent', [])
            total = lang_info.get('total_languages', 1)
            all_languages.update(fluent)
            language_counts.append(total)
            if total > 2:
                multilingual_count += 1

        return {
            "total_personas": len(self.personas),
            "age": {
                "min": min(ages),
                "max": max(ages),
                "average": sum(ages) / len(ages)
            },
            "experience": {
                "min": min(experiences),
                "max": max(experiences),
                "average": sum(experiences) / len(experiences)
            },
            "languages": {
                "total_unique_languages": len(all_languages),
                "average_languages_per_persona": sum(language_counts) / len(language_counts) if language_counts else 0,
                "multilingual_personas": multilingual_count,
                "most_common": list(all_languages)[:10]
            },
            "education_types": education_types,
            "unique_current_locations": len(current_locations),
            "most_common_locations": sorted(current_locations.items(), key=lambda x: x[1], reverse=True)[:5]
        }


def main():
    """Demo usage of PersonaSelector"""
    selector = PersonaSelector()

    print("=== HAIKU PERSONA SELECTOR DEMO ===\n")

    # Statistics
    stats = selector.get_statistics()
    print(f"Persona Database Statistics:")
    print(f"  Total personas: {stats['total_personas']}")
    print(f"  Age range: {stats['age']['min']}-{stats['age']['max']} (avg: {stats['age']['average']:.1f})")
    print(f"  Experience range: {stats['experience']['min']}-{stats['experience']['max']} years (avg: {stats['experience']['average']:.1f})")
    print(f"  Unique current locations: {stats['unique_current_locations']}")
    print(f"\n  Languages:")
    print(f"    Total unique languages: {stats['languages']['total_unique_languages']}")
    print(f"    Average per persona: {stats['languages']['average_languages_per_persona']:.1f}")
    print(f"    Multilingual (3+ languages): {stats['languages']['multilingual_personas']}")
    print(f"\n  Education types:")
    for edu_type, count in stats['education_types'].items():
        print(f"    {edu_type}: {count}")
    print()

    # Random persona
    print("Random Persona:")
    persona = selector.get_random_persona()
    print(selector.format_persona_description(persona, "full"))
    print()

    # Diverse sample
    print("Diverse Sample (5 personas):")
    diverse = selector.get_diverse_sample(5)
    for i, p in enumerate(diverse, 1):
        print(f"\n{i}. {selector.format_persona_description(p, 'medium')}")
    print()

    # Formatted for prompt
    print("Example Prompt Format:")
    print(selector.format_for_prompt(persona))
    print()

    # Filtering examples
    print("Young poets (18-30):")
    young = selector.filter_by_age(18, 30)
    print(f"  Found {len(young)} young poets")
    if young:
        sample = random.choice(young)
        print(f"  Example: {selector.format_persona_description(sample, 'brief')}")


if __name__ == "__main__":
    main()
