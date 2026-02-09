#!/usr/bin/env python3
"""
Theme Selector for Haiku Generation
Reads from haiku_themes.json and provides theme selection utilities
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


class ThemeSelector:
    def __init__(self, themes_file: str = "haiku_themes.json"):
        """Load themes database"""
        self.themes_path = Path(__file__).parent / themes_file
        with open(self.themes_path, 'r') as f:
            self.themes_data = json.load(f)

    def get_random_themes(self, count: int = 5, category: str = None) -> List[str]:
        """Get random themes from the database

        Args:
            count: Number of themes to return
            category: Optional category to select from
                     (e.g., 'urban_life', 'traditional_seasonal', 'environmental_concerns')
        """
        if category:
            themes = self._get_category_themes(category)
        else:
            themes = self._get_all_themes()

        return random.sample(themes, min(count, len(themes)))

    def get_seasonal_themes(self, season: str, count: int = 5) -> List[str]:
        """Get themes for a specific season

        Args:
            season: 'spring', 'summer', 'autumn', 'winter', or 'new_year'
            count: Number of themes to return
        """
        seasonal_data = self.themes_data['traditional_seasonal'].get(season, {})
        themes = []

        for category_themes in seasonal_data.values():
            if isinstance(category_themes, list):
                themes.extend(category_themes)

        return random.sample(themes, min(count, len(themes))) if themes else []

    def get_contemporary_themes(self, count: int = 5) -> List[str]:
        """Get random contemporary themes"""
        themes = []
        for category_themes in self.themes_data['contemporary_themes'].values():
            themes.extend(category_themes)
        return random.sample(themes, min(count, len(themes)))

    def get_mixed_themes(self, traditional: int = 3, contemporary: int = 2) -> List[str]:
        """Get a mix of traditional and contemporary themes"""
        trad_themes = []
        for season_data in self.themes_data['traditional_seasonal'].values():
            for category_themes in season_data.values():
                if isinstance(category_themes, list):
                    trad_themes.extend(category_themes)

        cont_themes = []
        for category_themes in self.themes_data['contemporary_themes'].values():
            cont_themes.extend(category_themes)

        selected = []
        if trad_themes:
            selected.extend(random.sample(trad_themes, min(traditional, len(trad_themes))))
        if cont_themes:
            selected.extend(random.sample(cont_themes, min(contemporary, len(cont_themes))))

        return selected

    def get_category_list(self) -> List[str]:
        """Get list of all available categories"""
        return self.themes_data.get('thematic_categories', [])

    def get_theme_by_category(self, category: str) -> List[str]:
        """Get all themes from a specific category"""
        return self._get_category_themes(category)

    def _get_all_themes(self) -> List[str]:
        """Get all themes from all categories"""
        themes = []

        # Traditional seasonal
        for season_data in self.themes_data['traditional_seasonal'].values():
            for category_themes in season_data.values():
                if isinstance(category_themes, list):
                    themes.extend(category_themes)

        # Contemporary
        for category_themes in self.themes_data['contemporary_themes'].values():
            themes.extend(category_themes)

        # Universal
        for category_themes in self.themes_data['universal_themes'].values():
            themes.extend(category_themes)

        return themes

    def _get_category_themes(self, category: str) -> List[str]:
        """Get themes from a specific category"""
        themes = []

        # Check contemporary themes
        if category in self.themes_data['contemporary_themes']:
            return self.themes_data['contemporary_themes'][category]

        # Check universal themes
        if category in self.themes_data['universal_themes']:
            return self.themes_data['universal_themes'][category]

        # Check traditional seasonal
        if category in self.themes_data['traditional_seasonal']:
            for cat_themes in self.themes_data['traditional_seasonal'][category].values():
                if isinstance(cat_themes, list):
                    themes.extend(cat_themes)
            return themes

        return themes

    def format_theme_prompt(self, themes: List[str]) -> str:
        """Format themes into a prompt-friendly string"""
        if not themes:
            return "Mix of traditional and contemporary themes"
        return f"Themes: {', '.join(themes)}"

    def get_theme_stats(self) -> Dict[str, int]:
        """Get statistics about the themes database"""
        stats = {
            'total_themes': len(self._get_all_themes()),
            'traditional_seasonal': 0,
            'contemporary': 0,
            'universal': 0
        }

        for season_data in self.themes_data['traditional_seasonal'].values():
            for category_themes in season_data.values():
                if isinstance(category_themes, list):
                    stats['traditional_seasonal'] += len(category_themes)

        for category_themes in self.themes_data['contemporary_themes'].values():
            stats['contemporary'] += len(category_themes)

        for category_themes in self.themes_data['universal_themes'].values():
            stats['universal'] += len(category_themes)

        return stats


def main():
    """Demo usage of ThemeSelector"""
    selector = ThemeSelector()

    print("=== HAIKU THEME SELECTOR DEMO ===\n")

    # Stats
    stats = selector.get_theme_stats()
    print(f"Theme Database Statistics:")
    print(f"  Total themes: {stats['total_themes']}")
    print(f"  Traditional seasonal: {stats['traditional_seasonal']}")
    print(f"  Contemporary: {stats['contemporary']}")
    print(f"  Universal: {stats['universal']}")
    print()

    # Random themes
    print("Random 5 themes:")
    for theme in selector.get_random_themes(5):
        print(f"  - {theme}")
    print()

    # Seasonal themes
    print("Spring themes:")
    for theme in selector.get_seasonal_themes('spring', 5):
        print(f"  - {theme}")
    print()

    # Contemporary themes
    print("Contemporary themes:")
    for theme in selector.get_contemporary_themes(5):
        print(f"  - {theme}")
    print()

    # Mixed themes
    print("Mixed traditional + contemporary:")
    for theme in selector.get_mixed_themes(3, 2):
        print(f"  - {theme}")
    print()

    # Categories
    print(f"Available categories: {len(selector.get_category_list())}")
    for i, category in enumerate(selector.get_category_list()[:5], 1):
        print(f"  {i}. {category}")
    print("  ...")


if __name__ == "__main__":
    main()
