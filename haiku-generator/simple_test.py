#!/usr/bin/env python3
"""
Simple Haiku Generator Test
Generates 20 haiku for testing
"""

import anthropic
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Initialize client
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def generate_test_haiku():
    """Generate a small batch of haiku for testing"""

    print("Generating 20 test haiku...")

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2000,
        temperature=0.9,
        messages=[{
            "role": "user",
            "content": """Generate 20 excellent haiku.

Mix of traditional and contemporary styles.
Use concrete imagery, present tense, seasonal awareness where appropriate.

Format: One haiku per entry, separated by blank lines.
Output only the haiku, no numbering."""
        }]
    )

    # Parse haiku
    text = response.content[0].text
    haiku_list = []

    # Split by double newlines
    blocks = text.strip().split('\n\n')
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) == 3:
            haiku_list.append('\n'.join(lines))

    # Save to file
    output_dir = Path("haiku_output")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"test_haiku_{timestamp}.txt"

    with open(filename, 'w') as f:
        for i, haiku in enumerate(haiku_list, 1):
            f.write(f"{i}.\n{haiku}\n\n")

    print(f"\n✓ Generated {len(haiku_list)} haiku")
    print(f"✓ Saved to: {filename}")

    # Display a few
    print(f"\nSample haiku:\n")
    for haiku in haiku_list[:3]:
        print(haiku)
        print()

    return filename

if __name__ == "__main__":
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        print("Create a .env file with your API key")
        exit(1)

    generate_test_haiku()
