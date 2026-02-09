#!/usr/bin/env python3
"""
Semantic Similarity Filter for Haiku Generation

Uses sentence-transformers to detect semantically similar haiku
across runs, ensuring variety even when the same persona is used.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple

# Embeddings storage
EMBEDDINGS_FILE = Path("haiku_output") / "haiku_embeddings.npz"
TEXTS_FILE = Path("haiku_output") / "haiku_texts.json"

# Load model lazily (only when needed)
_model = None

def get_model():
    """Lazy load the embedding model"""
    global _model
    if _model is None:
        print("  Loading semantic similarity model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def load_existing_embeddings() -> Tuple[List[str], np.ndarray]:
    """Load embeddings and texts of all previously generated haiku"""
    if EMBEDDINGS_FILE.exists() and TEXTS_FILE.exists():
        try:
            with open(TEXTS_FILE, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            data = np.load(EMBEDDINGS_FILE)
            embeddings = data['embeddings']
            return texts, embeddings
        except Exception as e:
            print(f"  Warning: Could not load existing embeddings: {e}")
    return [], np.array([])


def save_embeddings(texts: List[str], embeddings: np.ndarray):
    """Save embeddings and texts to files"""
    existing_texts, existing_embeddings = load_existing_embeddings()

    # Combine with existing
    all_texts = existing_texts + texts
    if len(existing_embeddings) > 0:
        all_embeddings = np.vstack([existing_embeddings, embeddings])
    else:
        all_embeddings = embeddings

    # Ensure output directory exists
    EMBEDDINGS_FILE.parent.mkdir(exist_ok=True)

    # Save
    with open(TEXTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False)
    np.savez_compressed(EMBEDDINGS_FILE, embeddings=all_embeddings)


def filter_similar_haiku(new_haiku: List[str], threshold: float = 0.80) -> List[str]:
    """Filter out haiku that are semantically too similar to existing ones.

    Args:
        new_haiku: List of new haiku to check
        threshold: Similarity threshold (0-1). Higher = more permissive.
                   0.80 = 80% similar gets filtered
                   0.85 = 85% similar gets filtered (more permissive)
                   0.75 = 75% similar gets filtered (stricter)

    Returns:
        List of haiku that are sufficiently different from existing ones
    """
    if not new_haiku:
        return []

    model = get_model()
    existing_texts, existing_embeddings = load_existing_embeddings()

    # Embed all new haiku at once (faster)
    print(f"  Embedding {len(new_haiku)} haiku...")
    new_embeddings = model.encode(new_haiku, show_progress_bar=False)

    if len(existing_embeddings) == 0:
        # No existing haiku - all are novel
        # Save these for future comparison
        save_embeddings(new_haiku, new_embeddings)
        return new_haiku

    # Normalize embeddings for cosine similarity
    existing_normalized = existing_embeddings / np.linalg.norm(existing_embeddings, axis=1, keepdims=True)
    new_normalized = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix: new_haiku x existing_haiku
    similarities = np.dot(new_normalized, existing_normalized.T)

    # For each new haiku, get max similarity to any existing haiku
    max_similarities = np.max(similarities, axis=1)

    # Keep haiku below threshold
    novel_mask = max_similarities < threshold
    novel_haiku = [h for h, is_novel in zip(new_haiku, novel_mask) if is_novel]
    novel_embeddings = new_embeddings[novel_mask]

    # Save novel haiku for future comparison
    if len(novel_haiku) > 0:
        save_embeddings(novel_haiku, novel_embeddings)

    filtered_count = len(new_haiku) - len(novel_haiku)
    if filtered_count > 0:
        print(f"  Filtered {filtered_count} semantically similar haiku")

    return novel_haiku


def get_stats() -> dict:
    """Get statistics about stored embeddings"""
    texts, embeddings = load_existing_embeddings()
    return {
        'total_haiku': len(texts),
        'embedding_dimensions': embeddings.shape[1] if len(embeddings) > 0 else 0,
        'file_size_mb': (EMBEDDINGS_FILE.stat().st_size / 1024 / 1024) if EMBEDDINGS_FILE.exists() else 0
    }


def clear_embeddings():
    """Clear all stored embeddings (use to reset)"""
    if EMBEDDINGS_FILE.exists():
        EMBEDDINGS_FILE.unlink()
    if TEXTS_FILE.exists():
        TEXTS_FILE.unlink()
    print("Cleared all stored haiku embeddings")


if __name__ == "__main__":
    # Test/demo
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats = get_stats()
        print(f"Stored haiku: {stats['total_haiku']}")
        print(f"Embedding dimensions: {stats['embedding_dimensions']}")
        print(f"File size: {stats['file_size_mb']:.2f} MB")
    elif len(sys.argv) > 1 and sys.argv[1] == "clear":
        clear_embeddings()
    else:
        # Demo with test haiku
        test_haiku = [
            "Snow falls on the roof\nCat watches from the window\nWinter afternoon",
            "Rain drums on the glass\nDog sleeps by the fireplace\nAutumn evening comes",
            "Snow drifts to the ground\nThe cat stares through frosted pane\nCold winter evening",  # Similar to first
        ]

        print("Testing semantic filter...")
        novel = filter_similar_haiku(test_haiku, threshold=0.80)
        print(f"\nInput: {len(test_haiku)} haiku")
        print(f"Novel: {len(novel)} haiku")
        for h in novel:
            print(f"\n{h}")
