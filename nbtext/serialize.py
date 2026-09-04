"""JSON persistence for a trained :class:`~nbtext.model.MultinomialNaiveBayes`."""

from __future__ import annotations

import json

from .model import MultinomialNaiveBayes


def save_model(path: str, model: MultinomialNaiveBayes) -> None:
    """Serialize a fitted model to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2, sort_keys=True)


def load_model(path: str) -> MultinomialNaiveBayes:
    """Load a model previously written by :func:`save_model`."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return MultinomialNaiveBayes.from_dict(data)
