"""Evaluation metrics and a train/test splitter - no numpy/sklearn."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Fraction of predictions that exactly match the true label."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot score an empty input")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[Sequence[str]] = None
) -> Dict[str, Dict[str, int]]:
    """Return ``matrix[true_label][predicted_label] = count``.

    Every ``(true_label, predicted_label)`` cell for every known label is
    present (defaulting to 0), so the result is safe to iterate over as a
    dense grid.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def precision_recall_f1(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[Sequence[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Per-class precision/recall/F1/support, plus a ``macro avg`` row.

    Returns a dict keyed by label (and ``"macro avg"``), each value a dict
    with ``precision``, ``recall``, ``f1``, and ``support`` (the true count
    of that label - omitted from the macro-avg row). Precision/recall are
    defined as 0.0 (not NaN) when their denominator is 0, so this never
    raises on classes with no predictions or no true examples.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    result: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        result[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    n = len(labels) if labels else 1
    result["macro avg"] = {
        "precision": sum(result[l]["precision"] for l in labels) / n,
        "recall": sum(result[l]["recall"] for l in labels) / n,
        "f1": sum(result[l]["f1"] for l in labels) / n,
    }
    return result


def train_test_split(
    texts: Sequence[str],
    labels: Sequence[str],
    test_size: float = 0.2,
    seed: Optional[int] = None,
    stratify: bool = True,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split parallel ``texts``/``labels`` into train and test sets.

    Args:
        test_size: fraction of examples (0 < test_size < 1) held out for
            the test set.
        seed: seed for the internal ``random.Random`` instance, for
            reproducible splits.
        stratify: if True (default), split each class independently so the
            train/test class ratio matches the overall class ratio as
            closely as rounding allows. If False, shuffle and split the
            whole dataset without regard to class.

    Returns:
        ``(train_texts, test_texts, train_labels, test_labels)``.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")

    rng = random.Random(seed)
    indices = list(range(len(texts)))

    test_idx: List[int] = []
    train_idx: List[int] = []

    if stratify:
        by_class: Dict[str, List[int]] = {}
        for i, label in zip(indices, labels):
            by_class.setdefault(label, []).append(i)
        for label, idxs in by_class.items():
            idxs = list(idxs)
            rng.shuffle(idxs)
            n_test = max(1, round(len(idxs) * test_size)) if len(idxs) > 1 else 0
            test_idx.extend(idxs[:n_test])
            train_idx.extend(idxs[n_test:])
    else:
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        test_idx = shuffled[:n_test]
        train_idx = shuffled[n_test:]

    if not train_idx:
        raise ValueError("test_size too large: no examples left for the training set")
    if not test_idx:
        raise ValueError("test_size too small: no examples selected for the test set")

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    return train_texts, test_texts, train_labels, test_labels
