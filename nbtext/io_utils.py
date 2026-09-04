"""CSV and plain-text loading helpers for the CLI (stdlib ``csv`` only)."""

from __future__ import annotations

import csv
from typing import List, Tuple


def load_csv(path: str, text_col: str = "text", label_col: str = "label") -> Tuple[List[str], List[str]]:
    """Load a labeled text dataset from a CSV file with a header row.

    Args:
        path: path to a CSV file containing at least ``text_col`` and
            ``label_col`` columns.
        text_col: name of the column holding the document text.
        label_col: name of the column holding the class label.

    Returns:
        ``(texts, labels)`` parallel lists, in file order.

    Raises:
        ValueError: if the file has no header row, or is missing either
            requested column.
    """
    texts: List[str] = []
    labels: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: file is empty or has no header row")
        missing = [c for c in (text_col, label_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{path}: missing column(s) {missing} - found columns {reader.fieldnames}"
            )
        for row in reader:
            texts.append(row[text_col])
            labels.append(row[label_col])
    if not texts:
        raise ValueError(f"{path}: no data rows found")
    return texts, labels


def save_csv(path: str, texts: List[str], labels: List[str], text_col: str = "text", label_col: str = "label") -> None:
    """Write parallel ``texts``/``labels`` lists to a CSV file with a header row."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([text_col, label_col])
        for text, label in zip(texts, labels):
            writer.writerow([text, label])


def load_lines(path: str) -> List[str]:
    """Load unlabeled text, one document per non-blank line."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
