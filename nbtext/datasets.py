"""Small labeled text datasets for demos, tests, and quick experiments.

Two sources are provided:

* :data:`SAMPLE_REVIEWS` / :func:`load_sample_reviews` - a template-generated
  set of short positive/negative product-review-style sentences, built by
  combining a bank of sentiment adjectives, sentence templates, and product
  nouns. Real (if formulaic) English text, useful for a demo that reads
  like an actual sentiment-analysis task rather than random word salad.

  Earlier versions of this dataset were ~60 fully hand-written, largely
  unique sentences. On a held-out split, most of the actual sentiment
  vocabulary ("outstanding", "flimsy", "impressive", ...) turned out to
  appear in only *one* sentence each - so whichever split it landed in,
  the word was out-of-vocabulary for the other split, and the classifier
  had almost nothing to learn from beyond stray stopword frequencies
  (observed: ~50% held-out accuracy, i.e. chance, on a balanced binary
  problem). The template/word-bank construction below deliberately reuses
  the same 10 adjectives across many sentences (and rotates 8 subject
  nouns and 6 templates) so every discriminative word appears often enough
  to survive a train/test split - which is exactly the sparse-data problem
  Laplace smoothing helps with, but only once a word has been *seen* at
  all in training.
* :func:`make_spam_ham` - a parametric synthetic generator that builds
  spam/ham-style messages from two word banks with different sampling
  weights. Reproducible via ``seed`` and scalable via ``n_samples``, which
  makes it useful for tests and for exercising the classifier on larger
  inputs than the embedded review set.
"""

from __future__ import annotations

import random
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Embedded sentiment dataset (template + word-bank generated, deterministic)
# ---------------------------------------------------------------------------

_POS_ADJECTIVES = [
    "amazing", "fantastic", "excellent", "wonderful", "outstanding",
    "great", "impressive", "reliable", "delightful", "superb",
]
_NEG_ADJECTIVES = [
    "terrible", "awful", "horrible", "disappointing", "unreliable",
    "poor", "frustrating", "flimsy", "overpriced", "dreadful",
]

_SUBJECTS = [
    "product", "purchase", "gadget", "device", "package",
    "service", "experience", "order",
]

_POS_TEMPLATES = [
    "this {subject} is {adj} and works perfectly every time",
    "i am extremely happy with this {adj} {subject}",
    "such a {adj} {subject}, i would happily buy it again",
    "the {subject} exceeded my expectations, truly {adj} overall",
    "absolutely {adj}, this {subject} is worth every penny",
    "what a {adj} {subject}, i highly recommend it to everyone",
]
_NEG_TEMPLATES = [
    "this {subject} is {adj} and stopped working within days",
    "i am extremely disappointed with this {adj} {subject}",
    "such a {adj} {subject}, i regret buying it at all",
    "the {subject} fell short of expectations, truly {adj} overall",
    "absolutely {adj}, this {subject} was a waste of money",
    "what a {adj} {subject}, i would not recommend it to anyone",
]


def _generate_reviews(adjectives: List[str], templates: List[str], subjects: List[str]) -> List[str]:
    """Build sentences from the cartesian product of adjectives x templates.

    The subject noun rotates deterministically (by adjective/template
    index) rather than being fixed per template, so subjects mix across
    sentences too - this keeps every (adjective, template) pair's output
    unique while still reusing the same small subject vocabulary often
    enough for the classifier to see it in both train and test splits.
    """
    reviews = []
    for i, adj in enumerate(adjectives):
        for j, template in enumerate(templates):
            subject = subjects[(i + j) % len(subjects)]
            reviews.append(template.format(adj=adj, subject=subject))
    return reviews


_POSITIVE_REVIEWS = _generate_reviews(_POS_ADJECTIVES, _POS_TEMPLATES, _SUBJECTS)
_NEGATIVE_REVIEWS = _generate_reviews(_NEG_ADJECTIVES, _NEG_TEMPLATES, _SUBJECTS)


def load_sample_reviews() -> Tuple[List[str], List[str]]:
    """Return the embedded review dataset as parallel ``(texts, labels)`` lists.

    Labels are ``"pos"`` / ``"neg"``. Deterministically ordered (positives
    then negatives) - callers that want a shuffled order should shuffle
    themselves or rely on :func:`nbtext.metrics.train_test_split`, which
    shuffles internally.
    """
    texts = list(_POSITIVE_REVIEWS) + list(_NEGATIVE_REVIEWS)
    labels = ["pos"] * len(_POSITIVE_REVIEWS) + ["neg"] * len(_NEGATIVE_REVIEWS)
    return texts, labels


# ---------------------------------------------------------------------------
# Synthetic spam/ham generator
# ---------------------------------------------------------------------------

_SPAM_WORDS = [
    "free", "winner", "cash", "prize", "urgent", "click", "link", "offer",
    "limited", "act", "now", "buy", "cheap", "discount", "guarantee",
    "credit", "loan", "viagra", "lottery", "claim", "congratulations",
    "bonus", "deal", "risk", "investment", "earn", "million", "subscribe",
]

_HAM_WORDS = [
    "meeting", "tomorrow", "project", "lunch", "thanks", "attached",
    "report", "schedule", "team", "review", "call", "morning", "office",
    "family", "weekend", "dinner", "flight", "hotel", "photos", "birthday",
    "class", "homework", "coffee", "update", "question", "reminder", "notes",
]

_COMMON_WORDS = ["the", "a", "is", "to", "for", "and", "your", "please", "will", "you"]


def _make_message(rng: random.Random, vocab: List[str], length_range: Tuple[int, int]) -> str:
    length = rng.randint(*length_range)
    words = []
    for _ in range(length):
        # Mostly draw from the class-specific vocab, occasionally from
        # generic connective words, so messages read a bit more like real
        # sentences and share *some* vocabulary across classes (making the
        # classification task non-trivial rather than a trivial vocab
        # lookup).
        pool = vocab if rng.random() < 0.75 else _COMMON_WORDS
        words.append(rng.choice(pool))
    return " ".join(words)


def make_spam_ham(
    n_samples: int = 200, seed: int = 42, message_length: Tuple[int, int] = (4, 12)
) -> Tuple[List[str], List[str]]:
    """Generate a synthetic, reproducible spam/ham text classification dataset.

    Each message is a random sequence of words drawn mostly from a
    class-specific word bank (spam-associated or ham-associated words),
    mixed with a small amount of shared "common word" noise, so the task
    is learnable but not trivially separable by a single keyword.

    Args:
        n_samples: total number of messages to generate (split evenly, +/-1,
            between the two classes).
        seed: seed for the internal RNG; the same seed always produces the
            same dataset.
        message_length: ``(min_words, max_words)`` per generated message.

    Returns:
        ``(texts, labels)`` with labels ``"spam"`` / ``"ham"``.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    rng = random.Random(seed)

    n_spam = n_samples // 2
    n_ham = n_samples - n_spam

    texts: List[str] = []
    labels: List[str] = []
    for _ in range(n_spam):
        texts.append(_make_message(rng, _SPAM_WORDS, message_length))
        labels.append("spam")
    for _ in range(n_ham):
        texts.append(_make_message(rng, _HAM_WORDS, message_length))
        labels.append("ham")

    # Interleave so a naive downstream .head()-style peek sees both classes,
    # and so any consumer that (incorrectly) skips shuffling before a split
    # still gets a reasonable mix rather than all-spam-then-all-ham.
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = (list(t) for t in zip(*combined))
    return texts, labels
