"""Tokenization utilities: no numpy/nltk/sklearn, just stdlib ``re``."""

from __future__ import annotations

import re
from typing import Iterable, List

# A small, deliberately conservative English stopword list. Not exhaustive -
# it exists to strip the highest-frequency function words that carry almost
# no class signal for a bag-of-words model, not to replicate a full NLP
# stopword corpus.
DEFAULT_STOPWORDS = frozenset(
    """
    a an the this that these those
    i me my we our you your he him his she her it its they them their
    is am are was were be been being
    do does did doing
    have has had having
    will would shall should may might must can could
    and or but if because as until while of at by for with about against
    between into through during before after above below to from up down
    in out on off over under again further then once here there when where
    why how all any both each few more most other some such no nor not only
    own same so than too very s t just don now
    """.split()
)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']*")


def tokenize(
    text: str,
    *,
    lowercase: bool = True,
    strip_stopwords: bool = False,
    stopwords: Iterable[str] | None = None,
    min_length: int = 1,
) -> List[str]:
    """Split ``text`` into word tokens.

    Tokens are maximal runs of ASCII letters and internal apostrophes (so
    "don't" stays one token, and numbers/punctuation are dropped entirely).
    This is intentionally simple: a bag-of-words Naive Bayes model does not
    need a full linguistic tokenizer, and a predictable, dependency-free
    tokenizer keeps behavior easy to reason about and to test.

    Args:
        text: input string.
        lowercase: fold case before returning tokens.
        strip_stopwords: drop tokens found in ``stopwords`` (or the built-in
            default list if ``stopwords`` is not given).
        stopwords: optional custom stopword set/iterable (used only when
            ``strip_stopwords`` is True).
        min_length: drop tokens shorter than this many characters (after any
            case-folding, before stopword removal).

    Returns:
        List of token strings, in order of appearance.
    """
    if text is None:
        return []

    raw = _TOKEN_RE.findall(text)
    tokens = []
    stop = frozenset(stopwords) if stopwords is not None else DEFAULT_STOPWORDS
    for tok in raw:
        # Trim stray leading/trailing apostrophes (e.g. quoting: 'word' -> word)
        tok = tok.strip("'")
        if not tok:
            continue
        if lowercase:
            tok = tok.lower()
        if len(tok) < min_length:
            continue
        if strip_stopwords and tok.lower() in stop:
            continue
        tokens.append(tok)
    return tokens


def ngrams(tokens: List[str], n: int) -> List[str]:
    """Join consecutive tokens into space-separated n-grams.

    ``n=1`` returns the tokens unchanged. Used optionally by callers that
    want bigram features in addition to unigrams; the model itself treats
    n-grams as opaque string features, so this is a pure preprocessing step.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return list(tokens)
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
