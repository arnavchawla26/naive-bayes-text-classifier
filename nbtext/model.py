"""A dependency-free multinomial Naive Bayes text classifier.

No numpy, no scikit-learn - just ``collections.Counter`` and ``math.log``.
Implements the standard multinomial-event-model Naive Bayes classifier
(as used for bag-of-words text classification) with Laplace (add-alpha)
smoothing.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .tokenize import tokenize

Tokenizer = Callable[[str], List[str]]


class NotFittedError(RuntimeError):
    """Raised when a prediction/inspection method is called before .fit()."""


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes classifier for text.

    For a document represented as a bag of tokens, the classifier scores
    each class ``c`` as::

        log P(c) + sum_over_tokens( log P(token | c) )

    where ``P(token | c)`` uses Laplace/add-``alpha`` smoothing over the
    per-class word-count distribution:

        P(w | c) = (count(w, c) + alpha) / (total_words(c) + alpha * |V|)

    Args:
        alpha: additive (Laplace) smoothing parameter. ``alpha=1.0`` is
            classic Laplace smoothing; smaller values weight the training
            counts more heavily, larger values smooth more aggressively.
        tokenizer: optional custom function ``str -> List[str]``. If not
            given, :func:`nbtext.tokenize.tokenize` is used with the
            ``lowercase``/``strip_stopwords``/``stopwords``/``min_length``
            keyword arguments below.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        tokenizer: Optional[Tokenizer] = None,
        *,
        lowercase: bool = True,
        strip_stopwords: bool = False,
        stopwords: Optional[Iterable[str]] = None,
        min_length: int = 1,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = float(alpha)
        self._custom_tokenizer = tokenizer
        self._lowercase = lowercase
        self._strip_stopwords = strip_stopwords
        self._stopwords = list(stopwords) if stopwords is not None else None
        self._min_length = min_length

        # Fitted state
        self.classes_: List[str] = []
        self.class_count_: Dict[str, int] = {}
        self.class_log_prior_: Dict[str, float] = {}
        self.class_word_counts_: Dict[str, Dict[str, int]] = {}
        self.class_total_words_: Dict[str, int] = {}
        self.vocabulary_: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        if self._custom_tokenizer is not None:
            return self._custom_tokenizer(text)
        return tokenize(
            text,
            lowercase=self._lowercase,
            strip_stopwords=self._strip_stopwords,
            stopwords=self._stopwords,
            min_length=self._min_length,
        )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("This model has not been fit yet. Call .fit(texts, labels) first.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "MultinomialNaiveBayes":
        """Train the classifier on parallel lists of documents and labels."""
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels must have the same length ({len(texts)} != {len(labels)})")
        if len(texts) == 0:
            raise ValueError("cannot fit on an empty dataset")

        n_docs = len(texts)
        class_doc_count: Counter = Counter(labels)
        self.classes_ = sorted(class_doc_count)
        self.class_count_ = dict(class_doc_count)
        self.class_log_prior_ = {c: math.log(class_doc_count[c] / n_docs) for c in self.classes_}

        word_counts: Dict[str, Counter] = {c: Counter() for c in self.classes_}
        for text, label in zip(texts, labels):
            word_counts[label].update(self._tokenize(text))

        vocab: set = set()
        for c in self.classes_:
            vocab.update(word_counts[c])
        self.vocabulary_ = sorted(vocab)

        self.class_word_counts_ = {c: dict(word_counts[c]) for c in self.classes_}
        self.class_total_words_ = {c: sum(word_counts[c].values()) for c in self.classes_}
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Scoring internals
    # ------------------------------------------------------------------
    def _word_log_prob(self, cls: str, word: str) -> float:
        """log P(word | cls) with Laplace smoothing over the full vocabulary."""
        count = self.class_word_counts_[cls].get(word, 0)
        total = self.class_total_words_[cls]
        v = len(self.vocabulary_)
        return math.log((count + self.alpha) / (total + self.alpha * v))

    def _class_scores(self, tokens: List[str]) -> Dict[str, float]:
        scores = {}
        for c in self.classes_:
            score = self.class_log_prior_[c]
            for tok in tokens:
                # Out-of-vocabulary tokens (never seen in *any* class during
                # training) carry no information and are skipped rather than
                # penalized - this is the standard treatment for unseen
                # words in a fixed-vocabulary multinomial NB model.
                if tok in self._vocab_set:
                    score += self._word_log_prob(c, tok)
            scores[c] = score
        return scores

    @property
    def _vocab_set(self) -> set:
        # Recomputed lazily rather than stored twice; cheap relative to
        # scoring itself and keeps to_dict()/from_dict() round-trips simple.
        return set(self.vocabulary_)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def decision_function(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        """Return raw (unnormalized) joint log-probability per class per document."""
        self._require_fitted()
        return [self._class_scores(self._tokenize(t)) for t in texts]

    def predict_log_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        """Return normalized log-posterior P(class | doc) per document."""
        raw = self.decision_function(texts)
        out = []
        for scores in raw:
            max_score = max(scores.values())
            # log-sum-exp for numerical stability
            log_denom = max_score + math.log(sum(math.exp(s - max_score) for s in scores.values()))
            out.append({c: s - log_denom for c, s in scores.items()})
        return out

    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        """Return normalized posterior probabilities P(class | doc) per document."""
        return [{c: math.exp(lp) for c, lp in doc.items()} for doc in self.predict_log_proba(texts)]

    def predict(self, texts: Sequence[str]) -> List[str]:
        """Return the single most likely class per document.

        Ties are broken deterministically in favor of the alphabetically
        first class (``self.classes_`` is sorted), matching Python's
        stable ``max`` behavior over a dict with sorted keys.
        """
        self._require_fitted()
        preds = []
        for scores in self.decision_function(texts):
            best = max(self.classes_, key=lambda c: scores[c])
            preds.append(best)
        return preds

    # ------------------------------------------------------------------
    # Model inspection
    # ------------------------------------------------------------------
    def top_features(self, label: str, n: int = 10) -> List[Tuple[str, float]]:
        """Return the ``n`` words most distinctive of ``label``.

        Distinctiveness is the log-likelihood ratio ``log P(w|label) -
        log P(w|not label)``, where "not label" pools the word counts of
        every other class. This surfaces words that are both frequent in
        ``label`` and rare elsewhere, rather than just the most frequent
        words in ``label`` overall (which tend to be uninformative
        function words shared by every class).
        """
        self._require_fitted()
        if label not in self.classes_:
            raise ValueError(f"unknown label {label!r}; known classes: {self.classes_}")

        other_counts: Counter = Counter()
        other_total = 0
        for c in self.classes_:
            if c == label:
                continue
            other_counts.update(self.class_word_counts_[c])
            other_total += self.class_total_words_[c]

        v = len(self.vocabulary_)
        scores = []
        for word in self.vocabulary_:
            log_p_label = self._word_log_prob(label, word)
            other_count = other_counts.get(word, 0)
            log_p_other = math.log((other_count + self.alpha) / (other_total + self.alpha * v))
            scores.append((word, log_p_label - log_p_other))

        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return scores[:n]

    # ------------------------------------------------------------------
    # Serialization support (see nbtext.serialize for the JSON layer)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        self._require_fitted()
        return {
            "alpha": self.alpha,
            "tokenizer_config": {
                "lowercase": self._lowercase,
                "strip_stopwords": self._strip_stopwords,
                "stopwords": self._stopwords,
                "min_length": self._min_length,
            },
            "classes_": self.classes_,
            "class_count_": self.class_count_,
            "class_log_prior_": self.class_log_prior_,
            "class_word_counts_": self.class_word_counts_,
            "class_total_words_": self.class_total_words_,
            "vocabulary_": self.vocabulary_,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultinomialNaiveBayes":
        cfg = data.get("tokenizer_config", {})
        model = cls(
            alpha=data["alpha"],
            lowercase=cfg.get("lowercase", True),
            strip_stopwords=cfg.get("strip_stopwords", False),
            stopwords=cfg.get("stopwords"),
            min_length=cfg.get("min_length", 1),
        )
        model.classes_ = list(data["classes_"])
        model.class_count_ = dict(data["class_count_"])
        model.class_log_prior_ = {k: float(v) for k, v in data["class_log_prior_"].items()}
        model.class_word_counts_ = {c: dict(counts) for c, counts in data["class_word_counts_"].items()}
        model.class_total_words_ = dict(data["class_total_words_"])
        model.vocabulary_ = list(data["vocabulary_"])
        model._fitted = True
        return model

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if not self._fitted:
            return f"MultinomialNaiveBayes(alpha={self.alpha}, fitted=False)"
        return (
            f"MultinomialNaiveBayes(alpha={self.alpha}, classes={self.classes_}, "
            f"vocab_size={len(self.vocabulary_)}, fitted=True)"
        )
