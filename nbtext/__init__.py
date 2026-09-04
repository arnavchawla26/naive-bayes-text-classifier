"""naive-bayes-text-classifier: a dependency-free multinomial Naive Bayes text classifier."""

from .model import MultinomialNaiveBayes
from .tokenize import tokenize

__all__ = ["MultinomialNaiveBayes", "tokenize"]

__version__ = "0.1.0"
