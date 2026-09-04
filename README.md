# naive-bayes-text-classifier

A dependency-free multinomial Naive Bayes text classifier, written from scratch in pure Python. No numpy, no scikit-learn, no nltk - just `collections.Counter`, `math.log`, and the standard library.

Includes a `MultinomialNaiveBayes` model with Laplace smoothing, a small tokenizer, evaluation metrics (accuracy, precision/recall/F1, confusion matrix), a stratified train/test splitter, two built-in datasets, JSON model persistence, and an `nbtext` CLI for training, predicting, and inspecting a model's most distinctive words per class.

## Why Naive Bayes, from scratch

Multinomial Naive Bayes is one of the oldest practical text classifiers - fast to train, cheap to run, and surprisingly hard to beat on small labeled datasets. Implementing it directly (rather than calling `sklearn.naive_bayes.MultinomialNB`) means the whole pipeline - tokenization, Laplace smoothing, log-space scoring to avoid underflow, and "most predictive words" via log-likelihood ratio - is visible and testable end to end.

## Tech stack

- Python 3.10+, standard library only (`re`, `math`, `csv`, `json`, `random`, `argparse`) for the library and CLI
- [pytest](https://docs.pytest.org/) for the test suite (dev-only dependency)

## How it works

For a document represented as a bag of tokens, the classifier scores each class `c` as:

```
log P(c) + sum_over_tokens( log P(token | c) )
```

where `P(token | c)` uses Laplace (add-`alpha`) smoothing over the per-class word-count distribution:

```
P(w | c) = (count(w, c) + alpha) / (total_words(c) + alpha * |vocabulary|)
```

Everything happens in log space (summed, then normalized back to a probability with a numerically-stable log-sum-exp) so long documents don't underflow to zero. Words never seen in *any* training class are skipped at prediction time rather than penalized - the standard treatment for out-of-vocabulary tokens in a fixed-vocabulary multinomial model.

"Most predictive words per class" (`top_features` / `nbtext top-words`) ranks words by log-likelihood ratio `log P(w | class) - log P(w | not class)`, which surfaces words that are both frequent in that class *and* rare elsewhere - not just the most frequent words in the class overall (which tend to be stopwords every class shares).

## Install

```bash
git clone https://github.com/arnavchawla26/naive-bayes-text-classifier.git
cd naive-bayes-text-classifier
pip install -e ".[dev]"
```

## CLI usage

### `nbtext demo` - train and evaluate on a built-in dataset, no files needed

```
$ nbtext demo --dataset reviews --seed 42 --top-n 8
Dataset: reviews (120 examples, 90 train / 30 test)
Vocabulary size: 59

Accuracy: 1.0000 (30/30)

label        precision    recall        f1   support
neg              1.000     1.000     1.000        15
pos              1.000     1.000     1.000        15
macro avg        1.000     1.000     1.000

Confusion matrix (rows = true, cols = predicted):
                   neg       pos
neg                 15         0
pos                  0        15

Top 8 predictive words per class:
  neg: buying (2.43), regret (2.43), anyone (2.23), days (2.11), fell (2.11), money (2.11), short (2.11), stopped (2.11)
  pos: every (2.74), buy (2.16), everyone (2.16), happily (2.16), highly (2.16), penny (2.16), worth (2.16), exceeded (2.05)
```

`--dataset spam-ham` runs the same pipeline on the synthetic spam/ham generator instead (`--n-samples` controls its size).

### `nbtext train` - train and evaluate on your own CSV

```
$ nbtext train --data examples/sample_reviews.csv --strip-stopwords --test-size 0.25 --seed 42 --top-n 6 --model-out examples/reviews_model.json
Loaded 120 examples from examples/sample_reviews.csv (90 train / 30 test)
Classes: ['neg', 'pos']  Vocabulary size: 59

Accuracy: 1.0000 (30/30)
...
Saved model to examples/reviews_model.json
```

The CSV needs a header row with a text column and a label column (`text`/`label` by default, overridable with `--text-col`/`--label-col`).

### `nbtext predict` - classify new text with a saved model

```
$ nbtext predict --model examples/reviews_model.json --text "the packaging was excellent and it arrived so much faster than expected"
pos        [pos=0.853, neg=0.147]  'the packaging was excellent and it arrived so much faster...'

$ nbtext predict --model examples/reviews_model.json --file examples/new_reviews.txt
neg        [neg=0.865, pos=0.135]  'such a disappointing purchase, it broke after one single use'
pos        [pos=0.967, neg=0.033]  'works great and I would recommend it to a friend'
```

### `nbtext top-words` - most distinctive words for one class

```
$ nbtext top-words --model examples/reviews_model.json --label neg --n 8

Top 8 predictive words per class:
  neg: buying (2.43), regret (2.43), anyone (2.23), days (2.11), fell (2.11), money (2.11), short (2.11), stopped (2.11)
```

## Library usage

```python
from nbtext.datasets import load_sample_reviews
from nbtext.metrics import precision_recall_f1, train_test_split
from nbtext.model import MultinomialNaiveBayes

texts, labels = load_sample_reviews()
train_x, test_x, train_y, test_y = train_test_split(texts, labels, test_size=0.25, seed=42)

model = MultinomialNaiveBayes(alpha=1.0, strip_stopwords=True).fit(train_x, train_y)

preds = model.predict(test_x)
report = precision_recall_f1(test_y, preds, labels=model.classes_)

print(model.top_features("pos", n=5))          # most distinctive words for "pos"
print(model.predict_proba(["a brand new sentence"]))  # {'neg': ..., 'pos': ...}
```

See [`examples/train_and_evaluate.py`](examples/train_and_evaluate.py) (pure API, no files) and [`examples/classify_new_text.py`](examples/classify_new_text.py) (train -> save -> reload -> predict) for complete runnable scripts - both are copied verbatim into this README's history from an actual run.

## Datasets

- **`load_sample_reviews()`** - 120 short positive/negative product-review sentences (60/60, balanced), built by combining 10 sentiment adjectives x 6 sentence templates x 8 rotating subject nouns. Labels: `pos`/`neg`.

  An earlier version of this dataset was ~60 fully hand-written, largely unique sentences. On a held-out split, most of the actual sentiment vocabulary ("outstanding", "flimsy", "impressive", ...) turned out to appear in only *one* sentence each - so whichever split it landed in, that word was out-of-vocabulary for the other split, and the classifier had almost nothing to learn from beyond stray stopword frequencies (measured: ~50% held-out accuracy, i.e. chance, on a balanced binary problem). The template/word-bank version deliberately reuses the same adjectives and subjects across many sentences so every discriminative word survives a train/test split - Laplace smoothing only helps with sparse counts for words the model has actually *seen* during training.

- **`make_spam_ham(n_samples=200, seed=42)`** - a parametric synthetic generator: messages are random word sequences drawn mostly from a spam-associated or ham-associated word bank, mixed with a little shared "common word" noise so the task is learnable but not a trivial keyword lookup. Reproducible via `seed`, scalable via `n_samples`. Labels: `spam`/`ham`.

You can also bring your own labeled CSV via `nbtext train --data yours.csv`.

## Current status

**v1 - functional and tested.** Core model (fit/predict/predict_proba/top_features), tokenizer, metrics (accuracy/precision/recall/F1/confusion matrix), stratified train/test split, two datasets, CSV + JSON I/O, and the full `nbtext` CLI (`demo`/`train`/`predict`/`top-words`) are implemented and covered by 126 passing tests, including a real subprocess end-to-end run of the installed `nbtext` console script.

One real issue was caught and fixed before this was pushed: the original hand-written review dataset scored ~50% (chance) held-out accuracy because most sentiment words appeared in only one sentence each and fell out-of-vocabulary across the train/test split - see the **Datasets** section above for the fix and why it worked.

Not yet built: n-gram features (the tokenizer supports bigrams via `nbtext.tokenize.ngrams`, but the CLI doesn't expose them yet), a Bernoulli (presence/absence) event-model variant, k-fold cross-validation, and a stdin-streaming mode for `nbtext predict`. None of these are blocking - the classifier, CLI, and persistence layer are complete and working end to end.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
