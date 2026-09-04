"""Train and evaluate a MultinomialNaiveBayes classifier using the Python API directly
(no CLI, no saved model file) - the shortest path from data to a scored model.

Run with:
    python examples/train_and_evaluate.py
"""

from nbtext.datasets import load_sample_reviews
from nbtext.metrics import confusion_matrix, precision_recall_f1, train_test_split
from nbtext.model import MultinomialNaiveBayes

texts, labels = load_sample_reviews()
train_x, test_x, train_y, test_y = train_test_split(texts, labels, test_size=0.25, seed=42)

model = MultinomialNaiveBayes(alpha=1.0, strip_stopwords=True)
model.fit(train_x, train_y)

print(f"Trained on {len(train_x)} reviews, holding out {len(test_x)} for evaluation.")
print(f"Classes: {model.classes_}  Vocabulary size: {len(model.vocabulary_)}\n")

preds = model.predict(test_x)
report = precision_recall_f1(test_y, preds, labels=model.classes_)
for label in model.classes_:
    r = report[label]
    print(f"{label:>5}:  precision={r['precision']:.2f}  recall={r['recall']:.2f}  f1={r['f1']:.2f}")

cm = confusion_matrix(test_y, preds, labels=model.classes_)
print(f"\nConfusion matrix: {cm}")

print("\nMost distinctive words per class:")
for label in model.classes_:
    words = ", ".join(w for w, _ in model.top_features(label, n=6))
    print(f"  {label}: {words}")

print("\nClassifying a few brand-new sentences the model has never seen:")
new_sentences = [
    "the packaging was excellent and it arrived so much faster than expected",
    "such a disappointing purchase, it broke after one single use",
    "works fine i suppose, nothing special either way",
]
for text, pred, proba in zip(new_sentences, model.predict(new_sentences), model.predict_proba(new_sentences)):
    proba_str = ", ".join(f"{c}={p:.2f}" for c, p in proba.items())
    print(f"  [{pred:>3}] ({proba_str})  {text!r}")
