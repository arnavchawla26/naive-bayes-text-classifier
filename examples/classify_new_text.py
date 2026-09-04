"""Train a spam/ham classifier, save it to disk, reload it, and classify new
messages - demonstrating the save/load round trip a CLI-driven workflow relies on.

Run with:
    python examples/classify_new_text.py
"""

import os
import tempfile

from nbtext.datasets import make_spam_ham
from nbtext.metrics import accuracy, train_test_split
from nbtext.model import MultinomialNaiveBayes
from nbtext.serialize import load_model, save_model

texts, labels = make_spam_ham(n_samples=300, seed=7)
train_x, test_x, train_y, test_y = train_test_split(texts, labels, test_size=0.2, seed=7)

model = MultinomialNaiveBayes(alpha=1.0).fit(train_x, train_y)
held_out_accuracy = accuracy(test_y, model.predict(test_x))
print(f"Trained on {len(train_x)} synthetic messages; held-out accuracy: {held_out_accuracy:.3f}")

with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "spam_ham_model.json")
    save_model(model_path, model)
    print(f"Saved model to {model_path}")

    reloaded = load_model(model_path)
    print("Reloaded model from disk - predictions below use the reloaded copy.\n")

    new_messages = [
        "urgent free cash prize claim your bonus now",
        "team meeting rescheduled to tomorrow morning, see attached notes",
        "limited time discount act now before the offer expires",
        "thanks for the report, let's discuss over coffee tomorrow",
    ]
    for text, pred, proba in zip(
        new_messages, reloaded.predict(new_messages), reloaded.predict_proba(new_messages)
    ):
        proba_str = ", ".join(f"{c}={p:.2f}" for c, p in sorted(proba.items(), key=lambda kv: -kv[1]))
        print(f"  [{pred:>4}] ({proba_str})  {text!r}")
