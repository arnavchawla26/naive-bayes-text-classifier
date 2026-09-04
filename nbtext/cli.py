"""Command-line interface: ``nbtext``.

Subcommands:
    demo        train + evaluate on an embedded dataset, no files needed
    train       train + evaluate on a CSV file, optionally save the model
    predict     classify new text with a saved model
    top-words   show the most distinctive words for a class in a saved model
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from .datasets import load_sample_reviews, make_spam_ham
from .io_utils import load_csv, load_lines
from .metrics import confusion_matrix, precision_recall_f1, train_test_split
from .model import MultinomialNaiveBayes
from .serialize import load_model, save_model


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _print_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]) -> None:
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true)
    print(f"\nAccuracy: {acc:.4f} ({correct}/{len(y_true)})")

    report = precision_recall_f1(y_true, y_pred, labels=labels)
    print(f"\n{'label':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label in labels:
        r = report[label]
        print(f"{label:<12}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>10.3f}{r['support']:>10}")
    m = report["macro avg"]
    print(f"{'macro avg':<12}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = " " * 12 + "".join(f"{l:>10}" for l in labels)
    print(header)
    for true_label in labels:
        row = "".join(f"{cm[true_label][pred_label]:>10}" for pred_label in labels)
        print(f"{true_label:<12}{row}")


def _print_top_words(model: MultinomialNaiveBayes, labels: List[str], n: int) -> None:
    print(f"\nTop {n} predictive words per class:")
    for label in labels:
        words = model.top_features(label, n=n)
        word_list = ", ".join(f"{w} ({score:.2f})" for w, score in words)
        print(f"  {label}: {word_list}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_demo(args: argparse.Namespace) -> int:
    if args.dataset == "reviews":
        texts, labels_all = load_sample_reviews()
    else:
        texts, labels_all = make_spam_ham(n_samples=args.n_samples, seed=args.seed)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels_all, test_size=args.test_size, seed=args.seed
    )
    model = MultinomialNaiveBayes(alpha=args.alpha, strip_stopwords=args.dataset == "reviews")
    model.fit(train_texts, train_labels)

    print(f"Dataset: {args.dataset} ({len(texts)} examples, {len(train_texts)} train / {len(test_texts)} test)")
    print(f"Vocabulary size: {len(model.vocabulary_)}")
    preds = model.predict(test_texts)
    _print_metrics(test_labels, preds, model.classes_)
    _print_top_words(model, model.classes_, args.top_n)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    texts, labels_all = load_csv(args.data, text_col=args.text_col, label_col=args.label_col)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels_all, test_size=args.test_size, seed=args.seed
    )
    model = MultinomialNaiveBayes(alpha=args.alpha, strip_stopwords=args.strip_stopwords)
    model.fit(train_texts, train_labels)

    print(f"Loaded {len(texts)} examples from {args.data} ({len(train_texts)} train / {len(test_texts)} test)")
    print(f"Classes: {model.classes_}  Vocabulary size: {len(model.vocabulary_)}")
    preds = model.predict(test_texts)
    _print_metrics(test_labels, preds, model.classes_)
    _print_top_words(model, model.classes_, args.top_n)

    if args.model_out:
        save_model(args.model_out, model)
        print(f"\nSaved model to {args.model_out}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)

    if args.text is not None:
        texts = [args.text]
    elif args.file is not None:
        texts = load_lines(args.file)
        if not texts:
            print(f"No non-blank lines found in {args.file}", file=sys.stderr)
            return 1
    else:
        print("error: must supply either --text or --file", file=sys.stderr)
        return 2

    preds = model.predict(texts)
    probs = model.predict_proba(texts)
    for text, pred, proba in zip(texts, preds, probs):
        proba_str = ", ".join(f"{c}={p:.3f}" for c, p in sorted(proba.items(), key=lambda kv: -kv[1]))
        display_text = text if len(text) <= 60 else text[:57] + "..."
        print(f"{pred:<10} [{proba_str}]  {display_text!r}")
    return 0


def _cmd_top_words(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    if args.label not in model.classes_:
        print(f"error: unknown label {args.label!r}; known classes: {model.classes_}", file=sys.stderr)
        return 2
    _print_top_words(model, [args.label], args.n)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nbtext",
        description="A dependency-free multinomial Naive Bayes text classifier.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="train and evaluate on a built-in dataset")
    p_demo.add_argument("--dataset", choices=["reviews", "spam-ham"], default="reviews")
    p_demo.add_argument("--n-samples", type=int, default=200, help="spam-ham dataset only: total messages")
    p_demo.add_argument("--alpha", type=float, default=1.0)
    p_demo.add_argument("--test-size", type=float, default=0.25)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--top-n", type=int, default=10)
    p_demo.set_defaults(func=_cmd_demo)

    p_train = sub.add_parser("train", help="train and evaluate on a CSV file")
    p_train.add_argument("--data", required=True, help="path to a CSV file with text/label columns")
    p_train.add_argument("--text-col", default="text")
    p_train.add_argument("--label-col", default="label")
    p_train.add_argument("--alpha", type=float, default=1.0)
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--top-n", type=int, default=10)
    p_train.add_argument("--strip-stopwords", action="store_true")
    p_train.add_argument("--model-out", default=None, help="path to save the trained model as JSON")
    p_train.set_defaults(func=_cmd_train)

    p_predict = sub.add_parser("predict", help="classify text with a saved model")
    p_predict.add_argument("--model", required=True, help="path to a model JSON file saved by 'train'")
    group = p_predict.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", default=None, help="a single string to classify")
    group.add_argument("--file", default=None, help="a file with one document per line")
    p_predict.set_defaults(func=_cmd_predict)

    p_top = sub.add_parser("top-words", help="show the most distinctive words for a class")
    p_top.add_argument("--model", required=True)
    p_top.add_argument("--label", required=True)
    p_top.add_argument("--n", type=int, default=15)
    p_top.set_defaults(func=_cmd_top_words)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
