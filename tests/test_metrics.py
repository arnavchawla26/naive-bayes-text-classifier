import pytest

from nbtext.metrics import accuracy, confusion_matrix, precision_recall_f1, train_test_split


class TestAccuracy:
    def test_perfect_match(self):
        assert accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_no_match(self):
        assert accuracy(["a", "a"], ["b", "b"]) == 0.0

    def test_partial_match(self):
        assert accuracy(["a", "b", "c", "d"], ["a", "b", "x", "y"]) == 0.5

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            accuracy(["a"], ["a", "b"])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            accuracy([], [])


class TestConfusionMatrix:
    def test_diagonal_for_perfect_predictions(self):
        cm = confusion_matrix(["a", "b", "a", "b"], ["a", "b", "a", "b"])
        assert cm == {"a": {"a": 2, "b": 0}, "b": {"a": 0, "b": 2}}

    def test_off_diagonal_counts_errors(self):
        cm = confusion_matrix(["a", "a", "b"], ["b", "a", "b"])
        assert cm["a"]["b"] == 1
        assert cm["a"]["a"] == 1
        assert cm["b"]["b"] == 1

    def test_explicit_labels_include_absent_classes(self):
        cm = confusion_matrix(["a", "a"], ["a", "a"], labels=["a", "b", "c"])
        assert set(cm.keys()) == {"a", "b", "c"}
        assert cm["b"] == {"a": 0, "b": 0, "c": 0}

    def test_all_cells_present_even_if_zero(self):
        cm = confusion_matrix(["a"], ["a"])
        assert cm["a"]["a"] == 1


class TestPrecisionRecallF1:
    def test_perfect_predictions_score_one(self):
        report = precision_recall_f1(["a", "b", "a", "b"], ["a", "b", "a", "b"])
        assert report["a"]["precision"] == 1.0
        assert report["a"]["recall"] == 1.0
        assert report["a"]["f1"] == 1.0
        assert report["macro avg"]["f1"] == 1.0

    def test_support_counts_true_occurrences(self):
        report = precision_recall_f1(["a", "a", "b"], ["a", "b", "b"])
        assert report["a"]["support"] == 2
        assert report["b"]["support"] == 1

    def test_zero_denominator_yields_zero_not_error(self):
        # class "c" never predicted and never true -> should not divide by zero
        report = precision_recall_f1(["a", "b"], ["a", "b"], labels=["a", "b", "c"])
        assert report["c"]["precision"] == 0.0
        assert report["c"]["recall"] == 0.0
        assert report["c"]["f1"] == 0.0

    def test_known_precision_recall_values(self):
        # true:    a a a b b
        # pred:    a b a b b
        # class a: tp=2, fp=0, fn=1 -> precision=1.0, recall=2/3
        # class b: tp=2, fp=1, fn=0 -> precision=2/3, recall=1.0
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "b", "a", "b", "b"]
        report = precision_recall_f1(y_true, y_pred)
        assert report["a"]["precision"] == pytest.approx(1.0)
        assert report["a"]["recall"] == pytest.approx(2 / 3)
        assert report["b"]["precision"] == pytest.approx(2 / 3)
        assert report["b"]["recall"] == pytest.approx(1.0)

    def test_macro_avg_is_mean_of_per_class(self):
        report = precision_recall_f1(["a", "a", "b"], ["a", "b", "b"])
        labels = ["a", "b"]
        expected_precision = sum(report[l]["precision"] for l in labels) / 2
        assert report["macro avg"]["precision"] == pytest.approx(expected_precision)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            precision_recall_f1(["a"], ["a", "b"])


class TestTrainTestSplit:
    def test_split_sizes_add_up(self):
        texts = [f"doc{i}" for i in range(20)]
        labels = (["pos"] * 10) + (["neg"] * 10)
        tr_x, te_x, tr_y, te_y = train_test_split(texts, labels, test_size=0.2, seed=1)
        assert len(tr_x) + len(te_x) == 20
        assert len(tr_y) + len(te_y) == 20
        assert len(te_x) == len(te_y)
        assert len(tr_x) == len(tr_y)

    def test_reproducible_with_same_seed(self):
        texts = [f"doc{i}" for i in range(30)]
        labels = (["pos"] * 15) + (["neg"] * 15)
        split1 = train_test_split(texts, labels, test_size=0.3, seed=7)
        split2 = train_test_split(texts, labels, test_size=0.3, seed=7)
        assert split1 == split2

    def test_different_seeds_can_differ(self):
        texts = [f"doc{i}" for i in range(40)]
        labels = (["pos"] * 20) + (["neg"] * 20)
        split1 = train_test_split(texts, labels, test_size=0.3, seed=1)
        split2 = train_test_split(texts, labels, test_size=0.3, seed=2)
        assert split1 != split2

    def test_stratified_preserves_class_ratio_roughly(self):
        texts = [f"doc{i}" for i in range(100)]
        labels = (["pos"] * 80) + (["neg"] * 20)
        tr_x, te_x, tr_y, te_y = train_test_split(texts, labels, test_size=0.25, seed=3, stratify=True)
        te_pos = te_y.count("pos")
        te_neg = te_y.count("neg")
        # 25% of 80 pos = 20, 25% of 20 neg = 5
        assert te_pos == 20
        assert te_neg == 5

    def test_no_overlap_between_train_and_test(self):
        texts = [f"unique-doc-{i}" for i in range(20)]
        labels = (["pos"] * 10) + (["neg"] * 10)
        tr_x, te_x, _, _ = train_test_split(texts, labels, test_size=0.3, seed=5)
        assert set(tr_x).isdisjoint(set(te_x))

    def test_invalid_test_size_raises(self):
        with pytest.raises(ValueError):
            train_test_split(["a", "b"], ["x", "y"], test_size=0.0)
        with pytest.raises(ValueError):
            train_test_split(["a", "b"], ["x", "y"], test_size=1.0)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            train_test_split(["a", "b", "c"], ["x", "y"], test_size=0.3)

    def test_non_stratified_split(self):
        texts = [f"doc{i}" for i in range(20)]
        labels = (["pos"] * 10) + (["neg"] * 10)
        tr_x, te_x, tr_y, te_y = train_test_split(texts, labels, test_size=0.2, seed=1, stratify=False)
        assert len(te_x) == 4
        assert len(tr_x) == 16
