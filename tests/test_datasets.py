import pytest

from nbtext.datasets import load_sample_reviews, make_spam_ham


class TestSampleReviews:
    def test_returns_parallel_lists(self):
        texts, labels = load_sample_reviews()
        assert len(texts) == len(labels)

    def test_only_two_labels(self):
        _, labels = load_sample_reviews()
        assert set(labels) == {"pos", "neg"}

    def test_reasonably_sized_and_balanced(self):
        texts, labels = load_sample_reviews()
        assert len(texts) >= 40
        assert labels.count("pos") == labels.count("neg")

    def test_all_texts_nonempty_strings(self):
        texts, _ = load_sample_reviews()
        assert all(isinstance(t, str) and t.strip() for t in texts)

    def test_deterministic(self):
        assert load_sample_reviews() == load_sample_reviews()

    def test_no_duplicate_reviews(self):
        texts, _ = load_sample_reviews()
        assert len(texts) == len(set(texts))


class TestMakeSpamHam:
    def test_returns_requested_count(self):
        texts, labels = make_spam_ham(n_samples=50, seed=1)
        assert len(texts) == 50
        assert len(labels) == 50

    def test_only_spam_and_ham_labels(self):
        _, labels = make_spam_ham(n_samples=40, seed=1)
        assert set(labels) == {"spam", "ham"}

    def test_roughly_balanced_classes(self):
        _, labels = make_spam_ham(n_samples=100, seed=1)
        assert labels.count("spam") == 50
        assert labels.count("ham") == 50

    def test_reproducible_with_same_seed(self):
        d1 = make_spam_ham(n_samples=30, seed=99)
        d2 = make_spam_ham(n_samples=30, seed=99)
        assert d1 == d2

    def test_different_seed_different_data(self):
        d1 = make_spam_ham(n_samples=30, seed=1)
        d2 = make_spam_ham(n_samples=30, seed=2)
        assert d1 != d2

    def test_message_length_bounds_respected(self):
        texts, _ = make_spam_ham(n_samples=20, seed=1, message_length=(3, 6))
        for text in texts:
            n_words = len(text.split())
            assert 3 <= n_words <= 6

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            make_spam_ham(n_samples=1)

    def test_texts_are_nonempty(self):
        texts, _ = make_spam_ham(n_samples=10, seed=1)
        assert all(t.strip() for t in texts)
