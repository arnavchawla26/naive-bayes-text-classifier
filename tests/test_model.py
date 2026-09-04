import math

import pytest

from nbtext.model import MultinomialNaiveBayes, NotFittedError


def _toy_dataset():
    texts = [
        "cheap loans free cash now",
        "win a free prize now click now",
        "urgent cash offer click now",
        "lets meet for lunch tomorrow",
        "attached is the project report",
        "see you at the team meeting",
    ]
    labels = ["spam", "spam", "spam", "ham", "ham", "ham"]
    return texts, labels


class TestFit:
    def test_fit_returns_self(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes()
        assert model.fit(texts, labels) is model

    def test_classes_sorted(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        assert model.classes_ == ["ham", "spam"]

    def test_class_counts(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        assert model.class_count_ == {"spam": 3, "ham": 3}

    def test_log_priors_match_class_frequency(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        assert model.class_log_prior_["spam"] == pytest.approx(math.log(0.5))
        assert model.class_log_prior_["ham"] == pytest.approx(math.log(0.5))

    def test_vocabulary_built_from_all_classes(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        assert "cash" in model.vocabulary_
        assert "meeting" in model.vocabulary_
        assert model.vocabulary_ == sorted(model.vocabulary_)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            MultinomialNaiveBayes().fit(["a", "b"], ["only-one"])

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError):
            MultinomialNaiveBayes().fit([], [])

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            MultinomialNaiveBayes(alpha=0)
        with pytest.raises(ValueError):
            MultinomialNaiveBayes(alpha=-1)


class TestNotFitted:
    def test_predict_before_fit_raises(self):
        model = MultinomialNaiveBayes()
        with pytest.raises(NotFittedError):
            model.predict(["hello"])

    def test_predict_proba_before_fit_raises(self):
        model = MultinomialNaiveBayes()
        with pytest.raises(NotFittedError):
            model.predict_proba(["hello"])

    def test_top_features_before_fit_raises(self):
        model = MultinomialNaiveBayes()
        with pytest.raises(NotFittedError):
            model.top_features("spam")

    def test_to_dict_before_fit_raises(self):
        model = MultinomialNaiveBayes()
        with pytest.raises(NotFittedError):
            model.to_dict()


class TestPredict:
    def test_predicts_clearly_separable_classes_correctly(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        preds = model.predict(["free cash prize now", "team lunch meeting tomorrow"])
        assert preds == ["spam", "ham"]

    def test_predict_returns_one_label_per_input(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        preds = model.predict(["a", "b", "c"])
        assert len(preds) == 3
        assert all(p in model.classes_ for p in preds)

    def test_predict_handles_empty_document(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        # An empty doc should still return a valid class (falls back to prior).
        preds = model.predict([""])
        assert preds[0] in model.classes_

    def test_out_of_vocabulary_words_do_not_crash(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        preds = model.predict(["supercalifragilisticexpialidocious zzzznotaword"])
        assert preds[0] in model.classes_

    def test_predict_matches_argmax_of_decision_function(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        docs = ["free cash now", "team meeting report"]
        preds = model.predict(docs)
        scores = model.decision_function(docs)
        for pred, score in zip(preds, scores):
            assert pred == max(score, key=score.get)


class TestPredictProba:
    def test_probabilities_sum_to_one(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        for proba in model.predict_proba(["free cash prize", "team lunch", "random unrelated text"]):
            assert sum(proba.values()) == pytest.approx(1.0, abs=1e-9)

    def test_probabilities_are_between_zero_and_one(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        for proba in model.predict_proba(["free cash prize now click"]):
            for p in proba.values():
                assert 0.0 <= p <= 1.0

    def test_predict_log_proba_exponentiates_to_predict_proba(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        docs = ["urgent free offer", "project meeting tomorrow"]
        log_probas = model.predict_log_proba(docs)
        probas = model.predict_proba(docs)
        for lp, p in zip(log_probas, probas):
            for c in model.classes_:
                assert math.exp(lp[c]) == pytest.approx(p[c])

    def test_confident_prediction_has_high_probability(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        proba = model.predict_proba(["free cash prize now urgent click"])[0]
        assert proba["spam"] > 0.9


class TestAlphaSmoothing:
    def test_larger_alpha_smooths_scores_closer_together(self):
        texts, labels = _toy_dataset()
        low_alpha = MultinomialNaiveBayes(alpha=0.01).fit(texts, labels)
        high_alpha = MultinomialNaiveBayes(alpha=10.0).fit(texts, labels)
        doc = ["free cash now"]
        low_proba = low_alpha.predict_proba(doc)[0]["spam"]
        high_proba = high_alpha.predict_proba(doc)[0]["spam"]
        # Heavier smoothing should pull the posterior away from the extreme
        # (closer to the class prior of 0.5) relative to very light smoothing.
        assert abs(high_proba - 0.5) <= abs(low_proba - 0.5)

    def test_unseen_word_probability_uses_alpha_over_denominator(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes(alpha=1.0).fit(texts, labels)
        v = len(model.vocabulary_)
        total = model.class_total_words_["spam"]
        expected = math.log(1.0 / (total + 1.0 * v))
        assert model._word_log_prob("spam", "totallyunseenword") == pytest.approx(expected)


class TestTopFeatures:
    def test_returns_requested_count(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        assert len(model.top_features("spam", n=3)) == 3

    def test_spam_words_rank_above_ham_words_for_spam_class(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        top_words = {w for w, _ in model.top_features("spam", n=5)}
        assert "cash" in top_words or "click" in top_words or "free" in top_words
        assert "meeting" not in top_words

    def test_scores_are_sorted_descending(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        scores = [s for _, s in model.top_features("ham", n=10)]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_label_raises(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        with pytest.raises(ValueError):
            model.top_features("not-a-real-class")


class TestMultiClass:
    def test_three_class_problem(self):
        texts = [
            "free cash prize now",
            "urgent click now offer",
            "team meeting report tomorrow",
            "lunch schedule attached notes",
            "python code function variable loop",
            "database query index table schema",
        ]
        labels = ["spam", "spam", "work", "work", "tech", "tech"]
        model = MultinomialNaiveBayes().fit(texts, labels)
        assert model.classes_ == ["spam", "tech", "work"]
        preds = model.predict(["free prize click now", "python loop variable", "team report tomorrow"])
        assert preds == ["spam", "tech", "work"]

    def test_three_class_probabilities_sum_to_one(self):
        texts = ["a a a b", "b b b c", "c c c a"]
        labels = ["x", "y", "z"]
        model = MultinomialNaiveBayes().fit(texts, labels)
        proba = model.predict_proba(["a b c"])[0]
        assert set(proba.keys()) == {"x", "y", "z"}
        assert sum(proba.values()) == pytest.approx(1.0)


class TestCustomTokenizer:
    def test_custom_tokenizer_is_used(self):
        calls = []

        def spy_tokenizer(text):
            calls.append(text)
            return text.split("|")

        model = MultinomialNaiveBayes(tokenizer=spy_tokenizer)
        model.fit(["a|b|c", "d|e|f"], ["x", "y"])
        assert calls == ["a|b|c", "d|e|f"]
        assert "a" in model.vocabulary_ and "d" in model.vocabulary_

    def test_strip_stopwords_reduces_vocabulary(self):
        texts, labels = _toy_dataset()
        with_stop = MultinomialNaiveBayes(strip_stopwords=False).fit(texts, labels)
        without_stop = MultinomialNaiveBayes(strip_stopwords=True).fit(texts, labels)
        assert len(without_stop.vocabulary_) <= len(with_stop.vocabulary_)


class TestToDictFromDict:
    def test_roundtrip_preserves_predictions(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes(alpha=0.7).fit(texts, labels)
        docs = ["free cash now", "team lunch tomorrow", "totally unrelated words here"]
        original_preds = model.predict(docs)
        original_proba = model.predict_proba(docs)

        restored = MultinomialNaiveBayes.from_dict(model.to_dict())
        restored_preds = restored.predict(docs)
        restored_proba = restored.predict_proba(docs)

        assert restored_preds == original_preds
        for orig, rest in zip(original_proba, restored_proba):
            for c in model.classes_:
                assert orig[c] == pytest.approx(rest[c])

    def test_roundtrip_preserves_alpha_and_classes(self):
        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes(alpha=2.5).fit(texts, labels)
        restored = MultinomialNaiveBayes.from_dict(model.to_dict())
        assert restored.alpha == 2.5
        assert restored.classes_ == model.classes_
        assert restored.vocabulary_ == model.vocabulary_

    def test_to_dict_is_json_serializable(self):
        import json

        texts, labels = _toy_dataset()
        model = MultinomialNaiveBayes().fit(texts, labels)
        # Should not raise.
        json.dumps(model.to_dict())

    def test_repr_reflects_fit_state(self):
        model = MultinomialNaiveBayes()
        assert "fitted=False" in repr(model)
        texts, labels = _toy_dataset()
        model.fit(texts, labels)
        assert "fitted=True" in repr(model)
