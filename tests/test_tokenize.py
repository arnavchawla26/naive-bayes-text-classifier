from nbtext.tokenize import DEFAULT_STOPWORDS, ngrams, tokenize


class TestTokenize:
    def test_basic_split(self):
        assert tokenize("Hello world") == ["hello", "world"]

    def test_lowercase_default(self):
        assert tokenize("HELLO World") == ["hello", "world"]

    def test_lowercase_false_preserves_case(self):
        assert tokenize("Hello World", lowercase=False) == ["Hello", "World"]

    def test_drops_numbers_and_punctuation(self):
        assert tokenize("Buy now!! 50% off, limited-time offer.") == [
            "buy",
            "now",
            "off",
            "limited",
            "time",
            "offer",
        ]

    def test_keeps_internal_apostrophes(self):
        assert tokenize("don't stop believing") == ["don't", "stop", "believing"]

    def test_strips_leading_trailing_apostrophes(self):
        # quoting style: 'word' should not keep the surrounding quotes
        assert tokenize("she said 'hello' to me") == ["she", "said", "hello", "to", "me"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_none_input(self):
        assert tokenize(None) == []

    def test_only_punctuation(self):
        assert tokenize("!!! ... ???") == []

    def test_min_length_filters_short_tokens(self):
        assert tokenize("a bb ccc dddd", min_length=3) == ["ccc", "dddd"]

    def test_strip_stopwords_default_list(self):
        result = tokenize("this is the best product", strip_stopwords=True)
        assert result == ["best", "product"]

    def test_strip_stopwords_custom_list(self):
        result = tokenize("cats and dogs are great", strip_stopwords=True, stopwords={"and", "are"})
        assert result == ["cats", "dogs", "great"]

    def test_strip_stopwords_false_keeps_everything(self):
        result = tokenize("this is the best product", strip_stopwords=False)
        assert result == ["this", "is", "the", "best", "product"]

    def test_default_stopwords_is_lowercase(self):
        assert all(w == w.lower() for w in DEFAULT_STOPWORDS)

    def test_order_preserved(self):
        assert tokenize("zebra apple mango") == ["zebra", "apple", "mango"]

    def test_repeated_words_kept(self):
        assert tokenize("go go go") == ["go", "go", "go"]


class TestNgrams:
    def test_unigrams_identity(self):
        tokens = ["a", "b", "c"]
        assert ngrams(tokens, 1) == tokens

    def test_bigrams(self):
        assert ngrams(["a", "b", "c"], 2) == ["a b", "b c"]

    def test_trigrams(self):
        assert ngrams(["a", "b", "c"], 3) == ["a b c"]

    def test_n_larger_than_tokens_returns_empty(self):
        assert ngrams(["a", "b"], 5) == []

    def test_empty_tokens(self):
        assert ngrams([], 2) == []

    def test_invalid_n_raises(self):
        import pytest

        with pytest.raises(ValueError):
            ngrams(["a", "b"], 0)
