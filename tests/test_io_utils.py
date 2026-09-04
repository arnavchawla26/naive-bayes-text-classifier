import pytest

from nbtext.io_utils import load_csv, load_lines, save_csv


class TestSaveLoadCsv:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "data.csv"
        texts = ["hello world", "goodbye, world!", 'quotes "inside" text']
        labels = ["a", "b", "a"]
        save_csv(str(path), texts, labels)
        loaded_texts, loaded_labels = load_csv(str(path))
        assert loaded_texts == texts
        assert loaded_labels == labels

    def test_custom_column_names(self, tmp_path):
        path = tmp_path / "data.csv"
        save_csv(str(path), ["t1", "t2"], ["l1", "l2"], text_col="review", label_col="sentiment")
        texts, labels = load_csv(str(path), text_col="review", label_col="sentiment")
        assert texts == ["t1", "t2"]
        assert labels == ["l1", "l2"]

    def test_mismatched_length_raises_on_save(self, tmp_path):
        with pytest.raises(ValueError):
            save_csv(str(tmp_path / "x.csv"), ["a", "b"], ["only-one"])


class TestLoadCsv:
    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("foo,bar\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing column"):
            load_csv(str(path))

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_csv(str(path))

    def test_header_only_raises(self, tmp_path):
        path = tmp_path / "header_only.csv"
        path.write_text("text,label\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no data rows"):
            load_csv(str(path))

    def test_preserves_row_order(self, tmp_path):
        path = tmp_path / "ordered.csv"
        path.write_text("text,label\nzzz,1\naaa,2\nmmm,3\n", encoding="utf-8")
        texts, _ = load_csv(str(path))
        assert texts == ["zzz", "aaa", "mmm"]


class TestLoadLines:
    def test_reads_one_per_line(self, tmp_path):
        path = tmp_path / "lines.txt"
        path.write_text("first line\nsecond line\nthird line\n", encoding="utf-8")
        assert load_lines(str(path)) == ["first line", "second line", "third line"]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "lines.txt"
        path.write_text("a\n\n\nb\n   \nc\n", encoding="utf-8")
        assert load_lines(str(path)) == ["a", "b", "c"]

    def test_strips_whitespace(self, tmp_path):
        path = tmp_path / "lines.txt"
        path.write_text("  padded text  \n", encoding="utf-8")
        assert load_lines(str(path)) == ["padded text"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        assert load_lines(str(path)) == []
