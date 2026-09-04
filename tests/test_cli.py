import json
import subprocess
import sys

import pytest

from nbtext.cli import main


def _write_csv(path, rows, text_col="text", label_col="label"):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([text_col, label_col])
        writer.writerows(rows)


TOY_ROWS = [
    ("free cash prize now click", "spam"),
    ("urgent offer claim your cash", "spam"),
    ("win a free prize today", "spam"),
    ("team meeting scheduled tomorrow", "ham"),
    ("project report attached for review", "ham"),
    ("lunch with the team at noon", "ham"),
    ("cheap loans guaranteed approval now", "spam"),
    ("please review the attached notes", "ham"),
    ("limited time discount act now", "spam"),
    ("reminder about tomorrow's meeting", "ham"),
]


class TestDemoCommand:
    def test_demo_reviews_exits_zero(self, capsys):
        rc = main(["demo", "--dataset", "reviews", "--seed", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Accuracy" in out
        assert "Confusion matrix" in out
        assert "Top" in out

    def test_demo_spam_ham_exits_zero(self, capsys):
        rc = main(["demo", "--dataset", "spam-ham", "--n-samples", "60", "--seed", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Accuracy" in out
        assert "spam" in out and "ham" in out


class TestTrainCommand:
    def test_train_without_model_out(self, tmp_path, capsys):
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, TOY_ROWS)
        rc = main(["train", "--data", str(csv_path), "--seed", "0", "--test-size", "0.3"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Accuracy" in out
        assert "Classes:" in out

    def test_train_saves_model(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        model_path = tmp_path / "model.json"
        _write_csv(csv_path, TOY_ROWS)
        rc = main(
            ["train", "--data", str(csv_path), "--seed", "0", "--test-size", "0.3", "--model-out", str(model_path)]
        )
        assert rc == 0
        assert model_path.exists()
        data = json.loads(model_path.read_text(encoding="utf-8"))
        assert set(data["classes_"]) == {"spam", "ham"}

    def test_train_custom_columns(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, TOY_ROWS, text_col="msg", label_col="cls")
        rc = main(
            [
                "train",
                "--data",
                str(csv_path),
                "--text-col",
                "msg",
                "--label-col",
                "cls",
                "--test-size",
                "0.3",
                "--seed",
                "0",
            ]
        )
        assert rc == 0

    def test_train_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["train", "--data", str(tmp_path / "nope.csv")])


class TestPredictCommand:
    def _train_and_save(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        model_path = tmp_path / "model.json"
        _write_csv(csv_path, TOY_ROWS)
        main(["train", "--data", str(csv_path), "--test-size", "0.3", "--seed", "0", "--model-out", str(model_path)])
        return model_path

    def test_predict_single_text(self, tmp_path, capsys):
        model_path = self._train_and_save(tmp_path)
        rc = main(["predict", "--model", str(model_path), "--text", "free cash prize click now"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "spam" in out

    def test_predict_from_file(self, tmp_path, capsys):
        model_path = self._train_and_save(tmp_path)
        lines_path = tmp_path / "lines.txt"
        lines_path.write_text("free cash prize now\nteam meeting tomorrow\n", encoding="utf-8")
        rc = main(["predict", "--model", str(model_path), "--file", str(lines_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert out.count("\n") >= 2

    def test_predict_empty_file_returns_error(self, tmp_path, capsys):
        model_path = self._train_and_save(tmp_path)
        lines_path = tmp_path / "empty.txt"
        lines_path.write_text("   \n\n", encoding="utf-8")
        rc = main(["predict", "--model", str(model_path), "--file", str(lines_path)])
        assert rc == 1

    def test_predict_requires_text_or_file(self, tmp_path):
        model_path = self._train_and_save(tmp_path)
        with pytest.raises(SystemExit):
            main(["predict", "--model", str(model_path)])

    def test_predict_text_and_file_mutually_exclusive(self, tmp_path):
        model_path = self._train_and_save(tmp_path)
        with pytest.raises(SystemExit):
            main(["predict", "--model", str(model_path), "--text", "a", "--file", "b.txt"])


class TestTopWordsCommand:
    def test_top_words_known_label(self, tmp_path, capsys):
        csv_path = tmp_path / "data.csv"
        model_path = tmp_path / "model.json"
        _write_csv(csv_path, TOY_ROWS)
        main(["train", "--data", str(csv_path), "--test-size", "0.3", "--seed", "0", "--model-out", str(model_path)])
        rc = main(["top-words", "--model", str(model_path), "--label", "spam", "--n", "5"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "spam" in out

    def test_top_words_unknown_label_returns_error(self, tmp_path, capsys):
        csv_path = tmp_path / "data.csv"
        model_path = tmp_path / "model.json"
        _write_csv(csv_path, TOY_ROWS)
        main(["train", "--data", str(csv_path), "--test-size", "0.3", "--seed", "0", "--model-out", str(model_path)])
        rc = main(["top-words", "--model", str(model_path), "--label", "not-a-class"])
        assert rc == 2


class TestArgumentParsing:
    def test_no_command_exits_nonzero(self):
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_command_exits_nonzero(self):
        with pytest.raises(SystemExit):
            main(["bogus-command"])


class TestEndToEndSubprocess:
    def test_installed_console_script_runs(self):
        """Exercise the real, installed `nbtext` entry point, not the Python API."""
        result = subprocess.run(
            [sys.executable, "-m", "nbtext.cli", "demo", "--dataset", "reviews", "--seed", "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Accuracy" in result.stdout

    def test_installed_console_script_entry_point(self):
        result = subprocess.run(
            ["nbtext", "demo", "--dataset", "spam-ham", "--n-samples", "40", "--seed", "2"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Accuracy" in result.stdout
