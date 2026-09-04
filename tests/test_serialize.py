import json

from nbtext.model import MultinomialNaiveBayes
from nbtext.serialize import load_model, save_model


def _fitted_model():
    texts = ["free cash prize", "urgent click now", "team lunch meeting", "project report attached"]
    labels = ["spam", "spam", "ham", "ham"]
    return MultinomialNaiveBayes(alpha=0.5).fit(texts, labels)


class TestSaveLoadModel:
    def test_roundtrip_predictions_match(self, tmp_path):
        model = _fitted_model()
        path = tmp_path / "model.json"
        save_model(str(path), model)
        loaded = load_model(str(path))

        docs = ["free cash now", "team meeting report"]
        assert loaded.predict(docs) == model.predict(docs)

    def test_saved_file_is_valid_json(self, tmp_path):
        model = _fitted_model()
        path = tmp_path / "model.json"
        save_model(str(path), model)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["classes_"] == model.classes_
        assert data["alpha"] == model.alpha

    def test_loaded_model_top_features_match(self, tmp_path):
        model = _fitted_model()
        path = tmp_path / "model.json"
        save_model(str(path), model)
        loaded = load_model(str(path))
        assert loaded.top_features("spam", n=5) == model.top_features("spam", n=5)

    def test_loaded_model_vocabulary_matches(self, tmp_path):
        model = _fitted_model()
        path = tmp_path / "model.json"
        save_model(str(path), model)
        loaded = load_model(str(path))
        assert loaded.vocabulary_ == model.vocabulary_
