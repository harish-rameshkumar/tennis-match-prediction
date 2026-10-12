"""predict_match tests. Skipped until a model is trained:
    python -m tennis.predict --train
"""
import pytest

pytest.importorskip("joblib")


@pytest.fixture(scope="module")
def predict_match():
    from tennis.predict import _path, predict_match
    if not _path("logreg").exists():
        pytest.skip("train first: python -m tennis.predict --train")
    return predict_match


def test_swapping_players_gives_the_complement(predict_match):
    r1 = predict_match("Novak Djokovic", "Carlos Alcaraz", "Hard", "2024-06-01")
    r2 = predict_match("Carlos Alcaraz", "Novak Djokovic", "Hard", "2024-06-01")
    assert r1["p_a"] + r2["p_a"] == pytest.approx(1.0, abs=1e-6)
    assert r1["p_a"] + r1["p_b"] == pytest.approx(1.0)
    assert 0.0 <= r1["p_a"] <= 1.0


def test_top_features_are_reported(predict_match):
    r = predict_match("Novak Djokovic", "Carlos Alcaraz", "Hard", "2024-06-01")
    assert 0 < len(r["top_features"]) <= 8
    assert {"feature", "contribution", "favours"} <= set(r["top_features"][0])


def test_unknown_player_raises(predict_match):
    with pytest.raises(Exception) as e:
        predict_match("Zzyzx Qwerty", "Carlos Alcaraz", "Hard", "2024-06-01")
    assert "Unknown" in type(e.value).__name__
