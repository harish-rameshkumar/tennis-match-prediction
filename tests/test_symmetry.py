"""Symmetry tests for combine() and for the logistic regression."""
import numpy as np
import pytest

from tennis.inference import combine


def test_combine_antisymmetric():
    rng = np.random.default_rng(0)
    p_ab, p_ba = rng.random(1000), rng.random(1000)
    # (p_ab + 1 - p_ba)/2 + (p_ba + 1 - p_ab)/2 == 1 exactly in real arithmetic;
    # allow float rounding of a few ulps.
    np.testing.assert_allclose(combine(p_ab, p_ba) + combine(p_ba, p_ab), 1.0,
                               rtol=0, atol=1e-12)


def test_combine_scalars_and_edges():
    assert combine(0.5, 0.5) == pytest.approx(0.5)
    assert combine(1.0, 0.0) == pytest.approx(1.0)
    assert combine(0.0, 1.0) == pytest.approx(0.0)


@pytest.fixture(scope="module")
def lr_scores():
    try:
        from tennis.dataset import get_train_test
        from tennis.evaluate import score_matches
        from tennis.models import build_model, get_xy
        train, test = get_train_test()
    except Exception as e:  # data not built on this machine
        pytest.skip(f"dataset unavailable: {e}")
    Xtr, ytr = get_xy(train)
    Xte, _ = get_xy(test)
    ids = train["match_id"].unique()
    keep = np.random.default_rng(0).choice(ids, min(30000, len(ids)), replace=False)
    mask = train["match_id"].isin(keep).to_numpy()      # whole matches, both rows
    model = build_model("logreg", Xtr).fit(Xtr[mask], ytr[mask])
    return score_matches(model, test, Xte)


def test_logreg_p_ab_plus_p_ba_is_about_one(lr_scores):
    """p(A,B) + p(B,A) ~ 1 for a model trained on mirrored pairs."""
    dev = np.abs(lr_scores["p_ab"] + lr_scores["p_ba"] - 1)
    assert dev.mean() < 0.02, f"mean |p_ab + p_ba - 1| = {dev.mean():.4f}"
    assert dev.max() < 0.15


def test_combined_probability_is_valid(lr_scores):
    assert lr_scores["p_win"].between(0, 1).all()
