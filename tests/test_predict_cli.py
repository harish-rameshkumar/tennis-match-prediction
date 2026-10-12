"""Command-line demo: python -m tennis.predict --match A B --surface S --date D."""

import re
import subprocess
import sys

import pytest

from tennis import predict

pytestmark = pytest.mark.skipif(not predict._path(predict.DEFAULT_MODEL).exists(),
                                reason="train first: python -m tennis.predict --train")

MATCH = ["--match", "Novak Djokovic", "Rafael Nadal", "--surface", "Clay", "--date", "2025-06-01"]


def _percentages(text: str) -> list[float]:
    return [float(p) for p in re.findall(r"^\s+.+?\s+(\d+\.\d)%$", text, flags=re.M)]


def test_cli_prints_both_probabilities_and_top_features(capsys):
    assert predict.main(MATCH) == 0
    out = capsys.readouterr().out
    assert "Novak Djokovic vs Rafael Nadal" in out
    p = _percentages(out)
    assert len(p) == 2 and abs(sum(p) - 100) <= 0.1
    assert "Top contributing features" in out
    assert re.search(r"^\s+[+-]\d\.\d{3}\s+\S+\s+favours ", out, flags=re.M)


def test_cli_matches_predict_match(capsys):
    predict.main(MATCH)
    p_a, p_b = _percentages(capsys.readouterr().out)
    r = predict.predict_match("Novak Djokovic", "Rafael Nadal", "Clay", "2025-06-01")
    assert (p_a, p_b) == (round(r["p_a"] * 100, 1), round(r["p_b"] * 100, 1))


def test_cli_unknown_player_suggests_names(capsys):
    assert predict.main(["--match", "Novak Djokovc", "Rafael Nadal", "--date", "2025-06-01"]) == 2
    out = capsys.readouterr().out
    assert out.startswith("error: Unknown player") and "Novak Djokovic" in out


def test_cli_runs_as_module():
    done = subprocess.run([sys.executable, "-m", "tennis.predict", *MATCH],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert len(_percentages(done.stdout)) == 2
