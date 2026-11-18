from pathlib import Path

from retrieval_tuner.tuner import grid_search

ROOT = Path(__file__).resolve().parents[1]


def test_grid_finds_best():
    report = grid_search(ROOT / "samples" / "docs")
    assert report["best"]["hit_rate"] >= 0.5
    assert len(report["grid"]) >= 10
