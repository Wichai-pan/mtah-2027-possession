"""Round trip: Match -> official ATD files (make_fake_bundle) -> load_mtah_match -> same Match."""

import importlib.util
from pathlib import Path

import numpy as np

from mtah import io_mtah
from mtah.schema import Match

SPEC = importlib.util.spec_from_file_location(
    "make_fake_bundle", Path(__file__).parents[1] / "scripts" / "make_fake_bundle.py")
mfb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mfb)


def test_atd_round_trip(tmp_path, monkeypatch):
    T, S = 300, 6
    monkeypatch.setitem(io_mtah.GAMES, "BAR-ATH", {
        "n_frames": T, "fps": 25, "pitch": (105.3, 68.0),
        "halves": {1: (20, 139), 2: (160, 279)}, "stem": "tactical/BAR-ATH_tactical"})
    rng = np.random.default_rng(0)
    half = np.zeros(T, np.int8)
    half[20:140], half[160:280] = 1, 2
    pxy = rng.random((T, S, 2)).astype(np.float32)
    pxy[rng.random((T, S)) < 0.3] = np.nan
    ball = rng.random((T, 2)).astype(np.float32)
    ball[rng.random(T) < 0.4] = np.nan
    spd = np.where(np.isnan(pxy[..., 0]), np.nan, rng.random((T, S)) * 8).astype(np.float32)
    m = Match(name="t", fps=25, pitch=(105.3, 68.0), half=half, ball_xy=ball, player_xy=pxy,
              player_speed=spd, slot_team=np.array([0, 1, 0, 1, 1, 0], np.int8))
    mfb.write_atd(m, tmp_path / "tactical")

    got = io_mtah.load_mtah_match(tmp_path)
    assert np.array_equal(got.half, half)
    assert np.array_equal(got.slot_team, m.slot_team)
    for a, b, tol in ((got.ball_xy, ball, 1e-5), (got.player_xy, pxy, 1e-5),
                      (got.player_speed, spd, 6e-3)):
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.nanmax(np.abs(a - b)) <= tol
