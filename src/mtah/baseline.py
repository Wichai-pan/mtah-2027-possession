"""The official reference baseline, re-implemented on `Match` so it runs on any source.

Same two steps as starter/baseline.py: nearest tracked player to the ball (in metres) gives the
team, no ball -> D; then a centred majority vote over +/- `window` frames. Ties go home.
"""

from __future__ import annotations

import numpy as np

from .schema import A, AWAY, D, H, HOME, X, Match

NOBODY = -1


def nearest_team(m: Match) -> np.ndarray:
    """(T,) HOME / AWAY of the closest tracked player to the ball, NOBODY when the ball is unseen."""
    pitch = np.asarray(m.pitch, dtype=np.float32)
    dxy = (m.player_xy - m.ball_xy[:, None, :]) * pitch          # (T, S, 2) metres
    dist = np.sqrt((dxy ** 2).sum(-1))                             # NaN where either is missing
    dist = np.where(np.isnan(dist), np.inf, dist)
    best = dist.argmin(1)
    has = np.isfinite(dist.min(1))
    out = np.where(has, m.slot_team[best], NOBODY).astype(np.int8)
    return out


def majority_smooth(codes: np.ndarray, window: int) -> np.ndarray:
    """Centred majority vote; returns label codes H / A / D."""
    if window <= 0:
        return np.where(codes == HOME, H, np.where(codes == AWAY, A, D)).astype(np.int8)
    hc = np.concatenate(([0], np.cumsum(codes == HOME)))
    ac = np.concatenate(([0], np.cumsum(codes == AWAY)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    home, away = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(len(codes), D, np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out


def baseline_predict(m: Match, window: int = 50) -> np.ndarray:
    """Full-length prediction, X outside the halves (ignored by the scorer anyway)."""
    pred = majority_smooth(nearest_team(m), window)
    pred[m.half == 0] = X
    return pred
