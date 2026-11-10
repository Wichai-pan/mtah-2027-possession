"""Dead-ball detector: per-frame P(D) from ID-free tracking features + a gradient-boosted classifier.

Features are permutation-invariant (no player identities) and multi-scale in time, in the spirit
of Lang et al. 2022 (in-play status from player positions), extended with ball cues:
  - player speed: mean, low-speed fraction, at several centred windows, and the future-minus-past
    change (restarts accelerate, stoppages decelerate)
  - ball: seen fraction, time since last / until next sighting, speed, distance outside the pitch
  - layout: players inside either penalty box, team spreads

Trained on other matches only (DFL, leave-one-match-out for validation) -- never on the target.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .schema import D, Match

WINDOWS_S = (0.4, 1, 2, 4, 8, 16)


def _roll(x: np.ndarray, w: int, how: str = "mean") -> np.ndarray:
    s = pd.Series(x)
    r = s.rolling(w, center=True, min_periods=1)
    return (r.mean() if how == "mean" else r.max() if how == "max" else r.min()).to_numpy()


def _shift_mean(x: np.ndarray, w: int, future: bool) -> np.ndarray:
    s = pd.Series(x)
    if future:
        return s[::-1].rolling(w, min_periods=1).mean()[::-1].to_numpy()
    return s.rolling(w, min_periods=1).mean().to_numpy()


def _since(flag: np.ndarray, cap: int) -> np.ndarray:
    """Frames since flag was last True (capped)."""
    out = np.full(len(flag), cap, np.float32)
    last = -cap
    for i, f in enumerate(flag):
        if f:
            last = i
        out[i] = min(i - last, cap)
    return out


def features(m: Match) -> pd.DataFrame:
    fps = m.fps
    pitch = np.asarray(m.pitch, np.float32)
    spd = m.player_speed
    seen_p = ~np.isnan(spd)
    n_seen = seen_p.sum(1).astype(np.float32)
    mean_spd = np.where(n_seen > 0, np.nansum(spd, 1) / np.maximum(n_seen, 1), 0)
    slow = np.where(n_seen > 0, np.nansum(spd < 1.0, 1) / np.maximum(n_seen, 1), 1)
    fast = np.where(n_seen > 0, np.nansum(spd > 4.0, 1) / np.maximum(n_seen, 1), 0)

    ball = m.ball_xy * pitch
    bseen = ~np.isnan(ball[:, 0])
    bv = np.full(m.n_frames, np.nan, np.float32)
    ok = bseen[1:] & bseen[:-1]
    bv[1:][ok] = np.linalg.norm(ball[1:][ok] - ball[:-1][ok], axis=1) * fps
    bv = np.clip(bv, 0, 40)
    out_x = np.maximum(-ball[:, 0], ball[:, 0] - pitch[0])
    out_y = np.maximum(-ball[:, 1], ball[:, 1] - pitch[1])
    out_dist = np.nan_to_num(np.maximum(out_x, out_y), nan=-5.0)
    edge = np.nan_to_num(np.minimum.reduce([ball[:, 0], pitch[0] - ball[:, 0],
                                            ball[:, 1], pitch[1] - ball[:, 1]]), nan=10.0)

    pxy = m.player_xy * pitch
    in_box = ((np.abs(pxy[:, :, 1] - pitch[1] / 2) < 20.16) &
              ((pxy[:, :, 0] < 16.5) | (pxy[:, :, 0] > pitch[0] - 16.5))).sum(1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spread = np.nan_to_num(np.nanstd(pxy[:, :, 0], 1)) + np.nan_to_num(np.nanstd(pxy[:, :, 1], 1))

    f = {"n_seen": n_seen, "ball_seen": bseen.astype(np.float32),
         "since_ball": _since(bseen, 25 * fps), "until_ball": _since(bseen[::-1], 25 * fps)[::-1],
         "out_dist": out_dist, "edge": edge, "in_box": in_box.astype(np.float32), "spread": spread}
    for ws in WINDOWS_S:
        w = max(1, int(ws * fps))
        f[f"spd_{ws}"] = _roll(mean_spd, w)
        f[f"slow_{ws}"] = _roll(slow, w)
        f[f"fast_{ws}"] = _roll(fast, w)
        f[f"bseen_{ws}"] = _roll(bseen.astype(np.float32), w)
        f[f"bv_{ws}"] = _roll(np.nan_to_num(bv, nan=-1), w)
        f[f"out_{ws}"] = _roll(out_dist, w, "max")
        f[f"edge_{ws}"] = _roll(edge, w, "min")
        f[f"spd_delta_{ws}"] = _shift_mean(mean_spd, w, True) - _shift_mean(mean_spd, w, False)
    return pd.DataFrame(f)


PLAYER_ONLY = ("spd", "slow", "fast", "n_seen", "in_box", "spread")


def select(df: pd.DataFrame, prefixes: tuple[str, ...] | None) -> pd.DataFrame:
    return df if prefixes is None else df[[c for c in df.columns if c.startswith(prefixes)]]


class DeadBallModel:
    """prefixes: restrict to feature groups, e.g. PLAYER_ONLY. On DFL the ball is frozen
    (speed 0, z 0) during dead time -- a provider artefact the official data will not share --
    so PLAYER_ONLY gives the conservative estimate."""

    def __init__(self, seed: int = 0, subsample: int = 5, prefixes: tuple[str, ...] | None = None):
        self.prefixes = prefixes
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                  max_leaf_nodes=31, random_state=seed)
        self.subsample = subsample  # frames are highly correlated; every 5th is plenty

    def fit(self, matches: list[Match]) -> "DeadBallModel":
        Xs, ys = [], []
        for m in matches:
            g = np.flatnonzero(m.half > 0)[:: self.subsample]
            Xs.append(select(features(m), self.prefixes).iloc[g])
            ys.append((m.labels[g] == D).astype(int))
        self.clf.fit(pd.concat(Xs), np.concatenate(ys))
        return self

    def predict_proba(self, m: Match) -> np.ndarray:
        p = np.zeros(m.n_frames)
        g = m.half > 0
        p[g] = self.clf.predict_proba(select(features(m), self.prefixes)[g])[:, 1]
        return p
