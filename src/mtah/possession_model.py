"""Learned P(home in possession | ball alive) from ID-free tracking features.

Why: the rule tracker needs to *see* touches, and short possessions (median ~4 s) are mostly played
while the ball is unseen -- the tracker has no evidence there (diagnosis in the README). This model
uses softer, longer-range cues and lets the decoder do the segmenting:

  per team k (home and away, built symmetrically):
    - distance of k's nearest player to the ball, now and as rolling past / future minima
    - time since / until k was last / next near the ball (2 m), and since / until a clear touch
    - k players within 5 m / 10 m of the ball
    - where the ball is relative to k's own goal; k's centroid advance and its velocity
  global: ball seen (rolling), the rule tracker's label and how long it has held

Training uses other matches only (DFL) and doubles the data by swapping home and away, so the model
cannot learn "home usually has the ball".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .schema import A, AWAY, D, H, HOME, Match
from .tracker import TrackerConfig, nearest_by_team, touch_team, track

WIN_S = (0.4, 1, 2, 4, 8)
FAR = 60.0


def _roll(x, w, how, future=False):
    s = pd.Series(x[::-1] if future else x)
    r = s.rolling(w, min_periods=1)
    v = (r.min() if how == "min" else r.mean()).to_numpy()
    return v[::-1] if future else v


def _since(flag: np.ndarray, cap: int) -> np.ndarray:
    idx = np.where(flag, np.arange(len(flag)), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(flag)) - last, cap).astype(np.float32)


def _attack_sign(m: Match) -> np.ndarray:
    """(T, 2): +1 if team k attacks towards x=1 in that frame's half, -1 otherwise."""
    out = np.ones((m.n_frames, 2), np.float32)
    x = m.player_xy[:, :, 0]
    for h in (1, 2):
        sel = m.half == h
        if not sel.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cx = [np.nanmean(x[sel][:, m.slot_team == k]) for k in (HOME, AWAY)]
        # a team's average position is on its own (defended) half
        for k in (HOME, AWAY):
            out[sel, k] = 1.0 if cx[k] < cx[1 - k] else -1.0
    return out


def team_features(m: Match, alive: np.ndarray, tcfg: TrackerConfig) -> pd.DataFrame:
    fps = m.fps
    pitch = np.asarray(m.pitch, np.float32)
    near, d_all = nearest_by_team(m)                          # (T,2) metres, inf if unknown
    seen = np.isfinite(near).all(1)
    dist = np.where(np.isfinite(near), near, FAR)
    touch = touch_team(m, tcfg)
    sign = _attack_sign(m)
    bx = np.nan_to_num(m.ball_xy[:, 0], nan=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cent = np.stack([np.nanmean(m.player_xy[:, m.slot_team == k, 0], 1) for k in (HOME, AWAY)], 1)
    cent = np.nan_to_num(cent, nan=0.5)

    f = {"ball_seen": seen.astype(np.float32)}
    for ws in WIN_S:
        f[f"seen_{ws}"] = _roll(seen.astype(np.float32), max(1, int(ws * fps)), "mean")
    trk = track(m, alive, tcfg)
    f["trk_home"] = np.where(trk == H, 1.0, np.where(trk == A, 0.0, 0.5))
    change = np.r_[True, trk[1:] != trk[:-1]]
    f["trk_age"] = _since(change, 30 * fps)
    f["trk_left"] = _since(change[::-1], 30 * fps)[::-1]

    for k, tag in ((HOME, "h"), (AWAY, "a")):
        dk = dist[:, k]
        f[f"d_{tag}"] = dk
        f[f"d_rel_{tag}"] = dk - dist[:, 1 - k]
        for ws in WIN_S:
            w = max(1, int(ws * fps))
            f[f"dmin_past_{ws}_{tag}"] = _roll(dk, w, "min")
            f[f"dmin_fut_{ws}_{tag}"] = _roll(dk, w, "min", future=True)
        nearf = dk < 2.0
        f[f"since_near_{tag}"] = _since(nearf, 15 * fps)
        f[f"until_near_{tag}"] = _since(nearf[::-1], 15 * fps)[::-1]
        tk = touch == k
        f[f"since_touch_{tag}"] = _since(tk, 15 * fps)
        f[f"until_touch_{tag}"] = _since(tk[::-1], 15 * fps)[::-1]
        cols = m.slot_team == k
        dk_all = np.where(np.isfinite(d_all[:, cols]), d_all[:, cols], FAR)
        f[f"n5_{tag}"] = (dk_all < 5).sum(1).astype(np.float32)
        f[f"n10_{tag}"] = (dk_all < 10).sum(1).astype(np.float32)
        adv_ball = (bx - 0.5) * sign[:, k] * pitch[0]          # metres past halfway, k's view
        adv_cent = (cent[:, k] - 0.5) * sign[:, k] * pitch[0]
        f[f"ball_adv_{tag}"] = np.where(seen, adv_ball, np.nan)
        for ws in (2, 8):
            w = int(ws * fps)
            f[f"cent_adv_{ws}_{tag}"] = _roll(adv_cent, w, "mean")
            vel = np.r_[np.zeros(w), adv_cent[w:] - adv_cent[:-w]] / ws
            f[f"cent_vel_{ws}_{tag}"] = vel
    return pd.DataFrame(f)


def swap_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Same frames seen with home and away exchanged (for augmentation)."""
    ren = {}
    for c in df.columns:
        if c.endswith("_h"):
            ren[c] = c[:-2] + "_a"
        elif c.endswith("_a"):
            ren[c] = c[:-2] + "_h"
    out = df.rename(columns=ren)[df.columns].copy()
    out["trk_home"] = 1.0 - df["trk_home"]
    return out


class PossessionModel:
    def __init__(self, tracker: TrackerConfig | None = None, seed: int = 0, subsample: int = 5):
        self.tcfg = tracker or TrackerConfig(touch_radius=1.0, t_control=1.0)
        self.subsample = subsample
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                  max_leaf_nodes=31, random_state=seed)

    def fit(self, matches: list[Match], alive_masks: list[np.ndarray] | None = None
            ) -> "PossessionModel":
        """alive_masks: the *predicted* alive masks the features will see at test time (so the
        tracker feature is built the same way); labels are taken on truly alive frames only."""
        Xs, ys = [], []
        for i, m in enumerate(matches):
            true_alive = (m.labels != D) & (m.half > 0)
            alive = alive_masks[i] if alive_masks is not None else true_alive
            g = np.flatnonzero(true_alive)[:: self.subsample]
            F = team_features(m, alive, self.tcfg).iloc[g]
            y = (m.labels[g] == H).astype(int)
            Xs += [F, swap_teams(F)]
            ys += [y, 1 - y]
        self.clf.fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys))
        return self

    def predict_home(self, m: Match, alive: np.ndarray) -> np.ndarray:
        """P(home | alive) for every frame (0.5 outside the halves)."""
        p = np.full(m.n_frames, 0.5)
        g = m.half > 0
        F = team_features(m, alive, self.tcfg)
        p[g] = self.clf.predict_proba(F[g])[:, 1]
        return p
