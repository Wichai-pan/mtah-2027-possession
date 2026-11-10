"""Snap dead-ball boundaries to instantaneous ball events.

A state classifier places stoppage boundaries to within ~1 s at best; the metric punishes a 1 s
offset hard. Events that mark the boundary to the frame, when the ball is seen:
  stoppage start  the ball crosses a pitch line (inside -> outside)
  stoppage end    the ball is kicked: speed jumps above `kick_speed` after resting below
                  `rest_speed` for `rest_s` seconds (restart from a placed / held ball)
Each D segment's start / end moves to the nearest such event within the snap window; no event in
the window -> the boundary stays where the decoder put it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .official import possession_metrics as pm
from .schema import D, Match


@dataclass
class SnapConfig:
    win_start_s: float = 2.0
    win_end_s: float = 2.0
    out_margin_m: float = 0.3
    kick_speed: float = 6.0     # m/s
    rest_speed: float = 2.0     # m/s
    rest_s: float = 0.5
    snap_start: bool = True
    snap_end: bool = True


def ball_speed(m: Match, span: int = 3) -> np.ndarray:
    """Ball speed (m/s) from a centred difference over 2*span frames, NaN where unknown."""
    b = pd.DataFrame(m.ball_xy * np.asarray(m.pitch, np.float32))
    b = b.rolling(5, center=True, min_periods=3).median().to_numpy()
    v = np.full(m.n_frames, np.nan)
    d = np.linalg.norm(b[2 * span:] - b[:-2 * span], axis=1) * m.fps / (2 * span)
    v[span:-span] = d
    return v


def events(m: Match, cfg: SnapConfig) -> tuple[np.ndarray, np.ndarray]:
    """(out_cross_frames, kick_frames)."""
    pitch = np.asarray(m.pitch, np.float32)
    b = m.ball_xy * pitch
    seen = ~np.isnan(b[:, 0])
    out = np.zeros(m.n_frames, bool)
    out[seen] = ((b[seen, 0] < -cfg.out_margin_m) | (b[seen, 0] > pitch[0] + cfg.out_margin_m) |
                 (b[seen, 1] < -cfg.out_margin_m) | (b[seen, 1] > pitch[1] + cfg.out_margin_m))
    # inside -> outside: last seen frame inside, next seen frame outside
    idx = np.flatnonzero(seen)
    o = out[idx]
    cross = idx[1:][(~o[:-1]) & o[1:]]

    v = ball_speed(m)
    rest = pd.Series(np.nan_to_num(v, nan=99.0) < cfg.rest_speed).rolling(
        max(1, int(cfg.rest_s * m.fps))).min().to_numpy() > 0
    fast = np.nan_to_num(v, nan=0.0) > cfg.kick_speed
    kick = np.flatnonzero(fast[1:] & rest[:-1]) + 1
    return cross, kick


def _nearest(cands: np.ndarray, t: int, win: int) -> int | None:
    if not len(cands):
        return None
    j = np.searchsorted(cands, t)
    best = None
    for k in (j - 1, j):
        if 0 <= k < len(cands) and abs(cands[k] - t) <= win:
            if best is None or abs(cands[k] - t) < abs(best - t):
                best = int(cands[k])
    return best


def snap(m: Match, pred: np.ndarray, cfg: SnapConfig | None = None) -> np.ndarray:
    cfg = cfg or SnapConfig()
    cross, kick = events(m, cfg)
    ws, we = int(cfg.win_start_s * m.fps), int(cfg.win_end_s * m.fps)
    out = pred.copy()
    for a, b in pm.valid_runs(np.where(m.half > 0, 0, 3)):
        segs = pm.segments(pred, [(a, b)])
        for i, (s0, s1, k) in enumerate(segs):
            if k != D:
                continue
            if cfg.snap_start and s0 > a and i > 0:
                t = _nearest(cross, s0, ws)
                if t is not None and t < s1:
                    prev = segs[i - 1][2]
                    if t < s0:
                        out[t:s0] = D
                    else:
                        out[s0:t] = prev
            if cfg.snap_end and s1 < b and i + 1 < len(segs):
                t = _nearest(kick, s1, we)
                if t is not None and t > s0:
                    nxt = segs[i + 1][2]
                    if t > s1:
                        out[s1:t] = D
                    else:
                        out[t:s1] = nxt
    return out
