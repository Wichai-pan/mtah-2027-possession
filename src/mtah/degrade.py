"""Make clean (DFL / TRACAB) tracking look like the official automatic tracking.

What the official data has that DFL does not (DATA.md):
  - ball missing on ~41% of graded frames, in bursts, more often when the ball is in the air,
    fast, or in a crowd
  - 46 slots for 22 players: tracks fragment into tracklets that land in different slots
  - ~20.5 players visible on average (some drop out for a while)
  - positions are smooth: the real high-frequency residual (position minus its 1 s moving
    average) is 0.055 m, below DFL's own 0.157 m, so no extra white noise is added

Calibrated on the official BAR-ATH tracking (2026-11-06, graded frames): ball missing 40.7%,
20.5 players seen per frame (DFL clean 21.6 -> 5% drop), slot runs median 35 s / mean 107 s
(lognormal, sigma 1.5).
Everything is driven by a seeded RNG so a degraded copy is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .schema import AWAY, HOME, Match


@dataclass
class DegradeConfig:
    seed: int = 0
    # ball dropout: alternating renewal process (seen <-> missing), in frames
    ball_missing_target: float = 0.41     # fraction of graded frames with no ball
    ball_gap_mean: float = 30.0           # mean missing burst (1.2 s)
    ball_air_boost: float = 3.0           # gap-start hazard multiplier when ball_z > 1.5 m
    ball_fast_boost: float = 2.0          # ... when ball speed > 15 m/s
    # player tracks
    tracklet_median_s: float = 35.0       # lognormal tracklet length before a fragment/ID change
    tracklet_sigma: float = 1.5           # -> mean ~107 s, as measured on the official data
    slots_per_team: int = 23              # 46 slots in total, like the official data
    player_drop_rate: float = 0.05        # fraction of player-frames missing
    player_gap_mean: float = 50.0
    jitter_m: float = 0.0                 # gaussian position noise, metres (real data is smooth)


def _bursty_mask(T: int, target: float, gap_mean: float, hazard_mult: np.ndarray,
                 rng: np.random.Generator, weight: np.ndarray | None = None) -> np.ndarray:
    """True where missing. Gap lengths ~ geometric(mean gap_mean); start hazard scaled per frame.

    The overall hazard is calibrated by bisection so the missing fraction (over `weight` frames,
    default all) hits `target`; the random draws are fixed first, so the result is reproducible.
    """
    u = rng.random(T)
    lengths = rng.geometric(1.0 / gap_mean, size=T)
    rel = hazard_mult / max(float(hazard_mult.mean()), 1e-9)
    weight = np.ones(T, bool) if weight is None else weight

    def mask_for(base):
        starts = np.flatnonzero(u < np.clip(base * rel, 0, 1))
        acc = np.zeros(T + 1, np.int32)
        np.add.at(acc, starts, 1)
        np.add.at(acc, np.minimum(starts + lengths[starts], T), -1)
        return np.cumsum(acc[:T]) > 0

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if mask_for(mid)[weight].mean() < target:
            lo = mid
        else:
            hi = mid
    return mask_for((lo + hi) / 2)


def degrade(m: Match, cfg: DegradeConfig | None = None) -> Match:
    cfg = cfg or DegradeConfig()
    rng = np.random.default_rng(cfg.seed)
    T, S = m.n_frames, m.n_slots
    pitch = np.asarray(m.pitch, np.float32)

    # --- ball -----------------------------------------------------------------------------------
    ball_xy = m.ball_xy.copy()
    mult = np.ones(T)
    if m.ball_z is not None:
        mult *= np.where(np.nan_to_num(m.ball_z) > 1.5, cfg.ball_air_boost, 1.0)
    v = np.zeros(T)
    v[1:] = np.linalg.norm(np.diff(np.nan_to_num(ball_xy) * pitch, axis=0), axis=1) * m.fps
    mult *= np.where(v > 15, cfg.ball_fast_boost, 1.0)
    # the mask is calibrated on graded frames only
    graded = m.half > 0
    miss = _bursty_mask(T, cfg.ball_missing_target, cfg.ball_gap_mean, mult, rng, graded)
    ball_xy[miss] = np.nan
    if cfg.jitter_m > 0:
        ball_xy += rng.normal(0, cfg.jitter_m, ball_xy.shape).astype(np.float32) / pitch

    # --- players: fragment into tracklets and scatter over slots ---------------------------------
    n_out = 2 * cfg.slots_per_team
    out_team = np.array([HOME] * cfg.slots_per_team + [AWAY] * cfg.slots_per_team, np.int8)
    pxy = np.full((T, n_out, 2), np.nan, np.float32)
    psp = np.full((T, n_out), np.nan, np.float32)
    free_at = np.zeros(n_out, np.int64)  # frame from which a slot is free again
    med_len = cfg.tracklet_median_s * m.fps

    segs = []   # (start, player slot, frames)
    for s in range(S):
        present = ~np.isnan(m.player_xy[:, s, 0])
        drop = _bursty_mask(T, cfg.player_drop_rate, cfg.player_gap_mean, np.ones(T), rng)
        present &= ~drop
        idx = np.flatnonzero(present)
        if not len(idx):
            continue
        # contiguous runs of presence, each further cut into lognormal-length tracklets
        breaks = np.flatnonzero(np.diff(idx) > 1) + 1
        for run in np.split(idx, breaks):
            pos = 0
            while pos < len(run):
                L = max(1, int(med_len * rng.lognormal(0.0, cfg.tracklet_sigma)))
                segs.append((int(run[pos]), s, run[pos:pos + L]))
                pos += L
    # assign in time order, so a slot is only reused once its previous tracklet has ended
    for _, s, seg in sorted(segs, key=lambda x: (x[0], x[1])):
        team = m.slot_team[s]
        cands = np.flatnonzero((out_team == team) & (free_at <= seg[0]))
        if not len(cands):   # more simultaneous tracklets than slots: overwrite the stalest
            cands = np.flatnonzero(out_team == team)
            cands = cands[[np.argmin(free_at[cands])]]
        k = int(rng.choice(cands))
        free_at[k] = seg[-1] + 1
        pxy[seg, k] = m.player_xy[seg, s]
        psp[seg, k] = m.player_speed[seg, s]

    if cfg.jitter_m > 0:
        pxy += rng.normal(0, cfg.jitter_m, pxy.shape).astype(np.float32) / pitch

    return replace(m, name=m.name + "-degraded", ball_xy=ball_xy, player_xy=pxy,
                   player_speed=psp, slot_team=out_team, ball_z=None,
                   meta={**m.meta, "degrade": repr(cfg),
                         "ball_missing_graded": float(np.isnan(ball_xy[graded, 0]).mean())})
