"""State-machine possession tracker implementing docs/possession_rules.md on ID-free tracking.

Works on team-level data only (slots carry a team, not a person), so it runs unchanged on the
official automatic tracking and on degraded DFL.

1. Touches: on frames where the ball is seen, team k touches if its nearest player is within
   `touch_radius` of the ball and clearly closer than the nearest opponent (else: contested, no touch).
2. Episodes: runs of touches by one team, bridging gaps of up to `bridge` frames.
3. Possession, per alive stretch (R2-R7):
     - the first episode's team owns the restart (R7)
     - an opponent episode opens a *candidate* change; the candidate run is the opponent's
       episodes until the current team touches again
     - the change is confirmed if the run spans >= t_control seconds, or contains >= n_players
       episodes at distinct places (a proxy for distinct players: two passes) (R3)
     - confirmed changes start at the opponent's first touch (R5); unconfirmed runs are absorbed (R4)
     - between touches, possession stays where it was (R2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import A, AWAY, D, H, HOME, Match

TEAM_TO_LABEL = {HOME: H, AWAY: A}


@dataclass
class TrackerConfig:
    touch_radius: float = 1.5     # metres
    contest_margin: float = 0.7   # nearest opponent must be this much further away
    bridge: int = 5               # frames: gaps inside one touch episode
    t_control: float = 2.0        # seconds (R3)
    n_players: int = 3            # distinct touch places (R3)
    distinct_m: float = 5.0       # metres between episodes to count as different players


def nearest_by_team(m: Match) -> tuple[np.ndarray, np.ndarray]:
    """(T, 2) distance in metres from the ball to the nearest home / away player (inf if none)."""
    pitch = np.asarray(m.pitch, np.float32)
    d = np.sqrt((((m.player_xy - m.ball_xy[:, None, :]) * pitch) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    out = np.full((m.n_frames, 2), np.inf, np.float32)
    for team in (HOME, AWAY):
        cols = m.slot_team == team
        if cols.any():
            out[:, team] = d[:, cols].min(1)
    return out, d


def touch_team(m: Match, cfg: TrackerConfig) -> np.ndarray:
    """(T,) HOME / AWAY touching the ball, -1 for no clear touch."""
    near, _ = nearest_by_team(m)
    k = near.argmin(1)
    dk = near.min(1)
    other = near.max(1)
    ok = (dk <= cfg.touch_radius) & (other > dk + cfg.contest_margin)
    return np.where(ok, k, -1).astype(np.int8)


def episodes(touch: np.ndarray, ball_xy: np.ndarray, pitch, bridge: int) -> list[tuple]:
    """[(start, stop, team, x_m, y_m)] runs of one team's touches, bridging short gaps."""
    idx = np.flatnonzero(touch >= 0)
    out = []
    if not len(idx):
        return out
    pitch = np.asarray(pitch)
    s = p = idx[0]
    for i in idx[1:]:
        if touch[i] == touch[s] and i - p <= bridge + 1:
            p = i
            continue
        xy = np.nanmean(ball_xy[s:p + 1], 0) * pitch
        out.append((s, p + 1, int(touch[s]), xy[0], xy[1]))
        s = p = i
    xy = np.nanmean(ball_xy[s:p + 1], 0) * pitch
    out.append((s, p + 1, int(touch[s]), xy[0], xy[1]))
    return out


def _n_distinct(eps: list[tuple], min_dist: float) -> int:
    places: list[tuple[float, float]] = []
    for e in eps:
        if all((e[3] - x) ** 2 + (e[4] - y) ** 2 > min_dist ** 2 for x, y in places):
            places.append((e[3], e[4]))
    return len(places)


def assign_possession(eps: list[tuple], start: int, stop: int, fps: int,
                      cfg: TrackerConfig, fallback: int) -> np.ndarray:
    """Team per frame (HOME/AWAY) for one alive stretch [start, stop)."""
    eps = [e for e in eps if start <= e[0] < stop]
    team = np.full(stop - start, fallback, np.int8)
    if not eps:
        return team
    cur = eps[0][2]                       # R7: restart team
    change_at = [(start, cur)]
    i = 1
    while i < len(eps):
        if eps[i][2] == cur:
            i += 1
            continue
        j = i
        while j < len(eps) and eps[j][2] != cur:
            j += 1
        run = eps[i:j]
        span = (run[-1][1] - run[0][0]) / fps
        if span >= cfg.t_control or _n_distinct(run, cfg.distinct_m) >= cfg.n_players:
            cur = run[0][2]
            change_at.append((run[0][0], cur))   # R5: dated back to the first touch
        i = j
    for (a, k), nxt in zip(change_at, change_at[1:] + [(stop, None)]):
        team[a - start:nxt[0] - start] = k
    return team


def alive_stretches(alive: np.ndarray, half: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for h in (1, 2):
        mask = alive & (half == h)
        edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
        out += list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return out


def track(m: Match, alive: np.ndarray, cfg: TrackerConfig | None = None) -> np.ndarray:
    """Labels H/A/D over the halves (X outside) given an alive mask."""
    cfg = cfg or TrackerConfig()
    touch = touch_team(m, cfg)
    eps = episodes(touch, m.ball_xy, m.pitch, cfg.bridge)
    labels = np.full(m.n_frames, 3, np.int8)
    labels[m.half > 0] = D
    last = HOME
    for a, b in alive_stretches(alive, m.half):
        team = assign_possession(eps, a, b, m.fps, cfg, last)
        labels[a:b] = np.where(team == HOME, H, A)
        last = int(team[-1])
    return labels
