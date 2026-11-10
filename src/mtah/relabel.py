"""Apply the official possession definition (docs/possession_rules.md) to DFL labels.

DFL's BallPossession flips on every touch. The official definition only changes possession on
*controlled* possession (R3) and absorbs touches, deflections and won-but-uncontrolled duels into
the surrounding possession (R4). This module rewrites a DFL label track accordingly, producing an
official-definition validation target. It needs clean DFL tracking (persistent player slots), so it
is run on the clean match and the result reused for the degraded copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .official import possession_metrics as pm
from .schema import A, D, H, Match


@dataclass
class RelabelConfig:
    t_control: float = 2.0     # R3: seconds a team must keep the ball
    n_players: int = 3         # R3: distinct players touching in sequence (two passes)
    touch_radius: float = 1.5  # metres: a player this close to a low ball is touching it
    touch_max_z: float = 2.0   # metres: higher balls are in the air, not at anyone's feet


def touch_slots(m: Match, cfg: RelabelConfig) -> np.ndarray:
    """(T,) slot of the player touching the ball, -1 for none."""
    pitch = np.asarray(m.pitch, np.float32)
    d = np.sqrt((((m.player_xy - m.ball_xy[:, None, :]) * pitch) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    best = d.argmin(1)
    ok = d.min(1) <= cfg.touch_radius
    if m.ball_z is not None:
        ok &= ~(np.nan_to_num(m.ball_z) > cfg.touch_max_z)
    return np.where(ok, best, -1)


def relabel_official(m: Match, cfg: RelabelConfig | None = None) -> np.ndarray:
    cfg = cfg or RelabelConfig()
    labels = m.labels.copy()
    touch = touch_slots(m, cfg)
    min_len = int(round(cfg.t_control * m.fps))

    for start, stop in pm.valid_runs(labels):
        while True:
            segs = pm.segments(labels, [(start, stop)])
            changed = False
            for i, (a, b, k) in enumerate(segs):
                if k not in (H, A) or i == 0 or segs[i - 1][2] == D:
                    continue  # R7: first possession of a half / after a stoppage is kept
                if b - a >= min_len:
                    continue
                t = touch[a:b]
                t = t[t >= 0]
                team_touchers = {s for s in t.tolist() if m.slot_team[s] == k}
                if len(team_touchers) >= cfg.n_players:
                    continue
                labels[a:b] = segs[i - 1][2]  # R4: absorbed into the previous possession
                changed = True
                break  # segments changed; recompute
            if not changed:
                break
    return labels


def spell_stats(labels: np.ndarray, fps: int = 25) -> dict:
    segs = pm.segments(labels, pm.valid_runs(labels))
    L = np.array([b - a for a, b, k in segs if k in (H, A)]) / fps
    return {"spells": len(L), "mean_s": L.mean(), "median_s": float(np.median(L)),
            "under_1s": float((L < 1).mean()), "under_2s": float((L < 2).mean())}
