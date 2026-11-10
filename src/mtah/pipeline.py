"""Pipeline: dead-ball probability + possession evidence -> constrained Viterbi.

    P(D)        from DeadBallModel (trained on other matches)
    P(H | alive)
        v2 (default when p_home is given): PossessionModel, a learned classifier over ID-free
            features that include the rule tracker's verdict
        v1 (fallback): the rule tracker on the provisional alive mask, softened to `team_conf`
    decode      3-state Viterbi with switch penalties and minimum durations
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decode import DecoderConfig, decode_match, probs_to_logprob
from .schema import A, H, Match
from .tracker import TrackerConfig, track


@dataclass
class PipelineConfig:
    """Defaults tuned on degraded DFL (official-definition target) by scripts/tune_dfl.py:
    chosen on 4 matches, held-out 3 matches 0.371 -> 0.416 spellF1. See results/tune_report*.json."""
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(
        touch_radius=1.0, contest_margin=0.7, t_control=1.0, n_players=3, distinct_m=5.0))
    decoder: DecoderConfig = field(default_factory=lambda: DecoderConfig(
        penalty=np.array([[0, 25, 25], [25, 0, 25], [25, 25, 0]], float), min_dur=(50, 50, 125)))
    team_conf: float = 0.9
    alive_threshold: float = 0.7


# v2: learned possession model. Decoder picked on the 4 tune matches; a further search over
# penalties / durations / temperature moved held-out by < 0.01 (noise), so the simpler point stays.
V2_DECODER = DecoderConfig(
    penalty=np.array([[0, 12, 25], [12, 0, 25], [25, 25, 0]], float), min_dur=(50, 50, 125))


def alive_mask(m: Match, p_dead: np.ndarray, cfg: PipelineConfig | None = None) -> np.ndarray:
    cfg = cfg or PipelineConfig()
    return (p_dead < cfg.alive_threshold) & (m.half > 0)


def predict(m: Match, p_dead: np.ndarray, cfg: PipelineConfig | None = None,
            p_home: np.ndarray | None = None) -> np.ndarray:
    """p_home given -> v2 (learned possession, V2_DECODER unless cfg is passed); else v1."""
    if p_home is not None:
        dec = cfg.decoder if cfg is not None else V2_DECODER
        p = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], 1)
        return decode_match(probs_to_logprob(p), m.half, dec)
    cfg = cfg or PipelineConfig()
    team = track(m, alive_mask(m, p_dead, cfg), cfg.tracker)
    q_home = np.where(team == H, cfg.team_conf, np.where(team == A, 1 - cfg.team_conf, 0.5))
    p = np.stack([(1 - p_dead) * q_home, (1 - p_dead) * (1 - q_home), p_dead], 1)
    return decode_match(probs_to_logprob(p), m.half, cfg.decoder)
