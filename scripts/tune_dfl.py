"""Tune tracker + decoder parameters on degraded DFL against the official-definition target.

    uv run python scripts/tune_dfl.py

Honest protocol: parameters are chosen on TUNE matches only and reported on HOLDOUT matches the
search never saw. Dead-ball probabilities come from the player-only model (the conservative one),
leave-one-match-out, and are cached so the search only re-runs the cheap parts.
"""

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from mtah.deadball import PLAYER_ONLY, DeadBallModel
from mtah.decode import DecoderConfig
from mtah.evaluate import score_many
from mtah.pipeline import PipelineConfig, predict
from mtah.relabel import relabel_official
from mtah.schema import Match
from mtah.tracker import TrackerConfig

NPZ = Path("../external_data/dfl/npz")
CACHE = Path("results/cache")
TUNE = ["DFL-J03WMX", "DFL-J03WOH", "DFL-J03WPY", "DFL-J03WR9"]
HOLDOUT = ["DFL-J03WN1", "DFL-J03WOY", "DFL-J03WQQ"]

# search space: name -> candidate values (first value = current default)
SPACE = {
    "touch_radius": [1.0, 0.75, 1.5],
    "contest_margin": [0.7, 0.3, 1.2],
    "t_control": [1.0, 0.6, 1.5],
    "n_players": [3, 2, 4],
    "distinct_m": [5.0, 3.0, 8.0],
    "team_conf": [0.9, 0.7, 0.97],
    "alive_threshold": [0.6, 0.5, 0.7],
    "pen_team": [25.0, 40.0, 60.0, 100.0],
    "pen_dead": [16.0, 25.0, 40.0, 60.0],
    "dur_team": [50, 75, 100, 150],
    "dur_dead": [125, 175, 250, 375],
}
# Round 1 (defaults pen 6/4, dur 25/50, t_control 2, touch 1.5, alive 0.5) -> results/tune_report_round1.json;
# this space starts from its best point and extends the four parameters that hit the grid edge.


def make_cfg(p: dict) -> PipelineConfig:
    return PipelineConfig(
        tracker=TrackerConfig(touch_radius=p["touch_radius"], contest_margin=p["contest_margin"],
                              t_control=p["t_control"], n_players=p["n_players"],
                              distinct_m=p["distinct_m"]),
        decoder=DecoderConfig(
            penalty=np.array([[0, p["pen_team"], p["pen_dead"]],
                              [p["pen_team"], 0, p["pen_dead"]],
                              [p["pen_dead"], p["pen_dead"], 0]], float),
            min_dur=(p["dur_team"], p["dur_team"], p["dur_dead"])),
        team_conf=p["team_conf"], alive_threshold=p["alive_threshold"])


def load_all():
    CACHE.mkdir(parents=True, exist_ok=True)
    names = TUNE + HOLDOUT
    clean = {n: Match.load(NPZ / f"{n}.npz") for n in names}
    truth = {n: relabel_official(m) for n, m in clean.items()}
    deg = {n: replace(Match.load(NPZ / f"{n}-degraded.npz"), labels=truth[n]) for n in names}
    pfile = CACHE / "p_dead_players_degraded_official.npz"
    if pfile.exists():
        p_dead = dict(np.load(pfile))
    else:
        p_dead = {}
        for n in names:
            model = DeadBallModel(prefixes=PLAYER_ONLY).fit([deg[k] for k in names if k != n])
            p_dead[n] = model.predict_proba(deg[n])
            print(f"  p_dead {n}", flush=True)
        np.savez_compressed(pfile, **p_dead)
    return deg, truth, p_dead


_STATE = {}


def _init():
    _STATE["deg"], _STATE["truth"], _STATE["p"] = load_all()


def _predict_one(args):
    name, params = args
    m = _STATE["deg"][name]
    return predict(m, _STATE["p"][name], make_cfg(params))


def evaluate(pool, params: dict, names: list[str]) -> dict:
    preds = list(pool.map(_predict_one, [(n, params) for n in names]))
    df = score_many([_TRUTH[n] for n in names], preds, names)
    r = df[df.match == "POOLED"].iloc[0]
    return {"spellF1": float(r.spell_f1_mean), "macroF1": float(r.macro_f1),
            "spells": int(r.n_pred_spells), "gt": int(r.n_gt_spells)}


_TRUTH = {}


def main():
    print("loading / caching dead-ball probabilities ...", flush=True)
    _, truth, _ = load_all()
    _TRUTH.update(truth)
    best = {k: v[0] for k, v in SPACE.items()}
    log = []
    with ProcessPoolExecutor(max_workers=min(7, os.cpu_count() or 1), initializer=_init) as pool:
        base = evaluate(pool, best, TUNE)
        print(f"start  tune spellF1 {base['spellF1']:.4f}  spells {base['spells']}/{base['gt']}")
        best_score = base["spellF1"]
        for rnd in range(1):
            for key, values in SPACE.items():
                for v in values:
                    if v == best[key]:
                        continue
                    trial = {**best, key: v}
                    r = evaluate(pool, trial, TUNE)
                    log.append({"round": rnd, "key": key, "value": v, **r})
                    mark = ""
                    if r["spellF1"] > best_score + 1e-4:
                        best, best_score, mark = trial, r["spellF1"], "  <- best"
                    print(f"r{rnd} {key:15s} = {v!s:6s} tune spellF1 {r['spellF1']:.4f}  "
                          f"spells {r['spells']}/{r['gt']}{mark}", flush=True)
        default = {k: v[0] for k, v in SPACE.items()}
        report = {
            "best_params": best,
            "tune": {"default": evaluate(pool, default, TUNE), "best": evaluate(pool, best, TUNE)},
            "holdout": {"default": evaluate(pool, default, HOLDOUT),
                        "best": evaluate(pool, best, HOLDOUT)},
            "all7": {"default": evaluate(pool, default, TUNE + HOLDOUT),
                     "best": evaluate(pool, best, TUNE + HOLDOUT)},
        }
    Path("results").mkdir(exist_ok=True)
    Path("results/tune_report.json").write_text(json.dumps({**report, "log": log}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
