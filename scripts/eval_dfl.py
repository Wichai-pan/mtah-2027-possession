"""Score methods on the DFL validation set with the official metric.

    uv run python scripts/eval_dfl.py                       # every method, both targets, both versions
    uv run python scripts/eval_dfl.py --methods baseline_smooth50 pipeline --versions degraded

Targets (what counts as the truth):
  raw       DFL's own BallStatus / BallPossession
  official  the same, rewritten by the official possession definition (mtah.relabel, T=2 s)
Versions (what the method sees): clean DFL tracking, or degraded to look like the official data.

Methods that learn (the dead-ball model) are trained leave-one-match-out: each match is predicted
by a model fitted on the other six.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from mtah.baseline import baseline_predict
from mtah.deadball import PLAYER_ONLY, DeadBallModel
from mtah.evaluate import score_many
from mtah.pipeline import PipelineConfig, alive_mask, predict
from mtah.possession_model import PossessionModel
from mtah.relabel import relabel_official
from mtah.schema import D, Match
from mtah.tracker import track


def lomo_p_dead(matches, prefixes=None):
    out = []
    for i, m in enumerate(matches):
        model = DeadBallModel(prefixes=prefixes).fit([x for j, x in enumerate(matches) if j != i])
        out.append(model.predict_proba(m))
    return out


def lomo_p_home(matches, p_dead):
    alive = [alive_mask(m, p) for m, p in zip(matches, p_dead)]
    out = []
    for i, m in enumerate(matches):
        tr = [j for j in range(len(matches)) if j != i]
        model = PossessionModel().fit([matches[j] for j in tr], [alive[j] for j in tr])
        out.append(model.predict_home(m, alive[i]))
    return out


METHODS = {
    # name: (p_dead kind or None, fn(match, p_dead, truth) -> pred)
    # p_dead kinds: "all" features, or "players" only (conservative: ignores DFL's frozen ball)
    "baseline_smooth50": (None, lambda m, p, y: baseline_predict(m, 50)),
    "tracker_oracleD": (None, lambda m, p, y: track(m, (y != D) & (m.half > 0))),  # diagnostic
    "deadball_baselineTeam": ("all", lambda m, p, y: np.where(
        (p >= 0.5) & (m.half > 0), D, baseline_predict(m, 50)).astype(np.int8)),
    "pipeline": ("all", lambda m, p, y: predict(m, p, PipelineConfig())),
    "pipeline_playersD": ("players", lambda m, p, y: predict(m, p, PipelineConfig())),
    # v2: learned possession model; p is (p_dead, p_home)
    "pipeline_v2": ("players+home", lambda m, p, y: predict(m, p[0], p_home=p[1])),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../external_data/dfl/npz")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--versions", nargs="*", default=["clean", "degraded"])
    ap.add_argument("--targets", nargs="*", default=["raw", "official"])
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--out", default="results/dfl_scores.csv")
    args = ap.parse_args()

    clean = [Match.load(p) for p in sorted(Path(args.npz).glob("DFL-J03W??.npz"))]
    official = [relabel_official(m) for m in clean]
    truth = {"raw": [m.labels for m in clean], "official": official}

    rows = []
    for version in args.versions:
        if version == "clean":
            matches = clean
        else:
            matches = [Match.load(Path(args.npz) / f"{m.name}-degraded.npz") for m in clean]
        for target in args.targets:
            ys = truth[target]
            # the dead-ball model is trained on whichever target is being scored
            trained = [replace(m, labels=y) for m, y in zip(matches, ys)]
            p_dead = {}
            kinds = {METHODS[n][0] for n in args.methods} - {None}
            if "players+home" in kinds:
                kinds.add("players")
            for kind in sorted(kinds - {"players+home"}):
                p_dead[kind] = lomo_p_dead(trained, None if kind == "all" else PLAYER_ONLY)
            if "players+home" in kinds:
                p_home = lomo_p_home(trained, p_dead["players"])
                p_dead["players+home"] = list(zip(p_dead["players"], p_home))
            for name in args.methods:
                kind, fn = METHODS[name]
                preds = [fn(m, p_dead[kind][i] if kind else None, ys[i])
                         for i, m in enumerate(matches)]
                df = score_many(ys, preds, [m.name for m in matches])
                df.insert(0, "method", name)
                df.insert(0, "target", target)
                df.insert(0, "version", version)
                rows.append(df)
                r = df[df.match == "POOLED"].iloc[0]
                print(f"{version:9s} {target:9s} {name:22s} spellF1 {r.spell_f1_mean:.4f}  "
                      f"macroF1 {r.macro_f1:.4f}  f1_D {r.f1_D:.3f}  "
                      f"spells {int(r.n_pred_spells)}/{int(r.n_gt_spells)}", flush=True)

    table = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    if args.detail:
        with pd.option_context("display.width", 220, "display.max_rows", 500,
                               "display.float_format", "{:.4f}".format):
            print(table.to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
