"""Official data bundle -> submission CSV, in one command.

    uv run python scripts/make_submission.py --data-dir ../repo/data --out results/submission.csv

Steps
  1. read the official ATD (mtah.io_mtah) and sanity-check it
  2. train on the degraded DFL matches, official-definition labels:
       DeadBallModel (player-only features) on all of them;
       PossessionModel on all of them, its tracker feature built on leave-one-match-out alive masks
       (so at training time it sees the same kind of imperfect dead-ball mask it gets at test time)
  3. predict P(dead), P(home | alive) for the match, decode (pipeline v2), X outside the halves
  4. write frame,label, run the official validate_submission.py, and score if a GT csv is given

Deterministic: fixed seeds, no wall-clock or ordering dependence -- the same inputs give the same
file byte for byte (the repository deliverable requires that).

Dry run without the official data:
    uv run python scripts/make_fake_bundle.py --match DFL-J03WQQ
    uv run python scripts/make_submission.py --data-dir ../external_data/fake_mtah \
        --exclude DFL-J03WQQ --gt ../external_data/fake_mtah/gt_full.csv --out results/dryrun.csv
"""

import argparse
import hashlib
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from mtah.deadball import PLAYER_ONLY, DeadBallModel
from mtah.evaluate import score
from mtah.io_mtah import load_mtah_match
from mtah.official import possession_metrics as pm
from mtah.pipeline import alive_mask, predict
from mtah.possession_model import PossessionModel
from mtah.relabel import relabel_official
from mtah.schema import Match, write_submission

DFL = ["DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY", "DFL-J03WQQ",
       "DFL-J03WR9"]


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


def check_match(m: Match) -> None:
    graded = m.half > 0
    ball = ~np.isnan(m.ball_xy[graded, 0])
    players = (~np.isnan(m.player_xy[graded, :, 0])).sum(1)
    log(f"match: {m.n_frames} frames, {m.n_slots} slots "
        f"(home {int((m.slot_team == 0).sum())} / away {int((m.slot_team == 1).sum())}), "
        f"{graded.sum()} graded")
    log(f"  graded frames: ball seen {ball.mean():.1%}, mean players seen {players.mean():.1f}")
    for name, ok in [("199319 frames", m.n_frames == 199319),
                     ("two teams of slots", set(np.unique(m.slot_team)) == {0, 1}),
                     ("coordinates normalised", np.nanmax(np.abs(m.player_xy)) < 2),
                     ("ball seen on 30-80% of graded frames", 0.3 < ball.mean() < 0.8),
                     ("15+ players seen on average", players.mean() > 15)]:
        log(f"  {'ok  ' if ok else 'WARN'} {name}")


def load_training(npz: Path, names: list[str]) -> list[Match]:
    out = []
    for n in names:
        truth = relabel_official(Match.load(npz / f"{n}.npz"))
        out.append(replace(Match.load(npz / f"{n}-degraded.npz"), labels=truth))
    return out


def train(train: list[Match]):
    alive = []
    for i, m in enumerate(train):
        others = [x for j, x in enumerate(train) if j != i]
        p = DeadBallModel(prefixes=PLAYER_ONLY).fit(others).predict_proba(m)
        alive.append(alive_mask(m, p))
        log(f"  LOMO dead-ball mask for {m.name}")
    dead = DeadBallModel(prefixes=PLAYER_ONLY).fit(train)
    log("  dead-ball model fitted on all training matches")
    poss = PossessionModel().fit(train, alive)
    log("  possession model fitted")
    return dead, poss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="unpacked official bundle (has tactical/)")
    ap.add_argument("--out", default="results/submission.csv")
    ap.add_argument("--npz", default="../external_data/dfl/npz")
    ap.add_argument("--exclude", nargs="*", default=[], help="DFL matches to keep out of training")
    ap.add_argument("--gt", default=None, help="frame,label csv to score against (optional)")
    ap.add_argument("--repo", default="../repo", help="official repo, for validate_submission.py")
    args = ap.parse_args()

    log(f"reading {args.data_dir}")
    m = load_mtah_match(args.data_dir)
    check_match(m)

    names = [n for n in DFL if n not in args.exclude]
    log(f"training on {len(names)} DFL matches: {' '.join(names)}")
    dead, poss = train(load_training(Path(args.npz), names))

    p_dead = dead.predict_proba(m)
    alive = alive_mask(m, p_dead)
    p_home = poss.predict_home(m, alive)
    pred = predict(m, p_dead, p_home=p_home)
    g = m.half > 0
    log("prediction on graded frames: " + "  ".join(
        f"{c} {np.mean(pred[g] == k):.1%}" for k, c in enumerate("HADX")))

    out = write_submission(args.out, pred)
    log(f"wrote {out}  md5 {hashlib.md5(out.read_bytes()).hexdigest()}")

    validator = Path(args.repo) / "evaluation" / "validate_submission.py"
    if validator.exists():
        r = subprocess.run([sys.executable, str(validator), str(out)], capture_output=True, text=True)
        print(r.stdout.rstrip())
        if r.returncode != 0:
            print(r.stderr.rstrip())
            sys.exit("official validation FAILED")
    else:
        log(f"WARN {validator} not found, file not validated")

    if args.gt:
        gt = pm.label_codes(pm.read_label_csv(args.gt))
        s = score(gt, pred)
        log(f"score vs {args.gt}: spellF1 {s['spell_f1_mean']:.4f}  macroF1 {s['macro_f1']:.4f}  "
            f"spells {s['n_pred_spells']}/{s['n_gt_spells']}")


if __name__ == "__main__":
    main()
