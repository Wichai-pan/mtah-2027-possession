"""How far can the Smart Tagging be trusted? Measured on the full match WITHOUT any ground truth.

    uv run python scripts/eval_smart_tagging.py

Compares the automatic tags against two independent references, neither of which uses labels:
  - raw tracking: "restart signatures" (ball still >= 1 s, then kicked), player-speed profiles
  - our own v2 prediction (an imperfect model, so agreement is not accuracy -- it is a consistency
    check; where they disagree, either can be wrong)

Prints aggregate statistics only (counts, rates, quantiles): no per-instance rows, no raw data.
Writes the same to results/smart_tagging_eval.md.
"""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from mtah.official import possession_metrics as pm
from mtah.schema import A, D, H, Match

FPS = 25
TEAM = {"barcelona": H, "atletic club": A}
# Team tag = the team *performing* the phase. For these codes that team is attacking (has the ball):
ATTACK = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
          "PLAYERS IN THE BOX", "LONG BALL", "ATTACKING TRANSITION"}
# ...and for these it is defending (the other team has the ball):
DEFEND = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD",
          "DEFENSIVE TRANSITION"}
KICKOFFS = {1: 11185, 2: 113742}    # published in DATA.md / FAQ.md


def load_tags(path):
    out = []
    for ins in ET.parse(path).iter("instance"):
        tags = {(lab.findtext("group") or "").strip().lower(): (lab.findtext("text") or "").strip()
                for lab in ins.findall("label")}
        out.append(dict(a=round(float(ins.findtext("start")) * FPS),
                        b=round(float(ins.findtext("end")) * FPS),
                        code=(ins.findtext("code") or "").strip().upper(),
                        team=TEAM.get(tags.get("team", "").lower()),
                        type=tags.get("type", "")))
    return out


def ball_speed(m: Match) -> np.ndarray:
    """(T,) ball speed m/s from consecutive seen frames, 5-frame median; NaN where unknown."""
    b = m.metres(m.ball_xy)
    v = np.full(m.n_frames, np.nan)
    v[1:] = np.linalg.norm(np.diff(b, axis=0), axis=1) * FPS
    pad = np.pad(v, 2, constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, 5)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmedian(win, axis=1)


def runs(mask):
    """[(start, stop)] of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def restart_candidates(m, vb, still_s=1.0, kick=5.0, within_s=1.5):
    """Frames where a ball that sat still >= still_s is kicked (> kick m/s) within within_s."""
    still = (vb < 1.0) & (m.half > 0)
    out = []
    for a, b in runs(still):
        if b - a < still_s * FPS:
            continue
        seg = vb[b:b + int(within_s * FPS)]
        hit = np.flatnonzero(seg > kick)
        if len(hit):
            out.append(b + hit[0])
    return np.array(out, int)


def nearest_offset(t, cands, max_s=10.0):
    if len(cands) == 0:
        return None
    i = np.searchsorted(cands, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(cands):
            d = cands[j] - t
            if abs(d) <= max_s * FPS and (best is None or abs(d) < abs(best)):
                best = d
    return best


def q(x):
    x = np.asarray(x, float) / FPS
    if len(x) == 0:
        return "n/a"
    return f"median {np.median(x):+.2f}s  IQR [{np.percentile(x, 25):+.2f}, {np.percentile(x, 75):+.2f}]s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../external_data/mtah_bar_ath.npz")
    ap.add_argument("--tags", default="../repo/data/tactical/BAR-ATH_tactical_smart_tagging.xml")
    ap.add_argument("--pred", default="results/submission_v2_firstpass.csv")
    ap.add_argument("--out", default="results/smart_tagging_eval.md")
    args = ap.parse_args()

    m = Match.load(args.npz)
    m = Match(**{**m.__dict__, "labels": None})          # never look at labels here
    tags = load_tags(args.tags)
    pred = pm.label_codes(pm.read_label_csv(args.pred))
    vb = ball_speed(m)
    pspd = np.nanmean(np.where(np.isnan(m.player_speed), np.nan, m.player_speed), axis=1)
    graded = m.half > 0
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("# Smart Tagging vs tracking and our v2 prediction (no ground truth used)\n")
    P(f"{len(tags)} instances; team-tagged {sum(t['team'] is not None for t in tags)}; "
      f"inside a graded half {sum(graded[min(t['a'], m.n_frames - 1)] for t in tags)}")

    # ---------------------------------------------------------------- set pieces
    cands = restart_candidates(m, vb)
    pa = np.r_[False, (pred[:-1] == D) & np.isin(pred[1:], (H, A))] & graded
    pred_restart = np.flatnonzero(pa)
    sp = [t for t in tags if t["code"] == "SET PIECES"]
    P(f"\n## Set pieces ({len(sp)})\n")
    P(f"raw-tracking restart signatures in graded frames: {len(cands)}; "
      f"v2 predicted D->H/A edges: {len(pred_restart)}\n")
    P("| type | n | team tagged | restart signature within 10 s | offset to it (sig - tag start) | "
      "v2 D->alive within 10 s | offset | v2 first team == tag team |")
    P("|---|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for t in sp:
        by[t["type"] or "(none)"].append(t)
    for typ, ts in sorted(by.items(), key=lambda kv: -len(kv[1])):
        o1 = [o for o in (nearest_offset(t["a"], cands) for t in ts) if o is not None]
        o2, agree = [], []
        for t in ts:
            o = nearest_offset(t["a"], pred_restart)
            if o is None:
                continue
            o2.append(o)
            if t["team"] is not None:
                agree.append(pred[t["a"] + o] == t["team"])
        P(f"| {typ} | {len(ts)} | {sum(t['team'] is not None for t in ts)} | "
          f"{len(o1)}/{len(ts)} | {q(o1)} | {len(o2)}/{len(ts)} | {q(o2)} | "
          f"{(f'{np.mean(agree):.0%} of {len(agree)}' if agree else 'n/a')} |")

    # mean profiles around the tag start (independent of our model)
    P("\nMean profiles around a set-piece tag start (seconds relative to tag start):\n")
    secs = np.arange(-10, 11, 2)
    P("| s | " + " | ".join(f"{s:+d}" for s in secs) + " |")
    P("|---|" + "---|" * len(secs))
    prof = defaultdict(list)
    for t in sp:
        for s in secs:
            f = t["a"] + s * FPS
            if 0 <= f < m.n_frames:
                w = slice(f, f + FPS)
                prof["player speed m/s"].append((s, np.nanmean(pspd[w])))
                prof["ball speed m/s"].append((s, np.nanmean(vb[w]) if np.isfinite(vb[w]).any() else np.nan))
                prof["ball seen"].append((s, np.mean(~np.isnan(m.ball_xy[w, 0]))))
                prof["v2 says D"].append((s, np.mean(pred[w] == D)))
    for k, v in prof.items():
        arr = defaultdict(list)
        for s, x in v:
            arr[s].append(x)
        P(f"| {k} | " + " | ".join(f"{np.nanmean(arr[s]):.2f}" for s in secs) + " |")

    # coverage: long predicted stoppages that a set-piece tag explains
    long_d = [(a, b) for a, b in runs((pred == D) & graded) if b - a >= 3 * FPS]
    sp_start = np.array(sorted(t["a"] for t in sp))
    # a tag starts ~4-6 s BEFORE the restart, so look for one in [end - 10 s, end + 2 s]
    def tagged(t):
        return bool(((sp_start >= t - 10 * FPS) & (sp_start <= t + 2 * FPS)).any())
    cov = [tagged(b) for a, b in long_d]
    P(f"\nv2 stoppages >= 3 s: {len(long_d)}; with a set-piece tag starting in [end-10s, end+2s]: "
      f"{np.mean(cov):.0%}")
    still_long = [c for c in cands if np.nanmean(pspd[max(c - 2 * FPS, 0):c]) < 2.0]
    cov2 = [tagged(c) for c in still_long]
    P(f"restart signatures with slow players (< 2 m/s in the 2 s before): {len(still_long)}; "
      f"tagged as a set piece in the same window: {np.mean(cov2):.0%}")
    slow = [np.nanmean(pspd[t["a"]:t["a"] + 2 * FPS]) < 1.5 for t in sp]
    rnd = np.random.default_rng(0).choice(np.flatnonzero(graded), 2000)
    P(f"set-piece tags with players slow (< 1.5 m/s) in their first 2 s: {np.mean(slow):.0%} "
      f"(random graded times: {np.mean([np.nanmean(pspd[i:i + 2 * FPS]) < 1.5 for i in rnd]):.0%})")

    # kickoffs
    ko = [t for t in sp if t["type"].lower().startswith("kick off (start)")]
    P("\nKick Off (Start) tags vs the published kickoff frames: " + ", ".join(
        f"{(t['a'] - min(KICKOFFS.values(), key=lambda k: abs(k - t['a']))) / FPS:+.1f}s start / "
        f"{(t['b'] - min(KICKOFFS.values(), key=lambda k: abs(k - t['b']))) / FPS:+.1f}s end" for t in ko))

    # ---------------------------------------------------------------- phases: team agreement
    P("\n## Phase team tags vs v2 possession (frames where v2 says H or A)\n")
    P("| code | n | dur median s | v2 says D inside | v2 team == implied possessor |")
    P("|---|---|---|---|---|")
    vote = np.zeros((m.n_frames, 2))
    for code in sorted(ATTACK | DEFEND):
        ts = [t for t in tags if t["code"] == code and t["team"] is not None]
        if not ts:
            continue
        agree, dead, dur = [], [], []
        for t in ts:
            implied = t["team"] if code in ATTACK else 1 - t["team"]
            w = pred[t["a"]:t["b"] + 1]
            g = graded[t["a"]:t["b"] + 1]
            w = w[g]
            if not len(w):
                continue
            dur.append((t["b"] - t["a"]) / FPS)
            dead.append(np.mean(w == D))
            alive = w[w != D]
            if len(alive):
                agree.append(np.mean(alive == implied))
            vote[t["a"]:t["b"] + 1, implied] += 1
        P(f"| {code} | {len(ts)} | {np.median(dur):.1f} | {np.mean(dead):.0%} | "
          f"{np.mean(agree):.0%} |")
    alive = graded & np.isin(pred, (H, A))
    covered = alive & (vote.sum(1) > 0)
    clear = covered & (vote[:, 0] != vote[:, 1])
    v_team = np.argmax(vote, axis=1)
    P(f"\nv2-alive graded frames covered by >= 1 team-tagged phase: {covered.sum() / alive.sum():.0%}; "
      f"with an unambiguous phase vote: {clear.sum() / alive.sum():.0%}; "
      f"vote == v2 on those: {np.mean(v_team[clear] == pred[clear]):.0%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
