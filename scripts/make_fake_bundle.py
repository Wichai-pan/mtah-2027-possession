"""Write a degraded DFL match out as a fake official data bundle, to dry-run the submission path.

    uv run python scripts/make_fake_bundle.py --match DFL-J03WQQ --out ../external_data/fake_mtah

Produces, in the official layout and file formats (DATA.md / starter/baseline.py):
  <out>/tactical/BAR-ATH_tactical_atd.xml   EPTS metadata: 2 <Team>, home first; 46 <Player teamId=..>
  <out>/tactical/BAR-ATH_tactical_atd.txt   "<frame_id>:<x,y,s>;...;<x,y,s>:<bx,by,bs>", frame_id from 1
  <out>/gt_full.csv                          the match's official-definition labels, frame,label
  <out>/gt_public.csv                        the same, X outside the first 5 minutes of play
  <out>/source.npz                           the Match exactly as placed on the fake timeline

Each DFL half is placed at the start of the corresponding official half (video frames 11185 and
110808), truncated if longer; graded frames it does not cover are X in the ground truth.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from mtah.io_mtah import GAMES
from mtah.relabel import relabel_official
from mtah.schema import X, Match, labels_to_chars

GAME = "BAR-ATH"


def place(m: Match, labels: np.ndarray) -> Match:
    cfg = GAMES[GAME]
    T = cfg["n_frames"]
    S = m.n_slots
    out = Match(name="FAKE-" + m.name, fps=m.fps, pitch=m.pitch, half=np.zeros(T, np.int8),
                ball_xy=np.full((T, 2), np.nan, np.float32),
                player_xy=np.full((T, S, 2), np.nan, np.float32),
                player_speed=np.full((T, S), np.nan, np.float32),
                slot_team=m.slot_team.copy(), labels=np.full(T, X, np.int8),
                meta={"source": m.name})
    for h, (a, b) in cfg["halves"].items():
        out.half[a:b + 1] = h
        src = np.flatnonzero(m.half == h)
        n = min(len(src), b + 1 - a)
        dst = slice(a, a + n)
        out.ball_xy[dst] = m.ball_xy[src[:n]]
        out.player_xy[dst] = m.player_xy[src[:n]]
        out.player_speed[dst] = m.player_speed[src[:n]]
        out.labels[dst] = labels[src[:n]]
    return out


def _group(x, y, s) -> str:
    if np.isnan(x):
        return "NaN,NaN,NaN"
    return f"{x:.5f},{y:.5f},{0.0 if np.isnan(s) else s:.2f}"


def write_atd(m: Match, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{GAME}_tactical_atd"
    teams = ("FC Home", "Away Club")
    players = "\n".join(
        f'      <Player id="P{s + 1}" teamId="T{int(k) + 1}"><Name>Track {s + 1}</Name>'
        f"<ShirtNumber>{s + 1}</ShirtNumber></Player>" for s, k in enumerate(m.slot_team))
    stem.with_suffix(".xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<main>\n  <Metadata>\n    <Teams>\n'
        + "".join(f'      <Team id="T{i + 1}"><Name>{n}</Name></Team>\n' for i, n in enumerate(teams))
        + "    </Teams>\n    <Players>\n" + players + "\n    </Players>\n  </Metadata>\n</main>\n")
    # ball speed is not used by any reader; write the per-frame displacement speed
    b = m.ball_xy * np.asarray(m.pitch, np.float32)
    bs = np.r_[np.nan, np.linalg.norm(np.diff(b, axis=0), axis=1) * m.fps]
    with open(stem.with_suffix(".txt"), "w") as fh:
        for t in range(m.n_frames):
            ps = ";".join(_group(*m.player_xy[t, s], m.player_speed[t, s]) for s in range(m.n_slots))
            fh.write(f"{t + 1}:{ps}:{_group(*m.ball_xy[t], bs[t])}\n")


def write_gt(path: Path, codes: np.ndarray) -> None:
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{c}\n" for i, c in enumerate(labels_to_chars(codes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../external_data/dfl/npz")
    ap.add_argument("--match", default="DFL-J03WQQ")
    ap.add_argument("--out", default="../external_data/fake_mtah")
    args = ap.parse_args()

    clean = Match.load(Path(args.npz) / f"{args.match}.npz")
    deg = Match.load(Path(args.npz) / f"{args.match}-degraded.npz")
    fake = place(deg, relabel_official(clean))
    out = Path(args.out)
    write_atd(fake, out / "tactical")
    write_gt(out / "gt_full.csv", fake.labels)
    a = GAMES[GAME]["halves"][1][0]
    public = np.full(fake.n_frames, X, np.int8)
    public[a:a + 5 * 60 * fake.fps] = fake.labels[a:a + 5 * 60 * fake.fps]
    write_gt(out / "gt_public.csv", public)
    fake.save(out / "source.npz")
    print(fake.summary())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
