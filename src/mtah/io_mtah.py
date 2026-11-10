"""Official challenge match (Metrica automatic tracking, FIFA/EPTS export) -> Match.

Mirrors the parsing in the official starter/baseline.py so the two agree frame for frame:
  raw line  "<frame_id>:<slot1>;<slot2>;...;<slot46>:<ball>", each group "x,y,speed", NaN if unseen
  frame_id counts from 1; submission row = frame_id - 1
  the first <Team> in the metadata XML is the home side (Barcelona)

UNTESTED until the data bundle arrives -- run scripts/prepare_mtah.py and compare the
nearest-player baseline it produces against the official baseline CSVs before trusting it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .official import possession_metrics as pm
from .schema import AWAY, HOME, Match

# From the official games.py / DATA.md: inclusive video-frame span of each half.
GAMES = {
    "BAR-ATH": {
        "n_frames": 199319,
        "fps": 25,
        "pitch": (105.3, 68.0),
        "halves": {1: (11185, 90651), 2: (110808, 190197)},
        "stem": "tactical/BAR-ATH_tactical",
    },
}


def _slot_teams(meta_path: Path) -> tuple[np.ndarray, tuple[str, str]]:
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(teams)}")
    home_id = teams[0].get("id")
    slots = [p.get("teamId") for p in root.iter("Player")]
    names = tuple(t.findtext("Name") for t in teams)
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8), names


def _parse_group(group: str) -> tuple[float, float, float]:
    parts = group.split(",")
    if not parts[0] or parts[0][0] == "N":
        return np.nan, np.nan, np.nan
    speed = float(parts[2]) if len(parts) > 2 and parts[2] and parts[2][0] != "N" else np.nan
    return float(parts[0]), float(parts[1]), speed


def load_mtah_match(data_dir: str | Path, game: str = "BAR-ATH",
                    gt_csv: str | Path | None = None) -> Match:
    cfg = GAMES[game]
    data_dir = Path(data_dir)
    meta_path = data_dir / f"{cfg['stem']}_atd.xml"
    raw_path = data_dir / f"{cfg['stem']}_atd.txt"

    slot_team, names = _slot_teams(meta_path)
    T, S = cfg["n_frames"], len(slot_team)
    ball_xy = np.full((T, 2), np.nan, np.float32)
    pxy = np.full((T, S, 2), np.nan, np.float32)
    psp = np.full((T, S), np.nan, np.float32)

    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            t = int(parts[0]) - 1
            if not 0 <= t < T:
                continue
            bx, by, _ = _parse_group(parts[2])
            ball_xy[t] = (bx, by)
            for s, group in enumerate(parts[1].split(";")[:S]):
                x, y, sp = _parse_group(group)
                pxy[t, s] = (x, y)
                psp[t, s] = sp

    half = np.zeros(T, np.int8)
    for h, (a, b) in cfg["halves"].items():
        half[a:b + 1] = h

    labels = None
    if gt_csv is not None:
        labels = pm.label_codes(pm.read_label_csv(str(gt_csv))).astype(np.int8)

    return Match(name=game, fps=cfg["fps"], pitch=cfg["pitch"], half=half, ball_xy=ball_xy,
                 player_xy=pxy, player_speed=psp, slot_team=slot_team, labels=labels,
                 meta={"source": "mtah", "home": names[0], "away": names[1]})
