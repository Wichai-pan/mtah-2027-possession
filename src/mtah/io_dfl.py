"""Sportec Open DFL dataset (Bassek et al. 2025, CC-BY 4.0) -> Match.

Seven Bundesliga matches, TRACAB optical tracking at 25 fps, with per-frame official flags on the
ball frames:
    BallStatus     1 = alive, 0 = dead
    BallPossession 1 = home,  2 = guest

which map one-to-one onto the challenge labels: dead -> D, else home -> H / guest -> A.

Raw coordinates are metres with the origin at the centre spot (x along the length, y across).
They are converted to the challenge convention: normalised [0, 1] on both axes.

The two halves are written back to back with a short block of X frames between them, so that the
official scorer (which splits on X) can never stitch a possession across halftime.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from lxml import etree

from .schema import A, AWAY, D, H, HOME, X, Match

SECTIONS = {"firstHalf": 1, "secondHalf": 2}
HALFTIME_GAP = 250  # X frames inserted between halves (10 s)


def _match_info(path: Path) -> dict:
    root = etree.parse(str(path)).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    teams = {}
    for team in root.iter("Team"):
        teams[team.get("TeamId")] = team.get("Role")  # "home" / "guest"
    return {
        "match_id": general.get("MatchId"),
        "title": general.get("MatchTitle"),
        "home": general.get("HomeTeamName"),
        "away": general.get("GuestTeamName"),
        "pitch": (float(env.get("PitchX")), float(env.get("PitchY"))),
        "team_role": teams,
    }


def _stream_positions(path: Path):
    """Yield (section, team_id, person_id, frames) per FrameSet; frames = (N, X, Y, S[, Z, poss, status])."""
    context = etree.iterparse(str(path), events=("start", "end"), huge_tree=True)
    current = None
    rows: list = []
    for event, elem in context:
        tag = elem.tag
        if event == "start" and tag == "FrameSet":
            current = (elem.get("GameSection"), elem.get("TeamId"), elem.get("PersonId"))
            rows = []
        elif event == "end" and tag == "Frame":
            get = elem.get
            if current[1] == "BALL":
                rows.append((int(get("N")), float(get("X")), float(get("Y")), float(get("S") or "nan"),
                             float(get("Z") or "nan"), int(get("BallPossession") or 0),
                             int(get("BallStatus") or 0)))
            else:
                rows.append((int(get("N")), float(get("X")), float(get("Y")), float(get("S") or "nan")))
            elem.clear()
        elif event == "end" and tag == "FrameSet":
            yield current[0], current[1], current[2], rows
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


def load_dfl_match(info_path: str | Path, positions_path: str | Path) -> Match:
    info = _match_info(Path(info_path))
    length, width = info["pitch"]

    # Collect per section: ball rows and per-person rows.
    ball: dict[int, np.ndarray] = {}
    people: dict[int, dict[tuple[str, str], np.ndarray]] = defaultdict(dict)
    for section, team_id, person_id, rows in _stream_positions(Path(positions_path)):
        half = SECTIONS.get(section)
        if half is None:
            continue  # extra time, if any
        arr = np.asarray(rows, dtype=np.float64)
        if team_id == "BALL":
            ball[half] = arr
        elif team_id in info["team_role"]:
            people[half][(team_id, person_id)] = arr
        # referees and anything else are skipped

    persons = sorted({key for half in people.values() for key in half},
                     key=lambda k: (info["team_role"][k[0]] != "home", k[1]))
    slot_of = {key: s for s, key in enumerate(persons)}
    slot_team = np.array([HOME if info["team_role"][t] == "home" else AWAY for t, _ in persons],
                         dtype=np.int8)

    blocks = []
    for half in (1, 2):
        b = ball[half]
        n0, n1 = int(b[:, 0].min()), int(b[:, 0].max())
        T = n1 - n0 + 1

        ball_xy = np.full((T, 2), np.nan, np.float32)
        ball_z = np.full(T, np.nan, np.float32)
        labels = np.full(T, X, np.int8)
        i = (b[:, 0] - n0).astype(int)
        ball_xy[i, 0] = (b[:, 1] + length / 2) / length
        ball_xy[i, 1] = (b[:, 2] + width / 2) / width
        ball_z[i] = b[:, 4]
        poss, status = b[:, 5].astype(int), b[:, 6].astype(int)
        labels[i] = np.where(status == 0, D, np.where(poss == 1, H, np.where(poss == 2, A, X)))

        pxy = np.full((T, len(persons), 2), np.nan, np.float32)
        psp = np.full((T, len(persons)), np.nan, np.float32)
        for key, arr in people[half].items():
            s = slot_of[key]
            j = (arr[:, 0] - n0).astype(int)
            keep = (j >= 0) & (j < T)
            j, arr = j[keep], arr[keep]
            pxy[j, s, 0] = (arr[:, 1] + length / 2) / length
            pxy[j, s, 1] = (arr[:, 2] + width / 2) / width
            psp[j, s] = arr[:, 3] / 3.6   # DFL 'S' is km/h; the official ATD is m/s
        blocks.append((half, ball_xy, ball_z, labels, pxy, psp, n0))

    def gap(shape, fill, dtype):
        return np.full((HALFTIME_GAP,) + shape, fill, dtype)

    S = len(persons)
    parts = {"half": [], "ball_xy": [], "ball_z": [], "labels": [], "pxy": [], "psp": []}
    for k, (half, ball_xy, ball_z, labels, pxy, psp, _) in enumerate(blocks):
        if k:
            parts["half"].append(gap((), 0, np.int8))
            parts["ball_xy"].append(gap((2,), np.nan, np.float32))
            parts["ball_z"].append(gap((), np.nan, np.float32))
            parts["labels"].append(gap((), X, np.int8))
            parts["pxy"].append(gap((S, 2), np.nan, np.float32))
            parts["psp"].append(gap((S,), np.nan, np.float32))
        parts["half"].append(np.full(len(labels), half, np.int8))
        parts["ball_xy"].append(ball_xy)
        parts["ball_z"].append(ball_z)
        parts["labels"].append(labels)
        parts["pxy"].append(pxy)
        parts["psp"].append(psp)
    cat = {k: np.concatenate(v) for k, v in parts.items()}

    return Match(
        name=info["match_id"].replace("DFL-MAT-", "DFL-"),
        fps=25, pitch=(length, width), half=cat["half"],
        ball_xy=cat["ball_xy"], player_xy=cat["pxy"], player_speed=cat["psp"],
        slot_team=slot_team, labels=cat["labels"], ball_z=cat["ball_z"],
        meta={"source": "dfl", "title": info["title"], "home": info["home"],
              "away": info["away"], "first_frame_N": [b[6] for b in blocks]},
    )
