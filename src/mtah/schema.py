"""The one in-memory representation every part of the pipeline works on.

Both the official match (Metrica automatic tracking) and the DFL open dataset (TRACAB, corrected)
are converted into a `Match`, so features, decoders and evaluation never care where data came from.

Conventions
-----------
- One row per frame at `fps` (25). Row i of the official match is video frame i == submission row i.
- Coordinates are normalised to [0, 1] on both axes, like the official ATD. Multiply by `pitch`
  (length, width) in metres to get distances. x runs along the length, y along the width.
- Players live in *slots*: `player_xy[:, s]` is slot s. A slot has a fixed team (`slot_team`) but is
  NOT a person -- in the official data slots fragment and swap. Missing = NaN.
- `half[t]` is 1 or 2 for frames inside a half (the graded frames), 0 elsewhere.
- `labels[t]` uses the scorer's codes: 0=H, 1=A, 2=D, 3=X (see mtah.official.possession_metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = np.array(["H", "A", "D", "X"])
HOME, AWAY = 0, 1


@dataclass
class Match:
    name: str
    fps: int
    pitch: tuple[float, float]          # (length, width) in metres
    half: np.ndarray                    # (T,) int8: 0 outside play, 1 / 2 inside a half
    ball_xy: np.ndarray                 # (T, 2) float32, normalised, NaN when unseen
    player_xy: np.ndarray               # (T, S, 2) float32, normalised, NaN when unseen
    player_speed: np.ndarray            # (T, S) float32, m/s, NaN when unseen
    slot_team: np.ndarray               # (S,) int8: HOME or AWAY
    labels: np.ndarray | None = None    # (T,) int8 ground truth codes, None if unknown
    ball_z: np.ndarray | None = None    # (T,) float32 metres, only where the source has it
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.half)

    @property
    def n_slots(self) -> int:
        return self.player_xy.shape[1]

    def half_blocks(self) -> list[tuple[int, int, int]]:
        """[(half, start, stop)] half-open frame ranges of each half."""
        out = []
        for h in (1, 2):
            idx = np.flatnonzero(self.half == h)
            if len(idx):
                out.append((h, int(idx[0]), int(idx[-1]) + 1))
        return out

    def metres(self, xy: np.ndarray) -> np.ndarray:
        """Normalised -> metres, broadcasting over any leading shape."""
        return xy * np.asarray(self.pitch, dtype=np.float32)

    def summary(self) -> str:
        graded = self.half > 0
        ball_seen = ~np.isnan(self.ball_xy[:, 0])
        n_players = (~np.isnan(self.player_xy[:, :, 0])).sum(1)
        lines = [
            f"{self.name}: {self.n_frames} frames @ {self.fps} fps, {self.n_slots} slots, "
            f"pitch {self.pitch[0]:.1f} x {self.pitch[1]:.1f} m",
            f"  graded frames {graded.sum()}  halves {self.half_blocks()}",
            f"  ball seen on {ball_seen[graded].mean():.1%} of graded frames; "
            f"mean players seen {n_players[graded].mean():.1f}",
        ]
        if self.labels is not None:
            lab = self.labels[graded]
            lines.append("  labels  " + "  ".join(
                f"{c} {np.mean(lab == k):.1%}" for k, c in enumerate("HAD")))
        return "\n".join(lines)

    # -- persistence -----------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        arrays = dict(half=self.half, ball_xy=self.ball_xy, player_xy=self.player_xy,
                      player_speed=self.player_speed, slot_team=self.slot_team)
        if self.labels is not None:
            arrays["labels"] = self.labels
        if self.ball_z is not None:
            arrays["ball_z"] = self.ball_z
        np.savez_compressed(path, name=self.name, fps=self.fps, pitch=np.asarray(self.pitch),
                            meta=np.asarray(repr(self.meta)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Match":
        z = np.load(path, allow_pickle=False)
        import ast
        return cls(
            name=str(z["name"]), fps=int(z["fps"]), pitch=tuple(float(v) for v in z["pitch"]),
            half=z["half"], ball_xy=z["ball_xy"], player_xy=z["player_xy"],
            player_speed=z["player_speed"], slot_team=z["slot_team"],
            labels=z["labels"] if "labels" in z else None,
            ball_z=z["ball_z"] if "ball_z" in z else None,
            meta=ast.literal_eval(str(z["meta"])),
        )


def labels_to_chars(codes: np.ndarray) -> np.ndarray:
    return LABEL_CHARS[codes]


def write_submission(path: str | Path, codes: np.ndarray) -> Path:
    """frame,label CSV in the official contract (one row per frame, X allowed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chars = labels_to_chars(codes)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{c}\n" for i, c in enumerate(chars))
    return path
