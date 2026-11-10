"""Constrained Viterbi decoding: per-frame class scores -> a clean H / A / D sequence.

Why this exists: the ranking metric (spellF1) scores possessions as whole segments, so a decoder
that refuses to flicker is worth more than a sharper per-frame classifier. Two levers:

  switch penalties   cost (in nats) of moving from class i to class j; H<->A can be made dearer
                     than entering / leaving D.
  minimum durations  a segment of class c must last at least min_dur[c] frames before it may end
                     (the first segment of each half is exempt).

Implementation: each class c is expanded into a chain of min_dur[c] sub-states
(c,1) -> (c,2) -> ... -> (c,d_c) with a self-loop on (c,d_c). Leaving c is only possible from
(c,d_c). The chain shifts are deterministic, so back-pointers are only needed for the two entry
points of each chain, which keeps memory at O(T * n_classes).

Decoding runs separately inside each half; frames outside the halves are written as X.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import X

N_CLASSES = 3  # H, A, D (codes 0, 1, 2)


@dataclass
class DecoderConfig:
    # penalty[i][j] for switching from class i to class j (diagonal ignored)
    penalty: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 8.0, 4.0],    # from H: to A, to D
        [8.0, 0.0, 4.0],    # from A
        [4.0, 4.0, 0.0],    # from D
    ]))
    min_dur: tuple[int, int, int] = (25, 25, 38)   # frames: H, A, D (1 s, 1 s, 1.5 s at 25 fps)


def viterbi(log_prob: np.ndarray, cfg: DecoderConfig) -> np.ndarray:
    """Best class path for one contiguous block. log_prob: (T, 3). Returns (T,) int8 in {0,1,2}."""
    T = len(log_prob)
    if T == 0:
        return np.zeros(0, np.int8)
    d = [max(1, int(v)) for v in cfg.min_dur]
    pen = np.asarray(cfg.penalty, dtype=np.float64)
    lp = np.asarray(log_prob, dtype=np.float64)

    # score[c] is a vector of length d[c]: score of being k+1 frames into a segment of class c
    # (the last entry means ">= d[c] frames", i.e. free to leave).
    NEG = -np.inf
    score = [np.full(d[c], NEG) for c in range(N_CLASSES)]
    for c in range(N_CLASSES):
        score[c][-1] = lp[0, c]  # first segment exempt from the minimum duration

    entry_from = np.zeros((T, N_CLASSES), np.int8)   # which class we came from on entering c
    stay = np.zeros((T, N_CLASSES), bool)            # at (c,d_c): True if came from (c,d_c)

    for t in range(1, T):
        ready = np.array([score[c][-1] for c in range(N_CLASSES)])  # classes free to leave
        new = []
        for c in range(N_CLASSES):
            cand = ready - pen[:, c]
            cand[c] = NEG
            j = int(np.argmax(cand))
            enter = cand[j]
            entry_from[t, c] = j
            s = np.empty(d[c])
            if d[c] == 1:
                stay_val = score[c][0]
                if stay_val >= enter:
                    s[0] = stay_val
                    stay[t, c] = True
                else:
                    s[0] = enter
            else:
                s[0] = enter
                s[1:-1] = score[c][:-2]
                a, b = score[c][-2], score[c][-1]
                stay[t, c] = b >= a
                s[-1] = max(a, b)
            new.append(s + lp[t, c])
        score = new

    # Backtrack. Only states (c, d_c) are valid end states? No: a half may end mid-segment, so any
    # sub-state may end the path.
    best_c, best_k, best_v = 0, 0, NEG
    for c in range(N_CLASSES):
        k = int(np.argmax(score[c]))
        if score[c][k] > best_v:
            best_c, best_k, best_v = c, k, score[c][k]

    path = np.empty(T, np.int8)
    c, k = best_c, best_k
    for t in range(T - 1, -1, -1):
        path[t] = c
        if t == 0:
            break
        if k == d[c] - 1:  # at the free state: either stayed, or arrived from k-1 / entry
            if stay[t, c]:
                continue
            if d[c] == 1:
                c, k = int(entry_from[t, c]), d[int(entry_from[t, c])] - 1
            else:
                k -= 1
        elif k == 0:
            j = int(entry_from[t, c])
            c, k = j, d[j] - 1
        else:
            k -= 1
    return path


def decode_match(log_prob: np.ndarray, half: np.ndarray, cfg: DecoderConfig | None = None) -> np.ndarray:
    """Decode each half independently; X outside the halves. log_prob: (T, 3)."""
    cfg = cfg or DecoderConfig()
    out = np.full(len(half), X, np.int8)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx):
            a, b = idx[0], idx[-1] + 1
            out[a:b] = viterbi(log_prob[a:b], cfg)
    return out


def probs_to_logprob(prob: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float64), floor, 1.0)
    return np.log(p / p.sum(1, keepdims=True))
