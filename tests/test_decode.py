import itertools
import numpy as np
from mtah.decode import DecoderConfig, viterbi


def brute(lp, cfg):
    T = len(lp)
    best, best_path = -np.inf, None
    for path in itertools.product(range(3), repeat=T):
        # segments
        segs, start = [], 0
        for t in range(1, T + 1):
            if t == T or path[t] != path[t - 1]:
                segs.append((path[start], t - start)); start = t
        ok = all(L >= cfg.min_dur[c] for i, (c, L) in enumerate(segs) if 0 < i < len(segs) - 1)
        if not ok:
            continue
        s = sum(lp[t, path[t]] for t in range(T))
        s -= sum(cfg.penalty[path[t - 1]][path[t]] for t in range(1, T) if path[t] != path[t - 1])
        if s > best + 1e-9:
            best, best_path = s, path
    return best


def path_score(lp, path, cfg):
    T = len(lp)
    return sum(lp[t, path[t]] for t in range(T)) - sum(
        cfg.penalty[path[t - 1]][path[t]] for t in range(1, T) if path[t] != path[t - 1])


def test_matches_bruteforce():
    rng = np.random.default_rng(0)
    for trial in range(300):
        T = int(rng.integers(1, 8))
        cfg = DecoderConfig(penalty=rng.uniform(0, 2, (3, 3)), min_dur=tuple(int(v) for v in rng.integers(1, 4, 3)))
        lp = np.log(rng.dirichlet(np.ones(3), T))
        path = viterbi(lp, cfg)
        segs, start = [], 0
        for t in range(1, T + 1):
            if t == T or path[t] != path[t - 1]:
                segs.append((path[start], t - start)); start = t
        assert all(L >= cfg.min_dur[c] for i, (c, L) in enumerate(segs) if 0 < i < len(segs) - 1)
        assert np.isclose(path_score(lp, path, cfg), brute(lp, cfg)), (trial, path)
