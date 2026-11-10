"""Convert the raw Sportec Open DFL XMLs into Match .npz files (clean + degraded copies).

    uv run python scripts/prepare_dfl.py --raw ../external_data/dfl/raw --out ../external_data/dfl/npz
"""

import argparse
import time
from pathlib import Path

from mtah.degrade import DegradeConfig, degrade
from mtah.io_dfl import load_dfl_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../external_data/dfl/raw")
    ap.add_argument("--out", default="../external_data/dfl/npz")
    ap.add_argument("--seed", type=int, default=0, help="seed for the degraded copy")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for pos in sorted(raw.glob("DFL_04_03_positions_raw_observed_*.xml")):
        mid = pos.stem.split("_")[-1]
        info = next(raw.glob(f"DFL_02_01_matchinformation_*_{mid}.xml"))
        t0 = time.time()
        m = load_dfl_match(info, pos)
        m.save(out / f"{m.name}.npz")
        d = degrade(m, DegradeConfig(seed=args.seed))
        d.save(out / f"{d.name}.npz")
        print(f"[{time.time() - t0:5.1f}s] {m.meta['title']}")
        print(m.summary())
        print(f"  degraded: ball missing on {d.meta['ball_missing_graded']:.1%} of graded frames, "
              f"{d.n_slots} slots")


if __name__ == "__main__":
    main()
