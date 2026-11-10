"""Render short example clips around Smart Tagging set-piece tags, for a human to WATCH.

    uv run --group viz python scripts/st_examples.py            # 2 per set-piece type
    uv run --group viz python scripts/st_examples.py --per-type 4

Each clip runs from 8 s before to 15 s after a tag start; the tag start is the green line on the
timeline and the "t - tag start" clock in the header. results/viz/st_examples/index.md lists them.

Purpose: understanding what a tag means (is it a real restart? where does play resume relative to
it? is the team right?). Per RULES.md, watching is fine; turning what is seen into labels or
frame-specific constants is not -- nothing watched here is fed back into the model.
Output stays local (results/ is gitignored).
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_smart_tagging import load_tags  # noqa: E402

FPS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="../repo/data/tactical/BAR-ATH_tactical_smart_tagging.xml")
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--before", type=float, default=8)
    ap.add_argument("--after", type=float, default=15)
    ap.add_argument("--out", default="results/viz/st_examples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by = defaultdict(list)
    for t in load_tags(args.tags):
        if t["code"] == "SET PIECES":
            by[t["type"] or "(none)"].append(t)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = ["| clip | type | team | video time of tag start | what you see |", "|---|---|---|---|---|"]
    team_name = {0: "Barcelona", 1: "Athletic", None: "N/A"}
    i = 0
    for typ in sorted(by):
        ts = by[typ]
        pick = sorted(rng.choice(len(ts), min(args.per_type, len(ts)), replace=False))
        for k in pick:
            t = ts[k]
            i += 1
            name = f"{i:02d}_{typ.lower().replace(' ', '_').replace('(', '').replace(')', '')}.mp4"
            s, e = t["a"] - int(args.before * FPS), t["a"] + int(args.after * FPS)
            subprocess.run([sys.executable, "scripts/render_overlay.py", "--start", str(s), "--end", str(e),
                            "--mark", str(t["a"]), "--span-s", "15", "--out", str(out / name)],
                           check=True, stdout=subprocess.DEVNULL)
            mm, ss = divmod(t["a"] / FPS, 60)
            rows.append(f"| {name} | {typ} | {team_name[t['team']]} | {int(mm):02d}:{ss:04.1f} | |")
            print(f"  rendered {i}", flush=True)
    (out / "index.md").write_text(
        "# Smart Tagging set-piece examples\n\nGreen line / clock = tag start. "
        "Fill in the last column: real restart? roughly when? right team?\n\n" + "\n".join(rows) + "\n")
    print(f"wrote {i} clips and index.md to {out}")


if __name__ == "__main__":
    main()
