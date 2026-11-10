# mtah — frame-level team possession (More than a Hack 2027, Phase 1)

Our solution code. The official challenge repo lives next to it in `../repo`.

## Layout

```
src/mtah/
  schema.py        Match: the one in-memory format (normalised coords, slots, half, labels)
  io_mtah.py       official match (Metrica EPTS ATD + GT csv) -> Match    [untested until data arrives]
  io_dfl.py        Sportec Open DFL (7 Bundesliga matches, CC-BY) -> Match, with per-frame H/A/D labels
  degrade.py       make clean DFL look like the official automatic tracking (ball dropout, slot
                   fragmentation, player dropout, jitter) -- parameters are placeholders for now
  baseline.py      the official nearest-player + majority-vote baseline, on any Match
  relabel.py       official possession definition applied to DFL labels (validation target)
  tracker.py       rule state machine: touches -> episodes -> confirmed possession changes
  deadball.py      P(dead ball) from ID-free multi-scale features + gradient boosting
  possession_model.py  v2: learned P(home | alive) from ID-free features (+ tracker verdict)
  pipeline.py      P(D) + possession evidence (v2 learned / v1 tracker) -> constrained Viterbi
  boundary.py      snap D boundaries to ball-out / kick events (optional; ~no gain on DFL)
  decode.py        constrained Viterbi (switch penalties + minimum durations), per half
  evaluate.py      official metric (vendored, byte-identical) on one or many matches
  official/        vendored possession_metrics.py from the challenge repo (MIT)
scripts/
  prepare_dfl.py   raw DFL XML -> ../external_data/dfl/npz/{DFL-xxx, DFL-xxx-degraded}.npz
  eval_dfl.py      score every registered method on clean + degraded DFL
docs/possession_rules.md  the official definition written as rules R1-R9
tests/             decoder vs brute force; tracker rules
```

## Setup

```bash
uv sync
```

DFL data (≈2.6 GB, CC-BY 4.0, https://doi.org/10.6084/m9.figshare.28196177) goes in
`../external_data/dfl/raw`, then:

```bash
uv run python scripts/prepare_dfl.py
uv run python scripts/eval_dfl.py
```

## Making the submission

```bash
uv run python scripts/make_submission.py --data-dir ../repo/data --out results/submission.csv
```

It reads the official ATD, checks it (frame count, teams, ball / player visibility), trains both
models on the 7 degraded DFL matches (~40 s), predicts, writes `frame,label`, runs the official
`validate_submission.py`, and prints the md5. `--gt <csv>` also scores it — use the public 5 minutes
only to confirm the pipeline is wired correctly, never to choose anything.

**Dry run without the official data (2026-11-06).** `scripts/make_fake_bundle.py` writes a degraded
DFL match out in the official file formats on the official timeline (halves at video frames
11185 / 110808, `frame_id = frame + 1`, home team first in the XML):

```bash
uv run python scripts/make_fake_bundle.py --match DFL-J03WQQ
uv run python scripts/make_submission.py --data-dir ../external_data/fake_mtah \
    --exclude DFL-J03WQQ --gt ../external_data/fake_mtah/gt_full.csv --out results/dryrun.csv
```

- `io_mtah` reads the fake files back identical to the source arrays (also `tests/test_io_mtah.py`).
- The official `starter/baseline.py`, run on the fake bundle, agrees with our `baseline_predict`
  on 100% of graded frames (outside the halves it writes D, we write X; both unscored).
- The file passes the official validator; spellF1 0.543 vs 0.544 for the same match in the LOMO
  evaluation — the script and the evaluation agree.
- Two runs give the same md5: the output is reproducible byte for byte.
- Not covered: the real files may differ in details the docs do not show (extra XML fields,
  ball-group layout). The loader follows `starter/baseline.py`, so whatever that parses, we parse.
  On the fake bundle the "15+ players seen" check warns only because the DFL halves are shorter
  than the official ones and the tail is empty.

## Rules we keep
- Tune only on DFL. The public 5 minutes of the official match are for checking the pipeline.
- No match data in this repo (`.gitignore` covers `data/`, `*.npz`, `*.xml`, `*.mp4`).
- Everything seeded and deterministic: the repo must regenerate the submission byte for byte.

## First results on DFL (2026-11-06)

7 matches, pooled spellF1 (all matches scored as one), official metric. "degraded" = ball dropped
on 41% of graded frames in bursts, tracks fragmented over 46 slots, 7% player dropout, 0.3 m jitter.

| method | clean spellF1 | degraded spellF1 | degraded macroF1 |
|---|---|---|---|
| baseline_raw | 0.098 | 0.012 | 0.371 |
| baseline_smooth25 | 0.296 | 0.117 | 0.363 |
| baseline_smooth50 | 0.307 | 0.165 | 0.349 |
| baseline_smooth125 | 0.217 | 0.164 | 0.348 |
| nearest_viterbi (decoder demo) | 0.319 | 0.171 | 0.339 |

Same ordering as the official full-match numbers (smooth50 best among baselines; official 0.2433).

Findings
- **D is the missing piece.** Clean DFL always has the ball, so the baseline never says D
  (f1_D = 0). Dead ball is 41–52% of DFL frames. The decoder cannot help without D evidence.
- **DFL possessions are much shorter than the official ones**: DFL mean 7.2 s / median ~3.5 s,
  13% under 1 s; official public 5 min mean 14.2 s / median 7.6 s, 1 of 17 under 1 s. DFL's
  `BallPossession` likely flips on touches that Metrica's definition ignores. Optimal penalties /
  minimum durations tuned on raw DFL labels will therefore be too twitchy for the official match;
  consider a "sticky" relabelling of DFL as a second validation target.
- Degradation parameters are placeholders; recalibrate from the official data (ball gap lengths,
  tracklet lengths, visibility, and whether ball loss correlates with dead ball).

## v1: rules from the official definition (2026-11-06)

Rules: [docs/possession_rules.md](docs/possession_rules.md). Two pieces:

1. **Official-definition validation target.** DFL labels rewritten by R3/R4/R7: an in-play spell
   that lasts < 2 s and has < 3 touching players is absorbed into the previous possession; the
   first spell after a stoppage is always kept. Spells per match 437 -> 289; mean 7.3 -> 11.0 s;
   median 3.8 -> 6.8 s; < 1 s 12% -> 0.6% (official public 5 min: mean 14.2 s, median 7.6 s).
   T_control = 2 s follows the definition ("a few seconds"); it was not fitted to the public data.
2. **Pipeline**: dead-ball model (leave-one-match-out on DFL) + rule tracker + Viterbi.

Pooled over 7 DFL matches, degraded version, official-definition target:

| method | spellF1 | macroF1 | f1_D | spells (gt 2024) |
|---|---|---|---|---|
| baseline_smooth50 | 0.169 | 0.350 | 0.06 | 4932 |
| tracker with oracle D (diagnostic: possession part only) | 0.512 | 0.869 | 1.00 | 1281 |
| pipeline, dead-ball model on player features only (**conservative**) | **0.396** | 0.789 | 0.90 | 1492 |
| pipeline, all dead-ball features | 0.447 | 0.828 | 0.95 | 1365 |

Caveats
- DFL freezes the ball during dead time (speed 0, z 0), so ball features over-predict D quality:
  clean ball-only features reach D F1 0.996. Player-only features give 0.89-0.90 (Lang et al. 2022
  report 0.93), which is the number to trust; treat 0.396 as the realistic v1 estimate.
- The tracker under-segments (1281-1492 spells vs 2024): it is stickier than the target. Tracker
  thresholds, decoder penalties and team confidence are untuned defaults -- next step is tuning
  them on DFL (spellF1, official target).
- Degradation is still a guess; recalibrate once the official data is in.

Run: `uv run python scripts/eval_dfl.py` (all variants, about 6 min; restrict with
`--versions degraded --targets official --methods baseline_smooth50 pipeline_playersD`).

## Tuning (2026-11-06)

`scripts/tune_dfl.py`: coordinate search over 11 tracker / decoder parameters on degraded DFL,
official-definition target, conservative player-only dead-ball model. Chosen on 4 matches
(J03WMX, J03WOH, J03WPY, J03WR9), reported on 3 held-out matches (J03WN1, J03WOY, J03WQQ).
Round 1 from the v1 defaults; round 2 extended the four parameters that hit the grid edge
(both penalties, both minimum durations) and found the optimum inside the range.

| | tune (4) | **held-out (3)** | all 7 |
|---|---|---|---|
| v1 defaults | 0.419 | 0.371 | 0.398 |
| after round 1 | 0.467 | 0.412 | 0.443 |
| after round 2 (new defaults) | 0.472 | **0.416** | 0.447 |

New defaults (in `PipelineConfig`): touch radius 1.0 m, tracker T_control 1.0 s, H<->A and D
switch penalties 25 nats, minimum durations 2 s (H/A) and 5 s (D), alive threshold P(D) < 0.7.
Re-scored with `eval_dfl.py` (all 7, degraded): official target 0.442, raw DFL target 0.319,
baseline 0.169 / 0.165.

Observations
- The held-out gain (+0.045) matches the tuning gain (+0.053): no sign of overfitting.
- The tuned pipeline predicts *fewer* spells than the truth (1303 vs 2024) and scores higher for
  it: short true possessions are rarely matched at IoU >= 0.3, so it pays to skip them rather than
  guess. Round 2 is within the ~0.04 spellF1 noise band, so the parameters have plateaued; the next
  gains must come from better evidence (dead-ball boundaries, possession on ball-less frames), not
  from tuning.
- These numbers are on DFL with guessed degradation. Re-tune once the official data is in and the
  degradation is calibrated.

## Boundary alignment (2026-11-06) -- tried, negligible on DFL

Tuned pipeline, degraded DFL, official target. Current D-boundary error: start median |offset|
0.96 s (51% within 1 s), end 0.64 s (62% within 1 s).

**Ceiling first** (snap every boundary to the true one if it is within the window -- an oracle):

| oracle snapping | spellF1 (all 7) |
|---|---|
| none | 0.447 |
| D boundaries, +/-3 s | 0.465 (+0.018) |
| H<->A boundaries, +/-3 s | 0.465 (+0.018) |
| both, +/-3 s | 0.482 (+0.035) |

**Real snapping** (`mtah.boundary`): ball crossing a line -> stoppage start; ball kicked from rest ->
stoppage end. The events are precise when they exist (out-cross median 0.16 s from the true start,
kick 0.12 s from the true end) but sparse with 41% of the ball missing: 58% of true starts and only
14% of true ends have an event within 2 s. Best setting on the tune matches gave +0.0004
(held-out 0.4164 -> 0.4167). Kept as an option, not enabled.

**Where the score is actually lost** (true spells by best same-team IoU with a prediction):

| best IoU | share | median length |
|---|---|---|
| >= 0.7 | 24% | 14.7 s |
| 0.3-0.7 (counted at some thresholds) | 28% | ~10 s |
| < 0.3 or no overlap | **49%** | **~4 s** |

Half of the true possessions are short ones (~4 s) the pipeline merges into a neighbour or misses.
That -- detecting real short possessions (quick turnovers, short restarts) -- is the next lever,
not boundary precision. On the official match, Smart Tagging set-piece times may still make
stoppage-end snapping worthwhile; re-check there.

## v2: learned possession model for short possessions (2026-11-06)

**Diagnosis.** Of the 989 true possessions v1 missed (IoU < 0.3), median length 4.0 s: the ball was
seen on 38% of their frames (57% for matched ones) and the possessing team had a median **0.04 s**
of clear touches (1.5 s for matched). The rule tracker labelled 0% of them right -- not because its
thresholds are wrong, but because it never saw the touches. Looser rules cannot fix that.

**Fix.** `PossessionModel`: gradient boosting on ID-free per-team features -- nearest-player
distance now and as rolling past/future minima, time since/until the team was near or touching
the ball (so a ball reappearing at a team's feet after a gap counts), players within 5/10 m,
ball and centroid advance in each team's attacking direction, plus the rule tracker's verdict.
Trained leave-one-match-out on DFL with home/away swap augmentation, on the predicted alive mask.
Its P(home) replaces the tracker as the decoder's emission.

Degraded DFL, official target:

| | team acc (alive frames) | held-out 3 | all 7 | spells (gt 2024) |
|---|---|---|---|---|
| baseline_smooth50 | -- | -- | 0.169 | 4932 |
| v1 (tracker, tuned) | 0.758 | 0.416 | 0.442 | 1303 |
| **v2 (learned possession)** | **0.854** | **0.514** | **0.525** | 1671 |

Raw-DFL target: v1 0.319 -> v2 0.408. Short true possessions (< 8 s) matched at IoU >= 0.3:
28% -> 46%; true possessions with IoU < 0.3: 49% -> 35%.

Decoder for v2 (`V2_DECODER`): H<->A penalty 12, D penalty 25, min durations 2 s / 2 s / 5 s,
picked on the tune matches. A further one-at-a-time search (penalties, durations, emission
temperature) moved tune by up to +0.007 and held-out by -0.005..0: plateau, kept the simpler point.

Reproduce: `uv run python scripts/eval_dfl.py --versions degraded --targets official --methods pipeline_v2`

## First pass on the official data (2026-11-06)

Data checks all pass (official alignment checks; our loader + baseline equal the shipped
`baseline_smooth50.csv` on every graded frame). The first pipeline run predicted D on 99.4% of
frames, which exposed two bugs in how DFL was made to look like the official data:

- **DFL speed is km/h.** `io_dfl` now divides by 3.6 (DFL median 5.3 -> 1.46 m/s; official 1.41).
  The dead-ball model learned "everyone at 5 m/s = play", so the official data looked dead.
- **The degradation lost players and added noise the real data does not have.** Tracklets were
  assigned to slots player by player instead of in time order, overwriting each other
  (15.3 players seen per frame vs 20.5 official); and 0.3 m white jitter was added although the
  official positions are smoother than DFL's. Now calibrated on the official tracking: see
  `mtah/degrade.py` (5% player drop, lognormal tracklets median 35 s / mean 107 s, no jitter).

After the fix, re-evaluated on degraded DFL (official target, LOMO, all 7):

| | spellF1 | macroF1 |
|---|---|---|
| baseline_smooth50 | 0.231 (official full match: 0.2433) | 0.393 |
| pipeline_v2 | **0.559** | 0.846 |

The baseline now scores on degraded DFL about what it scores on the real match — a sign the
degradation is realistic. On the official public 5 minutes (pipeline check only, not used to
choose anything): v2 spellF1 0.581 / macroF1 0.851 vs baseline 0.326 / 0.621; 14 of 17
possessions predicted. Over the full match v2 predicts D on 48% of graded frames (public window:
21% vs 20% true) — worth a look in the video.
