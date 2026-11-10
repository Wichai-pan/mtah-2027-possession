# Possession rules we implement

Source: the official [CHALLENGE.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md).
Each rule cites the definition it comes from. Parameters marked *(open)* are not fixed by the
definition; they are checked qualitatively against the public 5 minutes once the data arrives
(questions at the bottom), never tuned on it.

## States

| code | meaning |
|---|---|
| H / A | ball alive, home / away in possession |
| D | ball not in play (referee's decision) |

**R1 — While the ball is alive, exactly one team has possession.** There is no "contested",
"loose" or "unknown" alive state. (*"While it is, one of the two teams is considered to have
possession."*, *"Three states, not more."*)

## Possession changes

**R2 — Possession is sticky.** While the ball is contested or loose, it stays with the last team
that had it. Missing ball observations, balls in the air and scrambles never change possession by
themselves. (*"In phases where the ball is contested or loose, possession stays with the last team
that had it."*)

**R3 — Possession changes only on *controlled* possession by the other team.** A candidate change
(an opponent touches the ball) is confirmed when either
- the opponent team keeps the ball for at least `T_control` seconds *(open; default 2 s)*, or
- the opponent team strings together passes — at least `N_players` distinct players of that team
  touch the ball in sequence *(open; default 3, i.e. two completed passes)*.

(*"a player keeps the ball for a few seconds"*, *"a team completes more than two consecutive passes"*.)

**R4 — These are not changes of possession:** a touch without control, a deflection of a pass that
is not intercepted, an aerial duel won without subsequent control. All are unconfirmed candidates
under R3 and are absorbed into the surrounding possession. (*"What does not count"* list.)

**R5 — Confirmed changes are dated back to the controlling team's first touch.** Offline decoding
sees the whole match, so once control is confirmed the new possession starts where that team first
touched the ball *(open: first touch vs. moment of control — question Q1)*.

## Stoppages

**R6 — Dead ball is D, regardless of who will restart.** (*"When the ball is dead the answer is D."*)

**R7 — Every restart has a clear possessor.** The first possession after a D stretch belongs to the
restarting team, and is kept even if short (R3 does not apply to it). (*"Whenever play restarts
after a stoppage, there is a clear possessor."*)

**R8 — `H D H` is two possessions.** A stoppage always splits possessions. (*"`H H D D H H` is two
possessions with a stoppage between them"*.)

## What we are really predicting

**R9 — The ground truth is Metrica's corrected tracking, mechanically mapped:** D where the corrected
tracking marks the ball dead, otherwise the team it names as owner. So the operative questions are
what that tracking means by *dead* and by *owning* — answered by the public 5 minutes, not by event
data or by a human's reading of the video.

## Questions for the public 5 minutes (qualitative only)

- **Q1** Does a turnover start at the new team's first touch or once control is established?
- **Q2** Does D start when the ball crosses the line / at the whistle, and end at the restart touch?
- **Q3** What are the two very short official possessions (0.8 s, 1.3 s)? Which rule lets them exist?
- **Q4** Order of magnitude of `T_control` implied by the deflections / duels in the window.

## Using the rules on DFL

DFL's `BallPossession` flips on touches the official definition ignores (DFL spells: mean 7.2 s,
13% under 1 s; official public window: mean 14.2 s). `mtah.relabel` applies R3/R4/R7 to the DFL
labels to build an *official-definition* validation target; see the README for the numbers.
