import numpy as np
from mtah.schema import AWAY, HOME
from mtah.tracker import TrackerConfig, assign_possession

CFG = TrackerConfig(t_control=2.0, n_players=3, distinct_m=5.0)


def ep(a, b, team, x=50.0, y=34.0):
    return (a, b, team, x, y)


def test_deflection_is_absorbed():
    # home plays, away touches for 0.4 s at one place, home again -> one home possession (R4)
    eps = [ep(0, 50, HOME), ep(60, 70, AWAY), ep(80, 200, HOME)]
    team = assign_possession(eps, 0, 250, 25, CFG, HOME)
    assert (team == HOME).all()


def test_controlled_change_dated_to_first_touch():
    # away keeps the ball 3 s -> change confirmed and dated back to its first touch (R3, R5)
    eps = [ep(0, 50, HOME), ep(100, 175, AWAY)]
    team = assign_possession(eps, 0, 250, 25, CFG, HOME)
    assert (team[:100] == HOME).all() and (team[100:] == AWAY).all()


def test_quick_passes_confirm_change():
    # away: three short touches at different places within 1.5 s -> two passes, confirmed
    eps = [ep(0, 50, HOME), ep(60, 63, AWAY, 10, 10), ep(75, 78, AWAY, 25, 10),
           ep(90, 95, AWAY, 40, 10), ep(100, 110, HOME)]
    team = assign_possession(eps, 0, 120, 25, CFG, HOME)
    assert team[70] == AWAY and team[96] == AWAY


def test_restart_team_owns_short_first_spell():
    eps = [ep(0, 5, AWAY), ep(20, 200, HOME)]
    team = assign_possession(eps, 0, 250, 25, CFG, HOME)
    assert team[0] == AWAY
