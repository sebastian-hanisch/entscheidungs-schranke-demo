"""Konforme Schranken (esc_conformal): Handbeispiel von Hand, Eigenschaften (Monotonie, Verschiebung, Skalierung), Randfälle, die drei Verfahren an einer Hand-Instanz."""

import math

import numpy as np
import pytest

import esc_conformal as K
from esc_grid import two_paths

HAND = (-0.5, 0.2, 0.9, 1.4, 2.0)


def test_handbeispiel_ranks_and_bounds():
    """n = 5: Niveau 0,5 -> k = ceil(6 * 0,5) = 3, Schranke 0,9; Niveau 0,8 -> k = ceil(4,8) = 5, Schranke 2,0; Niveau 0,9 -> k = ceil(5,4) = 6 > 5, unendlich."""
    assert [K.conformal_rank(5, lv) for lv in (0.5, 0.8, 0.9)] == [3, 5, 6]
    assert K.conformal_quantile(HAND, 0.5) == 0.9
    assert K.conformal_quantile(HAND, 0.8) == 2.0
    assert K.conformal_quantile(HAND, 0.9) == math.inf


def test_handbeispiel_is_independent_of_the_order_of_the_scores():
    assert K.conformal_quantile(HAND[::-1], 0.5) == 0.9
    assert K.conformal_quantile((0.9, 2.0, -0.5, 1.4, 0.2), 0.8) == 2.0


def test_guaranteed_coverage_by_hand():
    """k / (n + 1): n = 5 -> 3/6 = 0,5 und 5/6; bei k > n gibt es keine Schranke, die Überdeckung der unendlichen Schranke ist 1."""
    assert K.guaranteed_coverage(5, 0.5) == pytest.approx(0.5)
    assert K.guaranteed_coverage(5, 0.8) == pytest.approx(5 / 6)
    assert K.guaranteed_coverage(5, 0.9) == 1.0
    assert K.guaranteed_coverage(1000, 0.9) == pytest.approx(901 / 1001)


def test_one_calibration_case_and_none():
    assert K.conformal_rank(1, 0.5) == 1 and K.conformal_quantile([3.25], 0.5) == 3.25   # k = ceil(2 * 0,5) = 1: der kleinste und einzige Wert
    assert K.conformal_quantile([3.25], 0.9) == math.inf                                  # k = ceil(1,8) = 2 > 1
    assert K.conformal_quantile([], 0.5) == math.inf                                      # keine Kalibrierfälle: keine Schranke


def test_rank_boundary_is_inclusive_and_free_of_float_residue():
    """(n + 1) * Niveau genau ganzzahlig: k ist diese Zahl (kein Aufrunden durch einen Gleitkommarest). n = 9, Niveau 0,9: 10 * 0,9 = 9 -> k = 9 = n (Maximum); n = 19, Niveau 0,95: k = 19."""
    assert K.conformal_rank(9, 0.9) == 9 and K.conformal_quantile(range(9), 0.9) == 8.0
    assert K.conformal_rank(19, 0.95) == 19 and K.conformal_quantile(range(19), 0.95) == 18.0
    assert K.conformal_rank(9, 0.95) == 10 and K.conformal_quantile(range(9), 0.95) == math.inf   # 10 * 0,95 = 9,5 -> 10 > 9
    assert K.conformal_rank(99, 0.9) == 90                                                       # 100 * 0,9 = 90, nicht 91 (Rest 90,00000000000001 wäre ein Fehler)


def test_quantile_is_monotone_in_the_level():
    rng = np.random.default_rng(0)
    s = rng.normal(size=47)
    qs = [K.conformal_quantile(s, lv) for lv in np.linspace(0.05, 0.97, 60)]
    assert all(a <= b for a, b in zip(qs, qs[1:]))


def test_shifting_all_scores_by_c_shifts_the_bound_by_c_and_scaling_scales_it():
    rng = np.random.default_rng(1)
    s = rng.normal(size=33)
    for lv in (0.5, 0.8, 0.9):
        q = K.conformal_quantile(s, lv)
        assert K.conformal_quantile(s + 7.25, lv) == pytest.approx(q + 7.25)
        assert K.conformal_quantile(2.5 * s, lv) == pytest.approx(2.5 * q)


def test_ties_do_not_change_the_rank_rule():
    assert K.conformal_quantile([1.0, 1.0, 1.0, 1.0, 1.0], 0.8) == 1.0
    assert K.conformal_quantile([0.0, 1.0, 1.0, 1.0, 2.0], 0.5) == 1.0        # k = 3: der dritte kleinste Wert ist eine der drei 1,0
    assert K.conformal_quantile([0.0, 1.0, 1.0, 1.0, 2.0], 0.8) == 2.0        # k = 5


@pytest.mark.parametrize("level", [0.0, 1.0, -0.2, 1.5])
def test_level_outside_zero_one_raises(level):
    with pytest.raises(ValueError):
        K.conformal_rank(10, level)
    with pytest.raises(ValueError):
        K.conformal_quantile([1.0, 2.0], level)


def test_negative_number_of_cases_raises():
    with pytest.raises(ValueError):
        K.conformal_rank(-1, 0.5)


def test_path_scores_by_hand():
    """Zwei Wege mit je einem Bogen, c = (3, 5), ĉ = (4, 3): gewählt wird Weg 2 (3 < 4), tatsächlich 5, vorhergesagt 3, Score 2."""
    solver = two_paths()
    C, Chat = np.array([[3.0, 5.0]]), np.array([[4.0, 3.0]])
    W, _ = solver.solve(Chat)
    assert W.tolist() == [[0.0, 1.0]]
    assert K.path_scores(W, C, Chat).tolist() == [2.0]


def _hand_instance():
    """Fünf Kalibrierfälle mit ĉ = (4, 3): Weg 2 wird gewählt; die tatsächlichen Kosten des zweiten Bogens 2,5; 3,2; 3,9; 4,4; 5,0 geben die Scores -0,5; 0,2; 0,9; 1,4; 2,0, der erste Bogen kostet immer 3."""
    C_cal = np.array([[3.0, 2.5], [3.0, 3.2], [3.0, 3.9], [3.0, 4.4], [3.0, 5.0]])
    Chat_cal = np.tile([4.0, 3.0], (5, 1))
    return C_cal, Chat_cal


def test_calibrate_and_bounds_on_the_hand_instance():
    solver = two_paths()
    C_cal, Chat_cal = _hand_instance()
    cal = K.calibrate(solver, C_cal, Chat_cal, 0.8, arcs_per_path=1)
    assert cal.q_path == 2.0 and cal.n == 5
    assert cal.q_edge.tolist() == [-1.0, 2.0]           # Bogen 1: 3 - 4 = -1 in jedem Fall; Bogen 2: dieselben Scores wie der Weg
    assert cal.q_edge_bonf.tolist() == [-1.0, 2.0]       # ein Bogen je Weg: Bonferroni ändert das Niveau nicht
    b = K.upper_bounds(solver, cal, np.array([[4.0, 3.0]]))
    assert b.W.tolist() == [[0.0, 1.0]] and b.pred.tolist() == [3.0]
    assert b.a.tolist() == [5.0] and b.b.tolist() == [5.0] and b.c.tolist() == [5.0]   # 3 + 2


def test_summing_edge_quantiles_with_two_arcs_per_path_by_hand():
    """Zwei Bögen je Weg (Bonferroni-Teiler 2): Weg 1 = Bögen (1, 2), Weg 2 = Bögen (3, 4). Die Bogenfehler sind hier so gebaut, dass die Summe der Quantile über dem Quantil des Wegscores liegt."""
    from esc_grid import PathSet

    solver = PathSet(np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]))
    Chat = np.tile([1.0, 1.0, 5.0, 5.0], (5, 1))                        # Weg 1 wird gewählt (Vorhersage 2 gegen 10)
    err = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])   # Fehler der Bögen 1 und 2 je Fall
    C = Chat + np.hstack([err, np.zeros((5, 2))])
    cal = K.calibrate(solver, C, Chat, 0.8, arcs_per_path=2)
    # Wegscore = Summe der beiden Bogenfehler: 0, 1, 1, 2, 2 -> k = 5: Quantil 2; Bogenquantile je Bogen: 2 und 2 (k = 5) -> Summe 4
    assert cal.q_path == 2.0 and cal.q_edge[:2].tolist() == [2.0, 2.0]
    b = K.upper_bounds(solver, cal, Chat[:1])
    assert b.pred.tolist() == [2.0] and b.c.tolist() == [4.0] and b.a.tolist() == [6.0]   # Summe 2 + 2 + 2 gegen Wegquantil 2 + 2


def test_infinite_quantiles_give_an_infinite_bound_and_never_nan():
    """Bonferroni mit zu wenig Kalibrierfällen: alle Bogenquantile unendlich; die Summe ist unendlich und nicht nan (0 * unendlich bei Bögen außerhalb des Weges)."""
    solver = two_paths()
    C_cal, Chat_cal = _hand_instance()
    cal = K.calibrate(solver, C_cal, Chat_cal, 0.8, arcs_per_path=2)   # Niveau 1 - 0,2 / 2 = 0,9: k = 6 > 5
    assert np.all(np.isinf(cal.q_edge_bonf)) and np.isfinite(cal.q_path)
    b = K.upper_bounds(solver, cal, np.array([[4.0, 3.0]]))
    assert b.b.tolist() == [math.inf] and not np.isnan(b.b).any()


def test_zero_noise_gives_the_bound_prediction_plus_zero():
    """Rauschen null und exakte Vorhersage: alle Scores sind 0, die Schranke ist die Vorhersage, und die tatsächlichen Kosten liegen genau darauf (Überdeckung 1, Breite 0)."""
    solver = two_paths()
    Chat_cal = np.tile([4.0, 3.0], (10, 1))      # n = 10: k = ceil(11 * 0,9) = 10 <= n
    cal = K.calibrate(solver, Chat_cal.copy(), Chat_cal, 0.9, arcs_per_path=1)
    assert cal.q_path == 0.0 and np.all(cal.q_edge == 0.0)
    b = K.upper_bounds(solver, cal, np.array([[4.0, 3.0], [1.0, 6.0]]))
    assert b.c.tolist() == b.pred.tolist() == [3.0, 1.0]
    assert K.coverage(b.pred, b.c) == 1.0 and K.width_percent(b.c, b.pred) == 0.0


def test_negative_scores_make_the_bound_lower_than_the_prediction():
    """Liegen alle Kalibrierwerte unter der Vorhersage (Score < 0), ist die Schranke kleiner als die Vorhersage; die Breite ist dann negativ."""
    solver = two_paths()
    Chat_cal = np.tile([4.0, 3.0], (5, 1))
    C_cal = Chat_cal - 0.5
    cal = K.calibrate(solver, C_cal, Chat_cal, 0.8, arcs_per_path=1)
    b = K.upper_bounds(solver, cal, np.array([[4.0, 3.0]]))
    assert cal.q_path == pytest.approx(-0.5) and b.c.tolist() == [2.5] and K.width_percent(b.c, b.pred) < 0


def test_coverage_width_and_optimism_by_hand():
    pred = np.array([2.0, 4.0])
    real = np.array([3.0, 4.0])
    bound = np.array([3.0, 6.0])
    assert K.coverage(real, bound) == 1.0                                # 3 <= 3 und 4 <= 6: die Grenze zählt mit
    assert K.coverage(real + 0.1, bound) == 0.5
    assert K.width_percent(bound, pred) == pytest.approx(100 * 3 / 6)    # mittlere Breite 1,5 gegen mittlere Vorhersage 3: 50 %
    assert K.optimism_percent(real, pred) == pytest.approx(100 * 0.5 / 3)    # tatsächlich 3,5, vorhergesagt 3,0: +16,67 %
