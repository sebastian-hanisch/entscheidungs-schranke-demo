"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1 und n = 2, Gleichstände, inklusive Grenzen, Randwerte der Regler."""

import math

import numpy as np

import esc_constants as C
import esc_conformal as K
import esc_evaluation as ev
import esc_presets as P
from esc_grid import Grid


def test_exactly_two_calibration_cases():
    """n = 2: k = ceil(3 * 0,5) = 2 bei Niveau 0,5 (der größere Wert), bei Niveau 0,3 k = ceil(0,9) = 1 (der kleinere); Niveau 0,7 liegt mit k = ceil(2,1) = 3 über n."""
    assert K.conformal_quantile([4.0, 1.0], 0.5) == 4.0
    assert K.conformal_quantile([4.0, 1.0], 0.3) == 1.0
    assert K.conformal_quantile([4.0, 1.0], 0.7) == math.inf


def test_single_calibration_case_in_a_full_calibration(grid5):
    rng = np.random.default_rng(0)
    Chat, Cc = rng.uniform(1, 3, size=(1, 40)), rng.uniform(1, 3, size=(1, 40))
    cal = K.calibrate(grid5, Cc, Chat, 0.5, grid5.steps)          # Niveau 0,5 mit n = 1: k = 1 (endlich); Bonferroni-Niveau 0,9375: k = 2 > 1 (unendlich)
    assert np.isfinite(cal.q_path) and np.all(np.isfinite(cal.q_edge)) and np.all(np.isinf(cal.q_edge_bonf))
    b = K.upper_bounds(grid5, cal, Chat)
    assert np.isinf(b.b).all() and not np.isnan(b.b).any()


def test_all_scores_equal_gives_the_value_for_every_level():
    for level in (0.1, 0.5, 0.9, 0.99):
        assert K.conformal_quantile([2.5] * 200, level) == 2.5


def test_level_limits_of_the_hand_slider_are_inclusive():
    """Niveau 0,5 und 0,95 sind erlaubt und rechnen: 0,5 -> k = 3; 0,95 -> k = ceil(5,7) = 6 > 5 (unendlich)."""
    lo, hi = ev.hand_numbers(C.LEVEL_MIN), ev.hand_numbers(C.LEVEL_MAX)
    assert lo["k"] == 3 and lo["q"] == 0.9 and hi["k"] == 6 and hi["infinite"]
    assert ev.hand_numbers(0.8)["q"] == 2.0


def test_every_slider_step_of_the_hand_example_has_a_defined_result():
    steps = int(round((C.LEVEL_MAX - C.LEVEL_MIN) / C.LEVEL_STEP))
    ks = []
    for i in range(steps + 1):
        h = ev.hand_numbers(round(C.LEVEL_MIN + i * C.LEVEL_STEP, 6))
        ks.append(h["k"])
        assert (h["infinite"] and h["k"] > 5) or (not h["infinite"] and h["k"] <= 5)
    assert ks == [3, 4, 4, 4, 5, 5, 5, 6, 6, 6]       # 6 * Niveau aufgerundet: 3; 3,3; 3,6; 3,9; 4,2; 4,5; 4,8; 5,1; 5,4; 5,7


def test_smallest_grid_chooses_and_bounds_without_error():
    g = Grid(2)
    rng = np.random.default_rng(1)
    Chat, Cc = rng.uniform(1, 3, size=(30, 4)), rng.uniform(1, 3, size=(30, 4))
    cal = K.calibrate(g, Cc, Chat, 0.8, g.steps)
    b = K.upper_bounds(g, cal, Chat)
    assert b.W.shape == (30, 4) and set(b.W.sum(axis=1)) == {2.0}


def test_slider_boundaries_of_the_permalink_snap():
    spec = P.SETTING_SPECS
    assert P.snap(spec["level_slider"], 0.0) == C.LEVEL_MIN and P.snap(spec["level_slider"], 2.0) == C.LEVEL_MAX
    assert P.snap(spec["level_slider"], 0.82) == 0.8 and P.snap(spec["level_slider"], 0.83) == 0.85
    assert P.snap(spec["noise_slider"], 0.1) == 0.25 and P.snap(spec["noise_slider"], 0.6) == 0.5 and P.snap(spec["noise_slider"], 0.7) == 0.75
    assert P.snap(spec["deg_slider"], 2) == 1 and P.snap(spec["deg_slider"], 3) == 4 and P.snap(spec["deg_slider"], 5) == 4 and P.snap(spec["deg_slider"], 99) == 6
    assert P.snap(spec["seed_input"], -5) == 0 and P.snap(spec["seed_input"], 10**9) == C.SEED_MAX
    assert P.snap(spec["case_slider"], 0) == C.CASE_MIN and P.snap(spec["case_slider"], 500) == C.CASE_MAX


def test_next_case_wraps_and_handles_the_empty_list():
    assert ev.next_case([], 5) is None
    assert ev.next_case([3, 9, 40], 1) == 3 and ev.next_case([3, 9, 40], 3) == 9 and ev.next_case([3, 9, 40], 40) == 3 and ev.next_case([3, 9, 40], 99) == 3


def test_a_world_with_noise_zero_and_linear_truth_is_not_offered():
    """Rauschstufe 0 ist kein Reglerwert: bei Grad 1 wäre jede Schranke Rundungsrauschen (Breite 1e-15, Überdeckung bedeutungslos)."""
    assert 0.0 not in C.NOISE_CHOICES and min(C.NOISE_CHOICES) > 0
