"""Lauf einer Welt (esc_runs): Formen, Reproduzierbarkeit, Messgrößen gegen Handformeln, Kalibrierblöcke, Kontrolle mit zufälligem Weg."""

from dataclasses import replace

import numpy as np
import pytest

import esc_conformal as K
import esc_runs as R

SMALL = R.RunSpec(deg=4, noise=0.5, seed=0, n_train=300, n_cal=200, n_test=400)


@pytest.fixture(scope="module")
def run():
    return R.run_world(SMALL)


def test_shapes_and_the_chosen_path_is_a_path(run):
    assert run.Chat_cal.shape == (200, 40) and run.Chat_test.shape == (400, 40) and run.coef.shape == (6, 40)
    assert run.bounds.W.shape == (400, 40) and set(run.bounds.W.sum(axis=1)) == {8.0}
    assert run.real.shape == run.pred.shape == (400,)


def test_pred_and_real_are_the_path_sums(run):
    assert np.allclose(run.pred, np.sum(run.Chat_test * run.bounds.W, axis=1))
    assert np.allclose(run.real, np.sum(run.data.C_test * run.bounds.W, axis=1))
    assert np.array_equal(run.pred, run.bounds.pred)


def test_bounds_are_the_prediction_plus_the_calibrated_quantiles(run):
    assert np.allclose(run.bounds.c, run.pred + run.cal.q_path)
    assert np.allclose(run.bounds.a, run.pred + run.bounds.W @ run.cal.q_edge)
    assert np.allclose(run.bounds.b, run.pred + run.bounds.W @ run.cal.q_edge_bonf)


def test_bonferroni_is_not_narrower_than_the_sum_of_edge_quantiles(run):
    """Das Niveau je Bogen steigt von 0,9 auf 0,9875: jedes Quantil ist größer oder gleich, also auch die Summe."""
    assert np.all(run.cal.q_edge_bonf >= run.cal.q_edge) and np.all(run.bounds.b >= run.bounds.a)


def test_same_seed_same_result_other_seed_other_result():
    a, b = R.cell_result(R.run_world(SMALL)), R.cell_result(R.run_world(SMALL))
    assert a == b
    c = R.cell_result(R.run_world(replace(SMALL, seed=1)))
    assert c != a and c["seed"] == 1


def test_cell_result_fields_in_percent_and_equal_to_the_hand_formulas(run):
    r = R.cell_result(run)
    assert r["cov_c"] == pytest.approx(100 * np.mean(run.real <= run.pred + run.cal.q_path))
    assert r["w_c"] == pytest.approx(100 * run.cal.q_path / run.pred.mean())
    assert r["optimism"] == pytest.approx(100 * (run.real.mean() - run.pred.mean()) / run.pred.mean())
    assert 0 <= r["cov_a"] <= 100 and 0 <= r["cov_b"] <= 100 and set(r) == {"seed", "cov_a", "cov_b", "cov_c", "w_a", "w_b", "w_c", "optimism", "optimism_random"}


def test_run_seeds_is_reproducible_and_names_the_seeds():
    rows = R.run_seeds(SMALL, [3, 4])
    assert [r["seed"] for r in rows] == [3, 4] and rows == R.run_seeds(SMALL, [3, 4])
    assert rows[0] == R.cell_result(R.run_world(replace(SMALL, seed=3)))


def test_random_control_uses_other_paths_with_the_same_prediction(run):
    """Der zufällig gewählte Weg ist ein gültiger Weg (8 Bögen), liegt bei allen 70 Wegen gleichverteilt, und seine Vorhersagekosten sind nie kleiner als die des gewählten (der Löser nimmt das Minimum)."""
    assert np.all(run.pred_random >= run.pred - 1e-9)
    assert np.mean(run.pred_random > run.pred + 1e-9) > 0.9           # in fast jedem Fall ist ein zufälliger Weg teurer vorhergesagt
    paths = run.grid.paths()
    # Rückrechnung: die Kosten eines Weges aus der Wegliste gehören zu genau einem Weg je Fall
    values = run.Chat_test @ paths.T
    assert all(any(abs(values[i, j] - run.pred_random[i]) < 1e-9 for j in range(70)) for i in range(0, 400, 40))


def test_calibration_blocks_count_and_the_full_block_equals_the_run(run):
    for n, blocks in ((20, 10), (50, 4), (100, 2), (200, 1)):
        b = R.calibration_blocks(run, n)
        assert len(b["coverage"]) == blocks and len(b["width"]) == blocks
    full = R.calibration_blocks(run, 200)
    r = R.cell_result(run)
    assert full["coverage"][0] == pytest.approx(r["cov_c"], abs=100 * 2 / 400)      # dieselbe Schranke, höchstens Rundung an der Grenze
    assert full["width"][0] == pytest.approx(r["w_c"])


def test_calibration_blocks_equal_an_explicit_loop(run):
    W_cal, _ = run.grid.solve(run.Chat_cal)
    s_cal = K.path_scores(W_cal, run.data.C_cal, run.Chat_cal)
    b = R.calibration_blocks(run, 50)
    for i in range(4):
        q = K.conformal_quantile(s_cal[50 * i:50 * (i + 1)], 0.9)
        assert b["width"][i] == pytest.approx(100 * q / run.pred.mean())
        assert b["coverage"][i] == pytest.approx(100 * np.mean(run.real - run.pred <= q))


def test_block_larger_than_the_calibration_set_gives_no_block(run):
    assert R.calibration_blocks(run, 201) == {"coverage": [], "width": []}


def test_a_block_too_small_for_the_level_gives_an_infinite_width(run):
    """9 Fälle und Niveau 0,9: k = ceil(10 * 0,9) = 9 <= 9 (noch endlich); 8 Fälle: k = ceil(8,1) = 9 > 8: unendlich, Überdeckung 100 %."""
    assert np.isfinite(R.calibration_blocks(run, 9)["width"]).all()
    b = R.calibration_blocks(run, 8)
    assert np.all(np.isinf(b["width"])) and np.all(np.array(b["coverage"]) == 100.0)


def test_summarize():
    assert R.summarize([3.0, 1.0, 2.0, 10.0]) == {"median": 2.5, "min": 1.0, "max": 10.0}


def test_linear_truth_without_noise_has_zero_width_and_zero_optimism():
    """Grad 1 ohne Rauschen: KQ ist exakt (bis auf Rundung), alle Scores sind kleiner als 1e-9, die Breite ist 0 (innerhalb von Rundung), der Optimismus ebenso. Die Überdeckung ist dann bedeutungslos und wird nicht ausgewiesen."""
    r = R.run_world(R.RunSpec(deg=1, noise=0.0, seed=0, n_train=300, n_cal=100, n_test=200))
    assert abs(r.cal.q_path) < 1e-9 and abs(R.cell_result(r)["w_c"]) < 1e-9 and abs(R.cell_result(r)["optimism"]) < 1e-9
