"""Orakel-Tests (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): Löser, Konformquantil, Überdeckung und die drei Schranken gegen unabhängige Rechenwege.

1. Der Löser (Rückwärtsrechnung, kopiert aus Stück 3) gegen die Aufzählung der 70 Wege (`conftest.independent_paths`, andere Konstruktion als `Grid.paths`).
2. Der Rang k = ceil((n + 1) * Niveau) gegen exakte Bruchrechnung (`fractions.Fraction` aus der Dezimaldarstellung des Niveaus): kein Gleitkommarest kann ihn verschieben.
3. Das Konformquantil gegen die Aufzählung der Rangstatistik: der kleinste Wert v, für den mindestens k Scores <= v zählen (Zählung statt Sortieren).
4. Die Überdeckung gegen eine geschlossene Formel: einfaches Modell mit zwei Wegen und Normalrauschen. Gegeben die Kalibrierung ist die Überdeckung Phi((q - mu) / sigma) (Gaußsche Verteilungsfunktion über `math.erf`);
   im Mittel über viele Kalibrierungen ist sie k / (n + 1) (Monte-Carlo-Band), und sie ist beta-verteilt mit den Parametern k und n + 1 - k (scipy).
5. Die drei Schranken (Summe der Kantenschranken, mit Bonferroni, konform auf den Wegkosten) auf einem kleinen Gitter gegen eine reine Python-Schleife mit eigener Aufzählung der Wege.
6. KQ gegen die Normalengleichungen.

Im Scratch-Lauf (nicht im Repo) wurde 2. für n = 0 bis 600 und alle Niveaus in Schritten von 0,001 (600399 Prüfungen), 3. an 5000 Zufallsfolgen mit 8 Niveaus (40000 Prüfungen) und 5. an 200 Zufallsinstanzen auf Gittern von 2 x 2 bis 4 x 4 (größte Abweichung 7·10⁻¹⁵, keine Abweichung bei unendlichen Schranken) ohne Befund wiederholt.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

import esc_conformal as K
from conftest import independent_solve
from esc_grid import Grid, PathSet
from esc_training import add_intercept, fit_least_squares


def test_solver_equals_enumeration_on_random_instances(grid5):
    rng = np.random.default_rng(0)
    C = rng.uniform(0.5, 5.0, size=(60, 40))
    W, z = grid5.solve(C)
    Wo, zo = independent_solve(grid5, C)
    assert np.abs(z - zo).max() < 1e-12
    assert np.array_equal(W, Wo)                      # stetige Kosten: der beste Weg ist eindeutig


def test_solver_equals_enumeration_with_ties_on_integer_costs(grid5):
    rng = np.random.default_rng(1)
    C = rng.integers(0, 3, size=(60, 40)).astype(float)
    W, z = grid5.solve(C)
    Wo, zo = independent_solve(grid5, C)
    assert np.array_equal(z, zo) and np.array_equal(W, Wo)


def test_rank_equals_exact_fractions():
    for n in range(0, 301):
        for hundredth in range(1, 100):
            level = hundredth / 100
            exact = math.ceil(Fraction(str(level)) * (n + 1))
            assert K.conformal_rank(n, level) == exact, (n, level)
    for n in (0, 1, 5, 19, 20, 99, 100, 999, 1000):
        for k in range(1, 40):
            level = round(k / 40, 6) if k < 40 else 0.975
            assert K.conformal_rank(n, level) == math.ceil(Fraction(str(level)) * (n + 1)), (n, level)


def _count_quantile(scores, level):
    """Rangstatistik durch Zählen: der kleinste vorkommende Wert v mit #{s <= v} >= k (k aus exakter Bruchrechnung); unendlich bei k > n."""
    n = len(scores)
    k = math.ceil((level if isinstance(level, Fraction) else Fraction(str(level))) * (n + 1))
    if k > n:
        return math.inf
    return min(v for v in scores if sum(1 for s in scores if s <= v) >= k)


def test_quantile_equals_the_enumerated_order_statistic():
    rng = np.random.default_rng(2)
    for _ in range(300):
        n = int(rng.integers(1, 60))
        scores = [round(float(v), 2) for v in rng.normal(size=n)]       # gerundet: absichtlich Gleichstände
        for level in (0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99):
            assert K.conformal_quantile(scores, level) == _count_quantile(scores, level), (scores, level)


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_path_calibrations(n, level, repeats, mu=2.0, sigma=1.5, seed=3):
    """Das einfache Modell: Vorhersage ĉ = (4, 3) fest, Weg 2 wird immer gewählt, Score s = c2 - 3 ~ N(mu, sigma^2) (wahre Kosten des Weges 5 + sigma-Rauschen). Gibt je Kalibrierung q und die geschlossene Überdeckung zurück."""
    rng = np.random.default_rng(seed)
    solver = PathSet(np.eye(2))
    Chat = np.tile([4.0, 3.0], (n, 1))
    qs, covs = [], []
    for _ in range(repeats):
        C = np.column_stack([np.full(n, 3.0), 3.0 + mu + sigma * rng.standard_normal(n)])
        q = K.calibrate(solver, C, Chat, level, arcs_per_path=1).q_path
        qs.append(q)
        covs.append(1.0 if math.isinf(q) else _phi((q - mu) / sigma))
    return np.array(qs), np.array(covs)


@pytest.mark.parametrize("n,level", [(5, 0.5), (20, 0.9), (100, 0.9), (50, 0.8)])
def test_mean_coverage_over_calibrations_equals_k_over_n_plus_1(n, level):
    """Geschlossene Überdeckung je Kalibrierung, gemittelt über 3000 Kalibrierungen: k / (n + 1) innerhalb des Monte-Carlo-Bandes. Die Standardabweichung der Überdeckung ist höchstens 0,19
    (Beta(3, 3) bei n = 5), der Standardfehler des Mittels also höchstens 0,0035; das Band 0,012 ist mehr als drei Standardfehler."""
    _, covs = _two_path_calibrations(n, level, 3000)
    assert abs(covs.mean() - K.guaranteed_coverage(n, level)) < 0.012


def test_coverage_given_the_calibration_is_beta_distributed():
    """n = 20, Niveau 0,9: k = 19, die Überdeckung einer festen Kalibrierung ist Beta(19, 2) (Scores ohne Gleichstand). Mittelwert 19/21, Anteil unter 0,9 aus der Beta-Verteilungsfunktion."""
    stats = pytest.importorskip("scipy.stats")
    _, covs = _two_path_calibrations(20, 0.9, 4000, seed=4)
    assert abs(covs.mean() - 19 / 21) < 0.006
    assert abs(float(np.mean(covs < 0.9)) - float(stats.beta.cdf(0.9, 19, 2))) < 0.03
    assert abs(covs.std() - math.sqrt(stats.beta.var(19, 2))) < 0.01


def test_coverage_of_a_large_calibration_is_close_to_the_target_in_every_draw():
    """n = 1000: die Überdeckung einer festen Kalibrierung liegt innerhalb von 5 Standardabweichungen von Beta(901, 100) (Standardabweichung 0,0095)."""
    _, covs = _two_path_calibrations(1000, 0.9, 200, seed=5)
    assert abs(covs.mean() - 901 / 1001) < 0.003
    assert covs.min() > 901 / 1001 - 5 * 0.0095 and covs.max() < 901 / 1001 + 5 * 0.0095


def test_unreachable_level_gives_an_infinite_quantile_and_full_closed_form_coverage():
    qs, covs = _two_path_calibrations(5, 0.9, 50)
    assert np.all(np.isinf(qs)) and np.all(covs == 1.0)


def _loop_bounds(h, C_cal, Chat_cal, Chat_test, level):
    """Die drei Schranken in reinen Python-Schleifen, mit eigener Wegaufzählung (conftest.independent_solve) und Quantilen per Zählung."""
    grid = Grid(h)
    m = grid.steps
    d = grid.d
    ln = len(C_cal)

    def kth(values, lv):
        return _count_quantile([float(v) for v in values], lv)

    Wc, _ = independent_solve(grid, Chat_cal)
    path_s = [sum((C_cal[i][e] - Chat_cal[i][e]) * Wc[i][e] for e in range(d)) for i in range(ln)]
    q_path = kth(path_s, level)
    level_b = 1 - (1 - Fraction(str(level))) / m      # exakt als Bruch: 1 - 0,2 / 6 = 29 / 30, nicht die Gleitkommazahl 0,96666666666666667
    q_e = [kth([C_cal[i][e] - Chat_cal[i][e] for i in range(ln)], level) for e in range(d)]
    q_eb = [kth([C_cal[i][e] - Chat_cal[i][e] for i in range(ln)], level_b) for e in range(d)]
    Wt, _ = independent_solve(grid, Chat_test)
    out = []
    for r in range(len(Chat_test)):
        pred = sum(Chat_test[r][e] * Wt[r][e] for e in range(d))
        out.append((pred, pred + sum(q_e[e] for e in range(d) if Wt[r][e] == 1.0), pred + sum(q_eb[e] for e in range(d) if Wt[r][e] == 1.0), pred + q_path))
    return np.array(out), Wt


@pytest.mark.parametrize("h,seed", [(3, 0), (3, 1), (4, 2)])
def test_three_bounds_equal_the_loop_implementation(h, seed):
    rng = np.random.default_rng(seed)
    grid = Grid(h)
    d = grid.d
    n_cal, n_test = 60, 25
    Chat_cal = rng.uniform(1, 4, size=(n_cal, d))
    C_cal = Chat_cal + rng.normal(0, 0.8, size=(n_cal, d))
    Chat_test = rng.uniform(1, 4, size=(n_test, d))
    for level in (0.8, 0.9):
        cal = K.calibrate(grid, C_cal, Chat_cal, level, grid.steps)
        b = K.upper_bounds(grid, cal, Chat_test)
        ref, Wt = _loop_bounds(h, C_cal, Chat_cal, Chat_test, level)
        assert np.array_equal(b.W, Wt)
        assert np.allclose(b.pred, ref[:, 0], atol=1e-12) and np.allclose(b.a, ref[:, 1], atol=1e-12)
        assert np.allclose(b.b, ref[:, 2], atol=1e-12) and np.allclose(b.c, ref[:, 3], atol=1e-12)


def test_measured_quantities_equal_explicit_loops():
    """Überdeckung, Breite und Optimismus einer Testmenge gegen Schleifen."""
    rng = np.random.default_rng(7)
    real = rng.uniform(1, 5, size=40)
    pred = rng.uniform(1, 5, size=40)
    bound = pred + rng.uniform(0, 2, size=40)
    inside = sum(1 for r, b in zip(real, bound) if r <= b)
    assert K.coverage(real, bound) == pytest.approx(inside / 40)
    assert K.width_percent(bound, pred) == pytest.approx(100 * (sum(bound) / 40 - sum(pred) / 40) / (sum(pred) / 40))
    assert K.optimism_percent(real, pred) == pytest.approx(100 * (sum(real) - sum(pred)) / sum(pred))


def test_least_squares_equals_the_normal_equations():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 5))
    C = rng.normal(size=(60, 7))
    Xa = add_intercept(X)
    expected = np.linalg.solve(Xa.T @ Xa, Xa.T @ C)
    assert np.abs(fit_least_squares(X, C) - expected).max() < 1e-10
