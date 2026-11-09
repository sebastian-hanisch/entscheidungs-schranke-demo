"""Kostenwelt (esc_world): Formeln von Hand, Reproduzierbarkeit, getrennte Teilströme für Training, Kalibrierung und Test, Rauschbereich."""

import numpy as np
import pytest

from esc_world import TAG_CAL, TAG_TEST, TAG_TRAIN, WorldSpec, draw_data, draw_weights, make_dataset, mean_costs


def test_mean_costs_by_hand():
    """p = 1, ein Bogen mit Gewicht 1: m = (x / 1 + 3)^deg / 3,5^deg + 1. Bei x = 0.5 und deg 1: 3,5 / 3,5 + 1 = 2; deg 2: (3,5/3,5)^2 + 1 = 2.
    Bei x = -0.5 und deg 2: (2,5 / 3,5)^2 + 1 = 1 + 25/49. Gewicht 0 (zweiter Bogen): m = (3/3,5)^deg + 1."""
    B = np.array([[1.0], [0.0]])
    X = np.array([[0.5], [-0.5]])
    m1 = mean_costs(B, X, WorldSpec(p=1, deg=1, noise=0.0))
    assert m1[0, 0] == pytest.approx(2.0) and m1[1, 0] == pytest.approx(2.5 / 3.5 + 1)
    m2 = mean_costs(B, X, WorldSpec(p=1, deg=2, noise=0.0))
    assert m2[0, 0] == pytest.approx(2.0) and m2[1, 0] == pytest.approx(1 + 25 / 49)
    assert m2[0, 1] == pytest.approx((3 / 3.5) ** 2 + 1)


def test_weights_are_zero_or_one_with_about_half_ones():
    B = draw_weights(40, WorldSpec(p=5), 11)
    assert B.shape == (40, 5) and set(np.unique(B)) <= {0.0, 1.0}
    assert 0.3 < B.mean() < 0.7


def test_same_seed_same_data_and_the_three_streams_differ():
    spec = WorldSpec(deg=6, noise=0.5)
    a, b = make_dataset(spec, 40, 50, 40, 30, 5), make_dataset(spec, 40, 50, 40, 30, 5)
    for name in ("X_train", "C_train", "X_cal", "C_cal", "X_test", "C_test", "B"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert not np.array_equal(a.X_train[:30], a.X_test) and not np.array_equal(a.X_cal[:30], a.X_test) and not np.array_equal(a.X_train[:40], a.X_cal)
    assert not np.array_equal(make_dataset(spec, 40, 50, 40, 30, 6).C_cal, a.C_cal)
    assert len({TAG_TRAIN, TAG_CAL, TAG_TEST}) == 3


def test_calibration_and_test_cases_come_from_the_same_distribution():
    """Austauschbarkeit der Kalibrier- und Testfälle: gleiche Verteilung der Kosten (Mittelwerte und Streuung stimmen innerhalb der Stichprobenschwankung überein)."""
    data = make_dataset(WorldSpec(deg=4, noise=0.5), 40, 100, 20000, 20000, 3)
    assert abs(data.C_cal.mean() - data.C_test.mean()) < 0.03 * data.C_test.mean()
    assert abs(data.C_cal.std() - data.C_test.std()) < 0.05 * data.C_test.std()


def test_without_noise_observed_equals_mean_and_with_noise_ratio_is_inside_the_band():
    spec0 = WorldSpec(noise=0.0)
    B = draw_weights(40, spec0, 3)
    X0, C0 = draw_data(B, spec0, 100, 3, TAG_TRAIN)
    assert np.array_equal(C0, mean_costs(B, X0, spec0))
    spec5 = WorldSpec(noise=0.5)
    X1, C1 = draw_data(B, spec5, 500, 3, TAG_TRAIN)
    ratio = C1 / mean_costs(B, X1, spec5)
    assert ratio.min() >= 0.5 and ratio.max() <= 1.5 and abs(ratio.mean() - 1.0) < 0.01


def test_noise_is_independent_between_arcs():
    """Das Rauschen ist je Bogen und Fall unabhängig: die Korrelation der Faktoren zweier Bögen liegt bei null (Streuung 1 / sqrt(n) bei n = 20000)."""
    spec = WorldSpec(deg=1, noise=0.5)
    B = draw_weights(40, spec, 1)
    X, C = draw_data(B, spec, 20000, 1, TAG_TEST)
    eps = C / mean_costs(B, X, spec)
    corr = np.corrcoef(eps[:, 0], eps[:, 1])[0, 1]
    assert abs(corr) < 0.05


def test_costs_are_positive_in_the_standard_worlds():
    for deg in (1, 4, 6):
        for noise in (0.25, 0.5, 0.75):
            data = make_dataset(WorldSpec(deg=deg, noise=noise), 40, 300, 300, 300, 2)
            assert data.C_train.min() > 0 and data.C_cal.min() > 0 and data.C_test.min() > 0
