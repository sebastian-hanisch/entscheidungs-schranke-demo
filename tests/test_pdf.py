"""PDF-Export: erzeugt ein gültiges PDF für jede Zelle der Auswahl (Umlaute und Sonderzeichen gehen durch die latin-1-Ersetzung)."""

import pytest

import esc_constants as C
import esc_pdf_export as X
from esc_evaluation import Settings
from esc_runs import RunSpec, run_world

SMALL = RunSpec(n_train=200, n_cal=100, n_test=200)


def test_ascii_replacement_keeps_latin1_and_maps_symbols():
    assert X._s("Überdeckung ≥ 90 % → ⌈k⌉ − ĉ") == "Überdeckung >= 90 % -> ceil(k) - c^"
    assert X._s("∞") == "?"            # nicht ersetzbare Zeichen werden zu einem Fragezeichen, kein Absturz


@pytest.mark.parametrize("deg,noise", [(1, 0.25), (4, 0.5), (6, 0.75)])
def test_pdf_is_valid_for_every_corner_of_the_choice_grid(deg, noise):
    settings = Settings(deg=deg, noise=noise, level=0.9)
    pdf = X.build_pdf(settings, run_world(SMALL.__class__(deg=deg, noise=noise, n_train=200, n_cal=100, n_test=200)))
    assert pdf[:5] == b"%PDF-" and len(pdf) > 1500


def test_pdf_for_the_hand_example_with_an_infinite_bound_does_not_fail():
    pdf = X.build_pdf(Settings(level=C.LEVEL_MAX), run_world(SMALL))
    assert pdf[:5] == b"%PDF-"
