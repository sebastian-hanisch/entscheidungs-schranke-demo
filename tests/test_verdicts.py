"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, nicht Seeds. Schwellen werden an der angezeigten Stelle geprüft (89,04 wird als 89,0 angezeigt und gilt als 89,0)."""

import math

import esc_evaluation as ev


# --- Experiment 1: Handbeispiel --------------------------------------------------------------------------------------------------------------

def test_hand_verdict_level_half():
    kind, text = ev.verdict_hand(ev.hand_numbers(0.5))
    assert kind == "success"
    assert "Rang k = ⌈(5 + 1) · 0,50⌉ = 3" in text and "der 3. kleinste Score, 0,9" in text
    assert "vorhergesagten Kosten 10,0" in text and "höchstens 10,9" in text and "mindestens 3 / 6 = 50,0 %" in text


def test_hand_verdict_level_eighty():
    kind, text = ev.verdict_hand(ev.hand_numbers(0.8))
    assert kind == "success" and "= 5:" in text and "der 5. kleinste Score, 2,0" in text and "höchstens 12,0" in text and "mindestens 5 / 6 = 83,3 %" in text


def test_hand_verdict_level_ninety_has_no_guarantee_at_n_5():
    kind, text = ev.verdict_hand(ev.hand_numbers(0.9))
    assert kind == "warning"
    assert "Rang k = ⌈(5 + 1) · 0,90⌉ = 6 liegt über n = 5" in text and "keine Garantie bei n = 5" in text and "die Schranke ist unendlich" in text
    assert "5 / 6 = 83,3 %" in text


def test_hand_numbers_fields():
    h = ev.hand_numbers(0.8)
    assert h["sorted"] == (-0.5, 0.2, 0.9, 1.4, 2.0) and h["n"] == 5 and h["k"] == 5 and not h["infinite"]
    assert h["bound"] == 12.0 and abs(h["guarantee"] - 5 / 6) < 1e-12 and abs(h["max_level"] - 5 / 6) < 1e-12
    assert ev.hand_numbers(0.9)["infinite"] and ev.hand_numbers(0.9)["bound"] == math.inf


# --- Kernfrage: drei Schranken ----------------------------------------------------------------------------------------------------------------

def test_bounds_verdict_success_names_every_number():
    kind, text = ev.verdict_bounds(98.3, 100.0, 90.1, 46.4, 113.7, 24.2, 30)
    assert kind == "success"
    assert "überdeckt 90,1 % bei einer Breite von 24,2 %" in text and "Kantenschranken 98,3 % bei 46,4 %" in text and "mit Bonferroni 100,0 % bei 113,7 %" in text
    assert "Median über 30 Zufalls-Seeds, Ziel 90 %" in text and "22,2 Prozentpunkte breiter" in text


def test_bounds_verdict_threshold_of_the_path_bound_is_one_point_inclusive():
    assert ev.verdict_bounds(95.0, 100.0, 89.04, 40.0, 90.0, 20.0, 30)[0] == "success"          # angezeigt 89,0: noch im Fenster
    assert ev.verdict_bounds(95.0, 100.0, 90.96, 40.0, 90.0, 20.0, 30)[0] == "success"          # angezeigt 91,0
    kind, text = ev.verdict_bounds(95.0, 100.0, 88.94, 40.0, 90.0, 20.0, 30)                      # angezeigt 88,9
    assert kind == "warning" and "mehr als einen Prozentpunkt vom Ziel" in text
    assert ev.verdict_bounds(95.0, 100.0, 91.06, 40.0, 90.0, 20.0, 30)[0] == "warning"          # angezeigt 91,1


def test_bounds_verdict_sum_below_target_is_a_warning():
    kind, text = ev.verdict_bounds(89.9, 100.0, 90.0, 40.0, 90.0, 20.0, 30)
    assert kind == "warning" and "Die Summe der Kantenschranken verfehlt das Ziel" in text
    assert ev.verdict_bounds(89.96, 100.0, 90.0, 40.0, 90.0, 20.0, 30)[0] == "success"          # angezeigt 90,0: erreicht
    assert ev.verdict_bounds(90.0, 100.0, 90.0, 40.0, 90.0, 20.0, 30)[0] == "success"


def test_bounds_verdict_path_bound_not_narrower_is_info():
    for w_a in (20.0, 19.0):                                                                        # gleich breit und schmaler
        kind, text = ev.verdict_bounds(95.0, 100.0, 90.0, w_a, 90.0, 20.0, 30)
        assert kind == "info" and "nicht schmaler" in text
    assert ev.verdict_bounds(95.0, 100.0, 90.0, 20.04, 90.0, 20.0, 30)[0] == "info"                # angezeigt gleich (20,0 gegen 20,0)
    assert ev.verdict_bounds(95.0, 100.0, 90.0, 20.06, 90.0, 20.0, 30)[0] == "success"             # angezeigt 20,1 gegen 20,0


# --- Kernfrage: Optimismus ----------------------------------------------------------------------------------------------------------------------

def test_optimism_verdict_positive():
    kind, text = ev.verdict_optimism(2.3, 0.8, 3.7, 0.04, 30, 4, 0.5)
    assert kind == "warning"
    assert "im Median 2,3 % zu optimistisch" in text and "Grad 4, Rauschstufe 0,50, Median über 30 Zufalls-Seeds, Spanne 0,8 bis 3,7 %" in text
    assert "zufällig gewählten Weg derselben Vorhersage liegt der Median bei 0,0 %" in text


def test_optimism_verdict_zero_and_negative():
    kind, text = ev.verdict_optimism(0.04, -0.5, 0.6, -0.1, 30, 1, 0.25)                           # angezeigt 0,0
    assert kind == "info" and "nicht optimistisch (0,0 %" in text and "bei -0,1 %" in text
    kind, text = ev.verdict_optimism(-0.3, -0.9, 0.2, 0.0, 30, 1, 0.25)
    assert kind == "info" and "im Median 0,3 % zu pessimistisch" in text


def test_optimism_trend_verdict():
    rows = [{"deg": 1, "chosen": {"median": 0.4}, "random": {"median": -0.0}}, {"deg": 4, "chosen": {"median": 2.3}, "random": {"median": 0.05}}, {"deg": 6, "chosen": {"median": 7.0}, "random": {"median": -0.1}}]
    kind, text = ev.verdict_optimism_trend(rows, 0.5, 30)
    assert kind == "warning" and "wächst mit dem Grad der Wahrheit: Grad 1: 0,4 %, Grad 4: 2,3 %, Grad 6: 7,0 %" in text
    assert "im Betrag höchstens bei 0,1 %" in text and "Rauschstufe 0,50, Median über 30 Zufalls-Seeds" in text
    rows[2]["chosen"]["median"] = 2.3                                                               # gleich weit angezeigt: wächst nicht
    kind, text = ev.verdict_optimism_trend(rows, 0.5, 30)
    assert kind == "info" and "wächst nicht in jedem Schritt" in text
    rows[2]["chosen"]["median"] = 1.0
    assert ev.verdict_optimism_trend(rows, 0.5, 30)[0] == "info"


# --- Experiment 2: diese Welt und ein Testfall --------------------------------------------------------------------------------------------------

def test_instance_verdict_names_seed_and_band():
    now = {"cov_c": 89.1, "w_c": 24.1, "cov_a": 97.3, "w_a": 46.6}
    kind, text = ev.verdict_instance(now, 7, 88.2, 92.4, 30)
    assert kind == "info"
    assert "Zufalls-Seed 7" in text and "Wegkosten 89,1 % bei einer Breite von 24,1 %" in text and "Kantenschranken 97,3 % bei 46,6 %" in text
    assert "über 30 Zufalls-Seeds lag die Überdeckung der Wegkosten-Schranke zwischen 88,2 und 92,4 %" in text


def test_case_verdict_threshold_is_inclusive_at_the_bound():
    kind, text = ev.verdict_case(5, 6.04, 8.32, 8.93, 12.19)
    assert kind == "success"and text.startswith("Testfall 5: vorhergesagt 6,04, tatsächlich 8,32;")
    assert "Schranke auf den Wegkosten 8,93, Summe der Kantenschranken 12,19" in text and "nicht über der Wegkosten-Schranke" in text
    assert ev.verdict_case(5, 6.0, 8.93, 8.93, 12.0)[0] == "success"                              # genau auf der Schranke
    kind, text = ev.verdict_case(6, 6.0, 9.0, 8.93, 12.0)
    assert kind == "warning" and "über der Wegkosten-Schranke" in text and "höchstens jedem zehnten Fall" in text


# --- Experiment 4 ------------------------------------------------------------------------------------------------------------------------------

def test_blocks_verdict_names_both_sizes():
    small = {"n": 20, "blocks": 1500, "coverage": {"mean": 90.49, "p05": 77.86, "p95": 98.14, "share_below_target": 0.3733}, "width": {"median": 26.86, "min": 1.6, "max": 75.3}}
    large = {"n": 1000, "blocks": 30, "coverage": {"mean": 90.04, "p05": 88.42, "p95": 91.75, "share_below_target": 0.4667}, "width": {"median": 24.22, "min": 19.7, "max": 29.4}}
    kind, text = ev.verdict_blocks(small, large)
    assert kind == "info"
    assert "Mit 20 Kalibrierfällen" in text and "über 1500 Kalibrierungen bei 90,5 %" in text and "von 77,9 bis 98,1 %" in text and "in 37 % der Kalibrierungen unter dem Ziel" in text
    assert "mit 1000 Kalibrierfällen sind es im Mittel 90,0 %" in text and "bei 88,4 bis 91,8 %" in text
    assert "Breite: Median 26,9 (Spanne 1,6 bis 75,3) gegen 24,2 (19,7 bis 29,4) % der Vorhersagekosten" in text


# --- Hilfsfunktionen ---------------------------------------------------------------------------------------------------------------------------

def test_de_formats_with_comma_and_without_negative_zero():
    assert ev.de(2.34, 2) == "2,34" and ev.de(0.0) == "0,0" and ev.de(1234.0, 0) == "1234"
    assert ev.de(-0.04) == "0,0" and ev.de(-0.06) == "-0,1"
